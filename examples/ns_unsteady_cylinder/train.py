import functools
from functools import partial
import time
import os

from absl import logging

import jax

import jax.numpy as jnp
from jax import random, vmap, pmap, local_device_count
from jax.tree_util import tree_map

import matplotlib.pyplot as plt

import numpy as np
import scipy.io
import ml_collections

import wandb

import models

from jaxpi.samplers import BaseSampler, SpaceSampler, TimeSpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

from utils import get_dataset, get_fine_mesh, parabolic_inflow


class ICSampler(SpaceSampler):
    def __init__(self, u, v, p, coords, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(coords, batch_size, rng_key)

        self.u = u
        self.v = v
        self.p = p

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))

        coords_batch = self.coords[idx, :]

        u_batch = self.u[idx]
        v_batch = self.v[idx]
        p_batch = self.p[idx]

        batch = (coords_batch, u_batch, v_batch, p_batch)

        return batch


class ResSampler(BaseSampler):
    def __init__(
        self,
        temporal_dom,
        coarse_coords,
        fine_coords,
        batch_size,
        rng_key=random.PRNGKey(1234),
    ):
        super().__init__(batch_size, rng_key)

        self.temporal_dom = temporal_dom

        self.coarse_coords = coarse_coords
        self.fine_coords = fine_coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        subkeys = random.split(key, 4)

        temporal_batch = random.uniform(
            subkeys[0],
            shape=(2 * self.batch_size, 1),
            minval=self.temporal_dom[0],
            maxval=self.temporal_dom[1],
        )

        coarse_idx = random.choice(
            subkeys[1],
            self.coarse_coords.shape[0],
            shape=(self.batch_size,),
            replace=True,
        )
        fine_idx = random.choice(
            subkeys[2],
            self.fine_coords.shape[0],
            shape=(self.batch_size,),
            replace=True,
        )

        coarse_spatial_batch = self.coarse_coords[coarse_idx, :]
        fine_spatial_batch = self.fine_coords[fine_idx, :]
        spatial_batch = jnp.vstack([coarse_spatial_batch, fine_spatial_batch])
        spatial_batch = random.permutation(
            subkeys[3], spatial_batch
        )  # mix the coarse and fine coordinates

        batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)

        return batch


def train_one_window(config, workdir, model, samplers, idx):
    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    step_offset = idx * config.training.max_steps

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        # Sample mini-batch
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)

        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(
                    workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
                )
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Get dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cyl_coords,
        nu,
    ) = get_dataset()

    (
        fine_coords,
        fine_coords_near_cyl,
    ) = get_fine_mesh()  # finer mesh for evaluating PDE residuals

    noslip_coords = jnp.vstack((wall_coords, cyl_coords))

    # T = 1.0  # final time of simulation

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = 1.0  # characteristic velocity
        L_star = 0.1  # characteristic length
        T_star = L_star / U_star  # characteristic time
        Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        # T = T / T_star
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        noslip_coords = noslip_coords / L_star

        coords = coords / L_star
        fine_coords = fine_coords / L_star
        fine_coords_near_cyl = fine_coords_near_cyl / L_star

        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star
        p_ref = p_ref / U_star**2

    else:
        U_star = 1.0
        L_star = 1.0
        T_star = 1.0
        Re = 1 / nu

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)])

    # Inflow boundary conditions
    U_max = 1.5  # maximum velocity
    inflow_fn = lambda y: parabolic_inflow(y * L_star, U_max)

    # Set initial condition
    # Use the last time step of a coarse numerical solution as the initial condition
    u0 = u_ref[-1, :]
    v0 = v_ref[-1, :]
    p0 = p_ref[-1, :]

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        # Initialize Sampler
        keys = random.split(random.PRNGKey(0), 5)
        ic_sampler = iter(
            ICSampler(
                u0, v0, p0, coords, config.training.ic_batch_size, rng_key=keys[0]
            )
        )
        inflow_sampler = iter(
            TimeSpaceSampler(
                temporal_dom,
                inflow_coords,
                config.training.inflow_batch_size,
                rng_key=keys[1],
            )
        )
        outflow_sampler = iter(
            TimeSpaceSampler(
                temporal_dom,
                outflow_coords,
                config.training.outflow_batch_size,
                rng_key=keys[2],
            )
        )
        noslip_sampler = iter(
            TimeSpaceSampler(
                temporal_dom,
                noslip_coords,
                config.training.noslip_batch_size,
                rng_key=keys[3],
            )
        )

        res_sampler = iter(
            ResSampler(
                temporal_dom,
                fine_coords,
                fine_coords,
                config.training.res_batch_size,
                rng_key=keys[4],
            )
        )

        samplers = {
            "ic": ic_sampler,
            "inflow": inflow_sampler,
            "outflow": outflow_sampler,
            "noslip": noslip_sampler,
            "res": res_sampler,
        }

        # Initialize model
        model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)

        # Train model for the current time window
        model = train_one_window(config, workdir, model, samplers, idx)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            u0 = vmap(model.u_net, (None, None, 0, 0))(
                params, t1, coords[:, 0], coords[:, 1]
            )
            v0 = vmap(model.v_net, (None, None, 0, 0))(
                params, t1, coords[:, 0], coords[:, 1]
            )
            p0 = vmap(model.p_net, (None, None, 0, 0))(
                params, t1, coords[:, 0], coords[:, 1]
            )

            del model, state, params
