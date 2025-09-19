import time
import os
from functools import partial

from absl import logging

import jax
from jax import random, pmap
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np
import scipy.io
import ml_collections
import wandb

from jaxpi.samplers import BaseSampler, SpaceSampler, TimeSpaceSampler, UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


class ICSampler(SpaceSampler):
    def __init__(self, u, v, p, temp, coords, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(coords, batch_size, rng_key)

        self.u = u
        self.v = v
        self.p = p
        self.temp = temp

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))

        coords_batch = self.coords[idx, :]
        u_batch = self.u[idx]
        v_batch = self.v[idx]
        p_batch = self.p[idx]
        temp_batch = self.temp[idx]

        batch = (coords_batch, u_batch, v_batch, p_batch, temp_batch)

        return batch


class BCsSampler(BaseSampler):
    def __init__(self, dom,  batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)

        self.dom = dom

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        subkeys = random.split(key, 3)

        t = random.uniform(subkeys[0], (self.batch_size // 2,), minval=self.dom[0][0], maxval=self.dom[0][1])
        x = random.uniform(subkeys[1], (self.batch_size // 2,), minval=self.dom[1][0], maxval=self.dom[1][1])

        bc1_batch = jnp.stack([t, x, jnp.zeros_like(x)]).T
        bc2_batch = jnp.stack([t, x, 2 * jnp.ones_like(x)]).T

        bc_batch = jnp.vstack([bc1_batch, bc2_batch])

        return bc_batch


def train_one_window(config, workdir, model, samplers, t, coords, u_ref, v_ref, temp_ref, idx):
    step_offset = idx * config.training.max_steps

    # Logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

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
                state = tree_map(lambda x: x[0], model.state)
                batch = tree_map(lambda x: x[0], batch)
                log_dict = evaluator(state, batch, t, coords, u_ref, v_ref, temp_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    velocity, pressure, temperature, t_star, coords, alpha1, alpha2, alpha3, alpha4, Ra, Pr, Ge = get_dataset()

    # Initial condition of the first time window
    u0 = velocity[0, :, 0]
    v0 = velocity[0, :, 1]
    p0 = pressure[0]
    temp0 = temperature[0]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Define the time and space domain
    dt = t[1] - t[0]
    t0 = t[0]
    t1 = t[-1] + 1.1 * dt

    x0 = 0.0
    x1 = 1.0

    y0 = 0.0
    y1 = 2.0

    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u_star = velocity[num_time_steps * idx: num_time_steps * (idx + 1), :, 0]
        v_star = velocity[num_time_steps * idx: num_time_steps * (idx + 1), :, 1]
        temp_star = temperature[num_time_steps * idx: num_time_steps * (idx + 1)]

        velocity_scale = jnp.max(jnp.sqrt(u0 ** 2 + v0 ** 2))
        print("Velocity scale: ", velocity_scale)

        # Initialize the model
        model = models.NavierStokes(config, alpha1, alpha2, alpha3, alpha4, t, coords, u0, v0, temp0, p0)

        # Initialize the samplers
        ics_sampler = ICSampler(u0, v0, p0, temp0, coords, config.training.batch_size_per_device * 2)
        bcs_sampler = BCsSampler(dom, config.training.batch_size_per_device * 2)
        res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

        samplers = {
            "ics": iter(ics_sampler),
            "bcs": iter(bcs_sampler),
            "res": iter(res_sampler),
        }

        # Training the current time window
        model = train_one_window(
            config, workdir, model, samplers, t, coords,  u_star, v_star, temp_star, idx
        )

        #  Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params

            u0 = model.u_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
            v0 = model.v_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
            p0 = model.p_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
            temp0 = model.temp_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])

            del model, state, params
