import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import vmap, jacrev
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import SpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset, inflow_profile


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        nu,
    ) = get_dataset()

    u_inflow, _ = inflow_profile(inflow_coords[:, 1])

    # Nondimensionalization
    if config.nondim == True:
        raise NotImplementedError

    else:
        U_star = 1.0
        L_star = 1.0
        Re = 1 / nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        Re,
    )

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize  residual sampler
    res_sampler = iter(SpaceSampler(coords, config.training.batch_size_per_device))

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
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
                log_dict = evaluator(state, batch, coords, u_ref, v_ref)
                wandb.log(log_dict, step)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model
