import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import vmap, jacrev
from jax.tree_util import tree_map

from flax import jax_utils

import ml_collections
import matplotlib.pyplot as plt

import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


def train_curriculum(config, workdir, model, step_offset, max_steps, Re):
    # Get dataset
    u_ref, v_ref, x_star, y_star, nu = get_dataset(Re)
    U_ref = jnp.sqrt(u_ref**2 + v_ref**2)

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Define domain
    dom = jnp.array([[x0, x1], [y0, y1]])

    # Initialize  residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size))

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    # Update  viscosity
    nu = 1 / Re

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch, nu)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch, nu)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, x_star, y_star, U_ref, nu)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "Re{}".format(Re))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)


    # Get step offset
    step_offset = step + step_offset

    return model, step_offset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize model
    model = models.NavierStokes2D(config)

    # Curriculum training
    step_offset = 0

    assert len(config.training.max_steps) == len(config.training.Re)
    num_Re = len(config.training.Re)

    for idx in range(num_Re):
        # Set Re and maximum number of training steps
        Re = config.training.Re[idx]
        max_steps = config.training.max_steps[idx]
        print("Training for Re = {}".format(Re))
        model, step_offset = train_curriculum(
            config, workdir, model, step_offset, max_steps, Re
        )

    return model
