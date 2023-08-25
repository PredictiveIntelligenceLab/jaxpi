import os
import time

import jax
from jax import random, pmap, local_device_count

import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    u_exact_fn = lambda x: (jnp.cos(jnp.pi * x))

    x0 = -1.0
    x1 = 1.0
    x_star = jnp.linspace(x0, x1, 128)
    u_ref = u_exact_fn(x_star)

    # full batch gradient descent, so we tile the data to each device
    num_devices = local_device_count()
    batch = jnp.tile(x_star, (num_devices, 1))

    # Initialize model
    model = models.PINN(config, x_star)

    # Initialize evaluator
    evaluator = models.PINNEvaluator(config, model)

    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                log_dict = evaluator(state, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
