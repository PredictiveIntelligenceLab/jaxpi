import time
import os

from absl import logging

import jax
from jax import vmap
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np
import ml_collections
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset, u0_v0_rho0


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    logger = Logger()

    u_ref, v_ref, p_ref, rho_ref, t, coords = get_dataset()

    t_star = jnp.linspace(0, 0.35, 100)
    x_star = jnp.linspace(0, 1, 100)
    y_star = jnp.linspace(0, 1, 100)

    dt = t_star[1] - t_star[0]
    t0 = t_star[0]
    t1 = t_star[-1] + 2 * dt  # Cover the start point of the next time window

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Set initial condition
    res = 128
    x = jnp.linspace(0, 1, res)
    y = jnp.linspace(0, 1, res)
    xy = jnp.vstack(jnp.meshgrid(x, y)).reshape(2, -1).T
    u0, v0, rho0 = vmap(u0_v0_rho0, (0, 0))(xy[:, 0], xy[:, 1])

    model = models.Euler(config, t, xy, u0, v0, rho0)

    evaluator = models.EulerEvaluator(config, model)

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, t, coords, u_ref, v_ref, rho_ref)
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
