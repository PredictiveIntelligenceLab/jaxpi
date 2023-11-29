import os
import time

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


def train_one_window(config, workdir, model, res_sampler, u_ref, idx):
    logger = Logger()

    evaluator = models.KSEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

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
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Save model checkpoint
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
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Get the reference solution
    u_ref, t_star, x_star = get_dataset(config.time_fraction)
    u0 = u_ref[0, :]  # initial condition of the first time window

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Define the time and space domain
    dt = t[1] - t[0]
    t0 = t[0]
    t1 = (
        t[-1] + 2 * dt
    )  # cover the start point of the next time window, which is t_star[num_time_steps]

    x0 = x_star[0]
    x1 = x_star[-1]
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize the residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size))

    for idx in range(config.training.num_time_windows):
        print("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Initialize the model
        model = models.KS(config, u0, t, x_star)

        # Training the current time window
        model = train_one_window(config, workdir, model, res_sampler, u, idx)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            u0 = vmap(model.u_net, (None, None, 0))(
                params, t_star[num_time_steps], x_star
            )

            del model, state, params
