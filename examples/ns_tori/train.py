import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np
import scipy.io
import ml_collections
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


def train_one_window(config, workdir, model, res_sampler, u_ref, v_ref, w_ref, idx):
    step_offset = idx * config.training.max_steps

    # Logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # jit warm up
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
                log_dict = evaluator(state, batch, u_ref, v_ref, w_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
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

    u_ref, v_ref, w_ref, t_star, x_star, y_star, nu = get_dataset()

    # Initial condition of the first time window
    u0 = u_ref[0, :, :]
    v0 = v_ref[0, :, :]
    w0 = w_ref[0, :, :]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Define the time and space domain
    dt = t[1] - t[0]
    t0 = t[0]
    t1 = (
        t[-1] + 2 * dt
    )  # cover the start point of the next time window, which is t_star[num_time_steps]

    x0 = 0.0
    x1 = 2 * jnp.pi

    y0 = 0.0
    y1 = 2 * jnp.pi

    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Initialize the residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
        v_star = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
        w_star = w_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]

        # Initialize the model
        model = models.NavierStokes(config, t, x_star, y_star, u0, v0, w0, nu)

        # Training the current time window
        model = train_one_window(
            config, workdir, model, res_sampler, u_star, v_star, w_star, idx
        )

        #  Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params

            u0 = model.u0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            v0 = model.v0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            w0 = model.w0_pred_fn(params, t_star[num_time_steps], x_star, y_star)

            del model, state, params
