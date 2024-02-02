import os
import time

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import scipy.io

import ml_collections
import wandb


from jaxpi.archs import PeriodEmbs, Embedding
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]

    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Define residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Physics-informed initialization
    if config.use_pi_init:
        logger.info("Use physics-informed initialization...")

        model = models.KDV(config, u0, t_star, x_star)
        state = jax.device_get(tree_map(lambda x: x[0], model.state))
        params = state.params

        # Initialization data source
        if config.pi_init_type == "linear_pde":
            # load data
            data = scipy.io.loadmat("data/kdv_linear.mat")
            # downsample the grid and data
            u = data["usol"][::10]
            t = data["t"].flatten()[::10]
            x = data["x"].flatten()

            tt, xx = jnp.meshgrid(t, x, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])

        elif config.pi_init_type == "initial_condition":
            t = t_star[::10]
            x = x_star
            u = u0

            tt, xx = jnp.meshgrid(t, x, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
            u = jnp.tile(u.flatten(), (t.shape[0], 1))

        feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

        coeffs, residuals, rank, s = jnp.linalg.lstsq(
            feat_matrix, u.flatten(), rcond=None
        )  # seems to work better!!!
        print("least square residuals: ", residuals)

        config.arch.pi_init = coeffs.reshape(-1, 1)

        del model, state, params

    # Initialize model
    model = models.KDV(config, u0, t_star, x_star)

    # Initialize evaluator
    evaluator = models.KDVEvaluator(config, model)

    logger.info("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)

        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            # Avoid computing gradients for the first few steps
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, config.wandb.name, "ckpt")
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
