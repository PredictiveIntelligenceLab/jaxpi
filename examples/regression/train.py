import os
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, random, jit, pmap
from jax.tree_util import tree_map

import ml_collections

# from absl import logging
import wandb

from jaxpi.samplers import BaseSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint
import models
from utils import get_dataset


class MySampler(BaseSampler):
    def __init__(self, x, y, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.x = x
        self.y = y

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(key, self.x.shape[0], shape=(self.batch_size,))

        x = self.x[idx]
        y = self.y[idx]
        batch = (x, y)

        return batch


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Problem setup
    L = 1.0  # length of the domain
    n_x = 256  # number of spatial points
    freq = 1.0  # frequency of the sine wave

    # Get  dataset
    x_star, u_star = get_dataset(L, n_x, freq)

    # Initialize model
    model = models.Regression(config, x_star, u_star)
    # Initialize residual sampler
    sampler = iter(
        MySampler(x_star, u_star, batch_size=n_x)
    )  # full batch gradient descent

    evaluator = models.RegressionEvaluator(config, model)

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(sampler)

        model.state = model.step(model.state, batch)
        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch)
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
