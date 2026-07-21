import os

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.flatten_util import ravel_pytree

from flax import jax_utils
from flax.training import checkpoints


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


@partial(jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = grad(apply_fn, argnums=0)(params, *args)
    J, _ = ravel_pytree(J)
    return J


@partial(jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = jacobian_fn(apply_fn, params, *args)
    K = jnp.dot(J, J)
    return K


def save_checkpoint(state, workdir, keep=5, name=None):
    workdir = os.path.abspath(workdir)

    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Save the checkpoint.
    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        state = jax.device_get(jax_utils.unreplicate(state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep)


def restore_checkpoint(state, workdir, step=None):
    workdir = os.path.abspath(workdir)

    # Model states are replicated for pmap. Check the scalar step rather than
    # relying on JAX sharding implementation classes, which change across JAX
    # versions.
    if jnp.ndim(state.step) > 0:
        state = jax_utils.unreplicate(state)

    state = checkpoints.restore_checkpoint(workdir, state, step=step)
    return state
