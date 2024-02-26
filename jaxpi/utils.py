import os

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, tree_map
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

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
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Save the checkpoint.
    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        state = jax.device_get(tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep)


def restore_checkpoint(state, workdir, step=None):
    # check if passed state is in a sharded state
    # if so, reduce to a single device sharding

    if isinstance(
        jax.tree_map(lambda x: jnp.array(x).sharding, jax.tree_leaves(state.params))[0],
        jax.sharding.PmapSharding,
    ):
        state = jax.tree_map(lambda x: x[0], state)

    # ensuring that we're in a single device setting
    assert isinstance(
        jax.tree_map(lambda x: jnp.array(x).sharding, jax.tree_leaves(state.params))[0],
        jax.sharding.SingleDeviceSharding,
    )

    state = checkpoints.restore_checkpoint(workdir, state, step=step)
    return state
