import os

from absl import logging
import ml_collections

from jax import vmap
import jax.numpy as jnp

import scipy.io
import matplotlib.pyplot as plt

import wandb

from jaxpi.utils import restore_checkpoint
import models

from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, p_ref, rho_ref, t, coords = get_dataset()

    res = 128
    x = jnp.linspace(0, 1, res)
    y = jnp.linspace(0, 1, res)
    xy = jnp.vstack(jnp.meshgrid(x, y)).reshape(2, -1).T
    u0, v0, rho0 = vmap(u0_v0_rho0, (0, 0))(xy[:, 0], xy[:, 1])

    model = models.Euler(config, t, xy, u0, v0, rho0)

    # Restore the checkpoint
    ckpt_path = os.path.join(
        workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = vmap(vmap(model.u_net, (None, None, 0, 0)), (None, 0, None, None))(
        params, t, coords[:, 0], coords[:, 1]
    )
    v_pred = vmap(vmap(model.v_net, (None, None, 0, 0)), (None, 0, None, None))(
        params, t, coords[:, 0], coords[:, 1]
    )
    rho_pred = vmap(vmap(model.rho_net, (None, None, 0, 0)), (None, 0, None, None))(
        params, t, coords[:, 0], coords[:, 1]
    )

    u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
    rho_error = jnp.linalg.norm(rho_pred - rho_ref) / jnp.linalg.norm(rho_ref)

    print("L2 error of the full prediction of u: {:.3e}".format(u_error))
    print("L2 error of the full prediction of v: {:.3e}".format(v_error))
    print("L2 error of the full prediction of rho: {:.3e}".format(rho_error))

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(coords[:, 0], coords[:, 1], c=u_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.scatter(coords[:, 0], coords[:, 1], c=u_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.scatter(
        coords[:, 0], coords[:, 1], c=jnp.abs(u_pred[-1] - u_ref[-1]), cmap="jet"
    )
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(coords[:, 0], coords[:, 1], c=rho_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.scatter(coords[:, 0], coords[:, 1], c=rho_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.scatter(
        coords[:, 0], coords[:, 1], c=jnp.abs(rho_pred[-1] - rho_ref[-1]), cmap="jet"
    )
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()
