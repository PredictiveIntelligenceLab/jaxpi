import os

from absl import logging
import ml_collections

import jax.numpy as jnp

import scipy.io
import matplotlib.pyplot as plt

import wandb

from jaxpi.utils import restore_checkpoint
import models

from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, t_star, x_star, y_star, eps, k = get_dataset(config.time_fraction)

    # Remove the last time step
    u_ref = u_ref[:-1, :]
    v_ref = v_ref[:-1, :]

    u0 = u_ref[0, :, :]
    v0 = v_ref[0, :, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.GinzburgLandau(config, t, x_star, y_star, u0, v0, eps, k)

    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        v_star = v_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]

        # Restore the checkpoint
        ckpt_path = os.path.join(
            workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
            )
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute the L2 error for the current time window
        u_error, v_error = model.compute_l2_error(
            params, t, x_star, y_star, u_star, v_star
            )
        print(
            "Time window: {}, u error: {:.3e}, v error: {:.3e}".format(
                idx + 1, u_error, v_error
                )
            )

        u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))

    # Plot the results
    XX, YY = jnp.meshgrid(x_star, y_star, indexing="ij")  # Grid for plotting

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Plot at the last time step
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, u_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reference u")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, u_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted u")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(u_ref[-1] - u_pred[-1]), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "gl_u.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # Plot at the last time step
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, v_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reference v")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, v_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted v")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(v_ref[-1] - v_pred[-1]), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "gl_v.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
