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
    u_ref, v_ref, w_ref, t_star, x_star, y_star, nu = get_dataset()

    # Remove the last time step
    u_ref = u_ref[:-1, :]
    v_ref = v_ref[:-1, :]
    w_ref = w_ref[:-1, :]

    u0 = u_ref[0, :]
    v0 = v_ref[0, :]
    w0 = w_ref[0, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Initialize the model
    # Warning: t must be the same as the one used in training, otherwise the prediction will be wrong
    # This is because the input t is scaled inside the model forward pass
    model = models.NavierStokes(config, t, x_star, y_star, u0, v0, w0, nu)

    u_pred_list = []
    v_pred_list = []
    w_pred_list = []
    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
        v_star = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
        w_star = w_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]

        # Restore the checkpoint
        ckpt_path = os.path.join(
            workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
        )
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute the L2 error for the current time window
        u_error, v_error, w_error = model.compute_l2_error(
            params, t, x_star, y_star, u_star, v_star, w_star
        )
        logging.info(
            "Time window: {}, u error: {:.3e}, v error: {:.3e}, w error: {:.3e}".format(
                idx + 1, u_error, v_error, w_error
            )
        )

        u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)
        w_pred = model.w_pred_fn(params, model.t_star, model.x_star, model.y_star)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        w_pred_list.append(w_pred)

    # Get the full prediction
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)
    w_pred = jnp.concatenate(w_pred_list, axis=0)

    u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
    w_error = jnp.linalg.norm(w_pred - w_ref) / jnp.linalg.norm(w_ref)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))
    logging.info("L2 error of the full prediction of w: {:.3e}".format(w_error))

    # Plot the results
    XX, YY = jnp.meshgrid(x_star, y_star, indexing="ij")  # Grid for plotting

    # Plot at the last time step
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, w_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, w_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(w_pred[-1] - w_ref[-1]), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    # Animation
