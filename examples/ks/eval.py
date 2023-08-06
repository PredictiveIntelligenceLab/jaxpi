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
    u_ref, t_star, x_star = get_dataset()

    u_ref = u_ref[:-1, :]
    u0 = u_ref[0, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Initialize the model
    # Warning: t must be the same as the one used in training, otherwise the prediction will be wrong
    # This is because the input t is scaled inside the model forward pass
    model = models.KS(config, u0, t, x_star)

    u_pred_list = []
    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Restore the checkpoint
        ckpt_path = os.path.join(
            workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
        )
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute the L2 error for the current time window
        l2_error = model.compute_l2_error(params, u)
        logging.info("Time window: {}, L2 error: {:.3e}".format(idx + 1, l2_error))

        u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
        u_pred_list.append(u_pred)

    # Get the full prediction
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    l2_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    logging.info("L2 error of the full prediction: {:.3e}".format(l2_error))

    # Plot the results
    TT, XX = jnp.meshgrid(t_star[:-1], x_star, indexing="ij")  # Grid for plotting

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_path)
    fig_path = os.path.join(save_dir, "ks.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
