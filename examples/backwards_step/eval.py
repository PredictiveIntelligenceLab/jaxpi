from functools import partial
import time
import os

from absl import logging

from flax.training import checkpoints

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.tree_util import tree_map

import scipy.io
import ml_collections

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import wandb

import models

from jaxpi.utils import restore_checkpoint

from utils import get_dataset, inflow_profile


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        nu,
    ) = get_dataset()

    u_inflow, _ = inflow_profile(inflow_coords[:, 1])

    # Nondimensionalization
    if config.nondim == True:
        raise NotImplementedError

    else:
        U_star = 1.0
        L_star = 1.0
        Re = 1 / nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        Re,
    )

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Predict
    u_pred = model.u_pred_fn(params, coords[:, 0], coords[:, 1])
    v_pred = model.v_pred_fn(params, coords[:, 0], coords[:, 1])

    u_error = jnp.sqrt(jnp.mean((u_ref - u_pred) ** 2)) / jnp.sqrt(jnp.mean(u_ref**2))
    v_error = jnp.sqrt(jnp.mean((v_ref - v_pred) ** 2)) / jnp.sqrt(jnp.mean(v_ref**2))

    print("l2_error of u: {:.4e}".format(u_error))
    print("l2_error of v: {:.4e}".format(v_error))

    # Plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if config.nondim == True:
        # Dimensionalize coordinates and flow field
        coords = coords * L_star

        u_ref = u_ref * U_star
        v_ref = v_ref * U_star

        u_pred = u_pred * U_star
        v_pred = v_pred * U_star

    # Triangulation
    x = coords[:, 0]
    y = coords[:, 1]
    triang = tri.Triangulation(x, y)

    fig1 = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.tricontourf(triang, u_ref, cmap="jet", levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.tricontourf(triang, u_pred, cmap="jet", levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted u(x, y)")
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.tricontourf(triang, jnp.abs(u_ref - u_pred), cmap="jet", levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "ns_steady_u.pdf")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)

    fig2 = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.tricontourf(triang, v_ref, cmap="jet", levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.tricontourf(triang, v_pred, cmap="jet", levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted u(x, y)")
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.tricontourf(triang, jnp.abs(v_ref - v_pred), cmap="jet", levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "ns_steady_v.pdf")
    fig2.savefig(save_path, bbox_inches="tight", dpi=300)
