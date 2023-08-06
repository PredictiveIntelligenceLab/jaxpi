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

import wandb

import models

from jaxpi.utils import restore_checkpoint

import matplotlib.pyplot as plt
import matplotlib.tri as tri


def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    v = jnp.zeros_like(y)
    return u, v


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
        cylinder_coords,
        nu,
    ) = get_dataset()

    U_max = 0.3  # maximum velocity
    u_inflow, _ = parabolic_inflow(inflow_coords[:, 1], U_max)

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = 0.2  # characteristic velocity
        L_star = 0.1  # characteristic length
        Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        wall_coords = wall_coords / L_star
        cylinder_coords = cylinder_coords / L_star
        coords = coords / L_star

        # Nondimensionalize flow field
        u_inflow = u_inflow / U_star
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star

    else:
        Re = nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
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

    # Mask the triangles inside the cylinder
    center = (0.2, 0.2)
    radius = 0.05

    x_tri = x[triang.triangles].mean(axis=1)
    y_tri = y[triang.triangles].mean(axis=1)
    dist_from_center = jnp.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
    triang.set_mask(dist_from_center < radius)

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
