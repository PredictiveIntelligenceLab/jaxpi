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

from utils import get_dataset, parabolic_inflow

import matplotlib.pyplot as plt
import matplotlib.tri as tri


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

    T = 1.0  # final time

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = 1.0  # characteristic velocity
        L_star = 0.1  # characteristic length
        T_star = L_star / U_star  # characteristic time
        Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        T = T / T_star
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        wall_coords = wall_coords / L_star
        cylinder_coords = cylinder_coords / L_star
        coords = coords / L_star

        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star
        p_ref = p_ref / U_star ** 2

    else:
        Re = nu

    # Inflow boundary conditions
    U_max = 1.5  # maximum velocity
    inflow_fn = lambda y: parabolic_inflow(y * L_star, U_max)

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training

    # Initialize model
    model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, 0)), (None, 0, None, None)))
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, 0)), (None, 0, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, 0)), (None, 0, None, None)))
    w_pred_fn = jit(vmap(vmap(model.w_net, (None, None, 0, 0)), (None, 0, None, None)))

    t_coords = jnp.linspace(0, t1, 20)[:-1]

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    w_pred_list = []
    U_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Restore the checkpoint
        ckpt_path = os.path.join('.', 'ckpt', config.wandb.name, 'time_window_{}'.format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        u_pred = u_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
        v_pred = v_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
        w_pred = w_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
        p_pred = p_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        w_pred_list.append(w_pred)
        p_pred_list.append(p_pred)

    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    w_pred = jnp.concatenate(w_pred_list, axis=0)

    # Dimensionalize coordinates and flow field
    if config.nondim == True:
        # Dimensionalize coordinates and flow field
        coords = coords * L_star

        u_ref = u_ref * U_star
        v_ref = v_ref * U_star

        u_pred = u_pred * U_star
        v_pred = v_pred * U_star

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


    # Plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig1 = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.tricontourf(triang, u_pred[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Predicted $u$')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.tricontourf(triang, v_pred[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Predicted $v$')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.tricontourf(triang, p_pred[-1], cmap='jet', levels=100)
    plt.colorbar()
    plt.title('Predicted $p$')
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(save_dir, "ns_unsteady_u.pdf")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)
