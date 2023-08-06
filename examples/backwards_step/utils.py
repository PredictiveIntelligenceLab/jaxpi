import jax.numpy as jnp

import scipy.io
import pyvista as pv


def inflow_profile(y):
    u = jnp.where(y > 0, 24 * y * (0.5 - y), 0)
    v = jnp.zeros_like(y)
    return u, v


def get_dataset():
    reader = pv.get_reader("./data/flow.vtu")
    data = reader.read()
    u_ref = jnp.array(data["Velocity"][:, 0])
    v_ref = jnp.array(data["Velocity"][:, 1])
    p_ref = jnp.array(data["Pressure"])
    coords = jnp.array(data.points)
    inflow_coords = jnp.concatenate(
        [jnp.zeros(100).reshape(-1, 1), jnp.linspace(-0.5, 0.5, 100).reshape(-1, 1)],
        axis=1,
    )
    outflow_coords = jnp.concatenate(
        [
            15 * jnp.ones(100).reshape(-1, 1),
            jnp.linspace(-0.5, 0.5, 100).reshape(-1, 1),
        ],
        axis=1,
    )
    top_wall_coords = jnp.concatenate(
        [jnp.linspace(0, 15, 1500).reshape(-1, 1), 0.5 * jnp.ones(1500).reshape(-1, 1)],
        axis=1,
    )
    bot_wall_coords = jnp.concatenate(
        [
            jnp.linspace(0, 15, 1500).reshape(-1, 1),
            -0.5 * jnp.ones(1500).reshape(-1, 1),
        ],
        axis=1,
    )
    wall_coords = jnp.concatenate([top_wall_coords, bot_wall_coords], axis=0)
    nu = 1.0 / 800.0

    return (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        nu,
    )
