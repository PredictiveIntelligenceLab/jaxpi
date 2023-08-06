import jax.numpy as jnp

import scipy.io


def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    v = jnp.zeros_like(y)
    return u, v


def get_dataset():
    data = jnp.load("data/ns_steady.npy", allow_pickle=True).item()
    u_ref = jnp.array(data["u"])
    v_ref = jnp.array(data["v"])
    p_ref = jnp.array(data["p"])
    coords = jnp.array(data["coords"])
    inflow_coords = jnp.array(data["inflow_coords"])
    outflow_coords = jnp.array(data["outflow_coords"])
    wall_coords = jnp.array(data["wall_coords"])
    cylinder_coords = jnp.array(data["cylinder_coords"])
    nu = jnp.array(data["nu"])

    return (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    )
