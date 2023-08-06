import jax.numpy as jnp


def get_dataset():
    data = jnp.load("data/ns_tori.npy", allow_pickle=True).item()
    u_ref = data["u"]
    v_ref = data["v"]
    w_ref = data["w"]

    t = data["t"].flatten()
    x = data["x"].flatten()
    y = data["y"].flatten()
    nu = data["viscosity"]

    return u_ref, v_ref, w_ref, t, x, y, nu
