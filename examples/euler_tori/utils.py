import jax.numpy as jnp
import scipy.io


def get_dataset():
    data = scipy.io.loadmat("data/euler.mat")
    u_ref = data["u"]
    v_ref = data["v"]
    p_ref = data["p"]
    rho_ref = data["rho"]
    coords = data["coords"]
    t = data["t"].flatten()
    return u_ref, v_ref, p_ref, rho_ref, t, coords


def u0_v0_rho0(x, y):
    z = jnp.array(
        [
            jnp.cos(2 * jnp.pi * x),
            jnp.sin(2 * jnp.pi * x),
            jnp.cos(2 * jnp.pi * y),
            jnp.sin(2 * jnp.pi * y),
        ]
    )
    u = jnp.exp(z[3])
    v = 0.5 * jnp.exp(z[1])
    rho = 1 + (z[1] + z[3]) ** 2

    return u, v, rho
