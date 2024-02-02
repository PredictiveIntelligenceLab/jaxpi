import jax.numpy as jnp
from jax import vmap


def get_dataset(L=1.0, n_x=512, freq=10.0):
    x_star = jnp.linspace(0, L, n_x)

    u_star = jnp.sin(freq * 2 * jnp.pi * x_star)

    return x_star, u_star
