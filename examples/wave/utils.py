import jax.numpy as jnp
from jax import vmap


# def get_dataset(T=2.0, L=2 * jnp.pi, c=50, n_t=200, n_x=128):
#     t_star = jnp.linspace(0, T, n_t)
#     x_star = jnp.linspace(0, L, n_x)
#
#     u_exact_fn = lambda t, x: jnp.sin(jnp.mod(x - c * t, L))
#
#     u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)
#
#     return u_exact, t_star, x_star



def get_dataset(T=1.0, L=1.0, a=0.5, c=2, n_t=200, n_x=128):
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    def u_fn(t, x):
        return jnp.sin(jnp.pi * x) * jnp.cos(c * jnp.pi * t) + \
            a * jnp.sin(2 * c * jnp.pi * x) * jnp.cos(4 * c * jnp.pi * t)

    # def u_t_fn(x, t):
    #     u_t = -  c * jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(c * jnp.pi * t) - \
    #           a * 4 * c * jnp.pi * jnp.sin(2 * c * jnp.pi * x) * jnp.sin(4 * c * jnp.pi * t)
    #     return u_t
    #
    # def u_tt_fn(x, t):
    #     u_tt = -(c * jnp.pi) ** 2 * jnp.sin(jnp.pi * x) * jnp.cos(c * jnp.pi * t) - \
    #            a * (4 * c * jnp.pi) ** 2 * jnp.sin(2 * c * jnp.pi * x) * jnp.cos(4 * c * jnp.pi * t)
    #     return u_tt
    #
    # def u_xx_fn(x, t):
    #     u_xx = - jnp.pi ** 2 * jnp.sin(jnp.pi * x) * jnp.cos(c * jnp.pi * t) - \
    #            a * (2 * c * jnp.pi) ** 2 * jnp.sin(2 * c * jnp.pi * x) * jnp.cos(4 * c * jnp.pi * t)
    #     return u_xx
    #
    # def f(x, t):
    #     '''
    #     the right-hand side of the PDE should be zero, i.e., f(x, t) = 0
    #     '''
    #     return u_tt_fn(x, t) - c ** 2 * u_xx_fn(x, t)

    u_exact = vmap(vmap(u_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, t_star, x_star