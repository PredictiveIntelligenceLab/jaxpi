import jax.numpy as jnp
from jax import random, vmap, jit


def get_dataset(Re=1e4, time_fraction=1.0):
    data = jnp.load("data/kolmogorov_flow_Re10000.npy", allow_pickle=True).item()
    w_ref = jnp.array(data["vorticity"])
    velocity = jnp.array(data["velocity"])

    u_ref = velocity[..., 0]
    v_ref = velocity[..., 1]

    t = jnp.array(data["t"]).flatten()
    t = t - t[0]

    # Truncate data
    num_steps = int(time_fraction * t.shape[0])
    u_ref = u_ref[:num_steps]
    v_ref = v_ref[:num_steps]
    w_ref = w_ref[:num_steps]
    t = t[:num_steps]

    print("t.shape", t.shape, "u_ref.shape", u_ref.shape, "v_ref.shape", v_ref.shape, "w_ref.shape", w_ref.shape)

    coords = jnp.array(data["coords"])
    nu = data["nu"]

    # ONLY FOR TEST!!! REMOVE
    nu = 1 / Re

    return u_ref, v_ref, w_ref, t, coords, nu


@jit
def relative_l2_error_periodic(field1, field2):
    """
    Compute relative L2 error with optimal periodic shift
    Returns: min ||field1 - shifted_field2||_2 / ||field1||_2
    """
    # Compute L2 norm of reference field
    norm_ref = jnp.sqrt(jnp.sum(field1 ** 2))

    shape = field1.shape
    min_error = jnp.inf

    def compute_error_for_shift(shift_x, shift_y):
        shifted = jnp.roll(jnp.roll(field2, shift_x, axis=0), shift_y, axis=1)
        diff = field1 - shifted
        return jnp.sqrt(jnp.sum(diff ** 2)) / norm_ref

    # Vectorize over all possible shifts
    shifts_x, shifts_y = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))
    errors = vmap(lambda x, y: compute_error_for_shift(x, y))(
        shifts_x.ravel(), shifts_y.ravel()
    )

    return jnp.min(errors)
