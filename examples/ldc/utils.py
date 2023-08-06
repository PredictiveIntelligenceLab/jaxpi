import scipy.io

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def get_dataset(Re):
    data = scipy.io.loadmat("data/ldc_Re{}.mat".format(Re))
    u_ref = data["u"]
    v_ref = data["v"]
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()
    nu = data["nu"]

    return u_ref, v_ref, x_star, y_star, nu


def sample_points_on_square_boundary(num_pts_per_side, eps):
    # Sample points along the top side (x=1 to x=0, y=1)
    top_coords = jnp.linspace(0, 1, num_pts_per_side)
    top = jnp.column_stack((top_coords, jnp.ones_like(top_coords)))

    # Sample points along the bottom side (x=0 to x=1, y=0)
    bottom_coords = jnp.linspace(0, 1, num_pts_per_side)
    bottom = jnp.column_stack((bottom_coords, jnp.zeros_like(bottom_coords)))

    # Sample points along the left side (x=0, y=1 to y=0)
    left_coords = jnp.linspace(0, 1 - eps, num_pts_per_side)
    left = jnp.column_stack((jnp.zeros_like(left_coords), left_coords))

    # Sample points along the right side (x=1, y=0 to y=1)
    right_coords = jnp.linspace(0, 1 - eps, num_pts_per_side)
    right = jnp.column_stack((jnp.ones_like(right_coords), right_coords))

    # Combine the points from all sides
    points = jnp.vstack((top, bottom, left, right))

    return points
