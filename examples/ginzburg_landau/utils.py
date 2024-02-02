import scipy.io
import jax.numpy as jnp


def get_dataset(fraction):
    # Load data from the file
    data = scipy.io.loadmat("data/ginzburg_landau_square.mat")

    u_ref = data["usol"]
    v_ref = data["vsol"]

    # PDE parameters
    eps = data["eps"].flatten()[0]
    k = data["k"].flatten()[0]

    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()

    start_time_step = int(fraction[0] * len(t_star))
    end_time_step = int(fraction[1] * len(t_star))
    num_time_steps = end_time_step - start_time_step

    u_ref = u_ref[start_time_step:end_time_step, :, :]
    v_ref = v_ref[start_time_step:end_time_step, :, :]
    t_star = t_star[:num_time_steps]

    # Return the processed data
    return u_ref, v_ref, t_star, x_star, y_star, eps, k
