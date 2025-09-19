import jax.numpy as jnp


def get_dataset():
    data = jnp.load("data/rayleigh_taylor_high_Ra.npy", allow_pickle=True).item()

    start_idx = 5

    velocity = jnp.array(data["velocity"])[start_idx:]
    pressure = jnp.array(data["pressure"])[start_idx:]
    temperature = jnp.array(data["temperature"])[start_idx:]

    t = jnp.array(data["t"])[start_idx:]
    t = t - t[0]
    coords = jnp.array(data["coords"])

    # parameters
    alpha1 = data["alpha1"]
    alpha2 = data["alpha2"]
    alpha3 = data["alpha3"]
    alpha4 = data["alpha4"]

    Ra = data["Ra"]
    Pr = data["Pr"]
    Ge = data["Ge"]

    return velocity, pressure, temperature, t, coords, alpha1, alpha2, alpha3, alpha4, Ra, Pr, Ge

