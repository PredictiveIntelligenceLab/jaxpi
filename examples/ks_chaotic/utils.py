import scipy.io


def get_dataset(fraction):
    # Load data
    data = scipy.io.loadmat("data/ks_chaotic.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    # Only use a fraction of the data
    num_time_steps = int(fraction * len(t_star))
    t_star = t_star[:num_time_steps]
    u_ref = u_ref[:num_time_steps, :]

    return u_ref, t_star, x_star
