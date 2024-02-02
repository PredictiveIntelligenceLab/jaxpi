import scipy.io


def get_dataset(fraction):
    data = scipy.io.loadmat("data/grey_scott.mat")

    u_ref = data["usol"]
    v_ref = data["vsol"]

    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()

    start_time_step = int(fraction[0] * len(t_star))
    end_time_step = int(fraction[1] * len(t_star))

    u_ref = u_ref[start_time_step:end_time_step, :, :]
    v_ref = v_ref[start_time_step:end_time_step, :, :]
    t_star = t_star[: end_time_step - start_time_step]

    b1 = data["b1"].flatten()[0]
    b2 = data["b2"].flatten()[0]

    c1 = data["c1"].flatten()[0]
    c2 = data["c2"].flatten()[0]

    eps1 = data["ep1"].flatten()[0]
    eps2 = data["ep2"].flatten()[0]

    return u_ref, v_ref, t_star, x_star, y_star, b1, b2, c1, c2, eps1, eps2
