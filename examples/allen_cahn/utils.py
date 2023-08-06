import scipy.io


def get_dataset():
    data = scipy.io.loadmat("data/allen_cahn.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    return u_ref, t_star, x_star
