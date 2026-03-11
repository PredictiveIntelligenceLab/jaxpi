import scipy.io
import os


def get_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "wave.mat")
    data = scipy.io.loadmat(data_path)
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    return u_ref, t_star, x_star
