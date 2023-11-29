import os

import jax.numpy as jnp
from jax import vmap

import scipy.io
import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset

# Ghia et al. (1982) data
y_Ghia = [
    0.0000,
    0.0547,
    0.0625,
    0.0703,
    0.1016,
    0.1719,
    0.2813,
    0.4531,
    0.5000,
    0.6172,
    0.7344,
    0.8516,
    0.9531,
    0.9609,
    0.9688,
    0.9766,
    1.0000,
]

u_Ghia = {
    100: [
        0.00000,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.00000,
    ],
    400: [
        0.00000,
        -0.08186,
        -0.09266,
        -0.10338,
        -0.14612,
        -0.24299,
        -0.32726,
        -0.17119,
        -0.11477,
        0.02135,
        0.16256,
        0.29093,
        0.55892,
        0.61756,
        0.68439,
        0.75837,
        1.00000,
    ],
    1000: [
        0.00000,
        -0.18109,
        -0.20196,
        -0.22220,
        -0.29730,
        -0.38289,
        -0.27805,
        -0.10648,
        -0.06080,
        0.05702,
        0.18719,
        0.33304,
        0.46604,
        0.51117,
        0.57492,
        0.65928,
        1.00000,
    ],
    3200: [
        0.00000,
        -0.32407,
        -0.35344,
        -0.37827,
        -0.41933,
        -0.34323,
        -0.24427,
        -0.86636,
        -0.04272,
        0.07156,
        0.19791,
        0.34682,
        0.46101,
        0.46547,
        0.48296,
        0.53236,
        1.00000,
    ],
    5000: [
        0.00000,
        -0.41165,
        -0.42901,
        -0.43643,
        -0.40435,
        -0.33050,
        -0.22855,
        -0.07404,
        -0.03039,
        0.08183,
        0.20087,
        0.33556,
        0.46036,
        0.45992,
        0.46120,
        0.48223,
        1.00000,
    ],
    7500: [
        0.00000,
        -0.43154,
        -0.43590,
        -0.43025,
        -0.38324,
        -0.32393,
        -0.23176,
        -0.07503,
        -0.03800,
        0.08342,
        0.20591,
        0.34228,
        0.47167,
        0.47323,
        0.47048,
        0.47244,
        1.00000,
    ],
    10000: [
        0.00000,
        -0.42735,
        -0.42537,
        -0.41657,
        -0.38000,
        -0.32709,
        -0.23186,
        -0.07540,
        0.03111,
        0.08344,
        0.20673,
        0.34635,
        0.47804,
        0.48070,
        0.47783,
        0.47221,
        1.00000,
    ],
}

x_Ghia = [
    0.0000,
    0.0625,
    0.0703,
    0.0781,
    0.0938,
    0.1563,
    0.2266,
    0.2344,
    0.5000,
    0.8047,
    0.8594,
    0.9063,
    0.9453,
    0.9531,
    0.9609,
    0.9688,
    1.0000,
]
v_Ghia = {
    100: [
        0.00000,
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
        0.00000,
    ],
    400: [
        0.00000,
        0.18360,
        0.19713,
        0.20920,
        0.22965,
        0.28124,
        0.30203,
        0.30174,
        0.05186,
        -0.38598,
        -0.44993,
        -0.23827,
        -0.22847,
        -0.19254,
        -0.15663,
        -0.12146,
        0.00000,
    ],
    1000: [
        0.00000,
        0.27485,
        0.29012,
        0.30353,
        0.32627,
        0.37095,
        0.33075,
        0.32235,
        0.02526,
        -0.31966,
        -0.42665,
        -0.51550,
        -0.39188,
        -0.33714,
        -0.27669,
        -0.21388,
        0.00000,
    ],
    3200: [
        0.00000,
        0.39560,
        0.40917,
        0.41906,
        0.42768,
        0.37119,
        0.29030,
        0.28188,
        0.00999,
        -0.31184,
        -0.37401,
        -0.44307,
        -0.54053,
        -0.52357,
        -0.47425,
        -0.39017,
        0.00000,
    ],
    5000: [
        0.00000,
        0.42447,
        0.43329,
        0.43648,
        0.42951,
        0.35368,
        0.28066,
        0.27280,
        0.00945,
        -0.30018,
        -0.36214,
        -0.41442,
        -0.52876,
        -0.55408,
        -0.55069,
        -0.49774,
        0.00000,
    ],
    7500: [
        0.00000,
        0.43979,
        0.44030,
        0.43564,
        0.41824,
        0.35060,
        0.28117,
        0.27348,
        0.00824,
        -0.30448,
        -0.36213,
        -0.41050,
        -0.48590,
        -0.52347,
        -0.55216,
        -0.53858,
        0.00000,
    ],
    10000: [
        0.00000,
        0.43983,
        0.43733,
        0.43124,
        0.41487,
        0.35070,
        0.28003,
        0.27224,
        0.00831,
        -0.30719,
        -0.36737,
        -0.41496,
        -0.45863,
        -0.49099,
        -0.52987,
        -0.54302,
        0.00000,
    ],
}


def evaluate(config: ml_collections.ConfigDict, workdir: str, Re: int):
    Re = 5000
    # Load dataset
    u_ref, v_ref, p_ref, p_x_ref, p_y_ref, x_star, y_star, nu = get_dataset(Re)

    # Initialize model
    model = models.NavierStokes2D(config, p_x_ref, p_y_ref, x_star, y_star, nu)

    # Restore checkpoint
    path = os.path.join(".", "ckpt", config.wandb.name, "Re{}".format(Re))
    model.state = restore_checkpoint(model.state, path)
    params = model.state.params

    # Predict
    u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(
        params, x_star, y_star
    )
    v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(
        params, x_star, y_star
    )

    u_l2_error = jnp.sqrt(jnp.mean((u_ref - u_pred) ** 2)) / jnp.sqrt(
        jnp.mean(u_ref**2)
    )
    v_l2_error = jnp.sqrt(jnp.mean((v_ref - v_pred) ** 2)) / jnp.sqrt(
        jnp.mean(v_ref**2)
    )
    print("u_l2_error: {:.4e}".format(u_l2_error))
    print("v_l2_error: {:.4e}".format(v_l2_error))

    # Plot
    n_x, n_y = u_ref.shape
    XX, YY = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Save path
    save_path = os.path.join(workdir, "figures", config.wandb.name, "Re{}/".format(Re))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    fig1 = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted u(x, y)")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    fig1.savefig(save_path + "ldc_u" + ".pdf", bbox_inches="tight", dpi=300)

    # Plot the results
    fig2 = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, v_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, v_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted v(x, y)")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(v_ref - v_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()
    fig2.savefig(save_path + "ldc_v" + ".pdf", bbox_inches="tight", dpi=300)

    fig3 = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_Ghia, u_Ghia[Re], "o", label="Ghia")
    plt.plot(y_star, u_ref[n_x // 2, :], label="Reference")
    plt.plot(y_star, u_pred[n_x // 2, :], label="Predicted")
    plt.xlabel("y")
    plt.ylabel("u(0.5, y)")
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(x_Ghia, v_Ghia[Re], "o", label="Ghia")
    plt.plot(x_star, v_ref[:, n_y // 2], label="Reference")
    plt.plot(y_star, v_pred[:, n_y // 2], label="Predicted")
    plt.xlabel("x")
    plt.ylabel("v(x, 0.5)")
    plt.legend()
    plt.tight_layout()
    fig3.savefig(save_path + "ldc_Ghia" + ".pdf", bbox_inches="tight", dpi=300)
