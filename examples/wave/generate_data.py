"""Generate reference solution for the 1D wave equation.

Solves:  u_tt = c^2 * u_xx  on [0, 1] x [0, 1]
BCs:     u(0, t) = u(1, t) = 0  (fixed ends)
ICs:     u(x, 0) = sin(pi * x),  u_t(x, 0) = 0
Exact:   u(x, t) = sin(pi * x) * cos(pi * c * t)
"""

import numpy as np
import scipy.io as sio
import os

c = 1.0  # wave speed

nx = 256
nt = 201

x_star = np.linspace(0, 1, nx)
t_star = np.linspace(0, 1, nt)

TT, XX = np.meshgrid(t_star, x_star, indexing="ij")
usol = np.sin(np.pi * XX) * np.cos(np.pi * c * TT)

save_path = os.path.join(os.path.dirname(__file__), "data", "wave.mat")
sio.savemat(save_path, {"usol": usol, "t": t_star.reshape(-1, 1), "x": x_star.reshape(-1, 1)})
print(f"Saved reference solution to {save_path}")
print(f"  usol shape: {usol.shape}")
print(f"  t shape: {t_star.shape}")
print(f"  x shape: {x_star.shape}")
