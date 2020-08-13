import numpy as np
from solutions.util import j0

# Load in data from J_n_sigma_omega.py output
n_grid = np.load("./outputs/n_grid.npy")
sigma_grid = np.load("./outputs/sigma_grid.npy")
omega_grid = np.load("./outputs/omega_grid.npy")
J = np.load("./outputs/J.npy")

# Grids
r_grid = np.linspace(0, 9, 10)
t_grid = np.linspace(0, 1e4, int(1e6))

# Other parameters
d_omega = np.diff(omega_grid)[0]
R = r_grid[-1]

# Sum & discretized integral
for i in range(len(t_grid)):

    for j in range(len(r_grid)):
        n_sum = 0.

        for k in range(len(n_grid)):
            n = n_grid[k]
            omega_integral = 0.

            for l in range(len(omega_grid):
                omega = omega_grid[l]
                kappa_n = n * np.pi / R
                J_n_sigma_omega = J[k, :, l, 0] + 1j*J[k, :, l, 1]

                # Eq 34
                omega_integral += d_omega / (2.*np.pi) * J_n_sigma_omega * j0(kappa_n / r) * np.exp(-1j*omega*t)
            n_sum += omega_integral
