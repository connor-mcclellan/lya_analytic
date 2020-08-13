from solutions.boundaryvalue import BoundaryValue
import numpy as np

def process(n_grid, omega_grid, i, j, p):
    bv = BoundaryValue(n_grid[j], omega_grid[i], p)
    _, J_real, J_imag, _, _ = bv.solve()
    J = np.zeros((len(sigma_grid), 2))
    J[:, 0] = J_real
    J[:, 1] = J_imag
    np.save("./outputs/n8_sigma1e6_omega64/J_omega{}_n{}".format(i, j), J)
