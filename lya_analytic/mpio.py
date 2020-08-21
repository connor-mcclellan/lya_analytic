from solutions.boundaryvalue import BoundaryValue
import h5py
import numpy as np

def process(n_grid, omega_grid, sigma_grid, i, j, p, fname):
    bv = BoundaryValue(n_grid[j], omega_grid[i], p)
    _, J_real, J_imag, _, _ = bv.solve()
    J = J_real + 1j*J_imag
    return (i, j, J, fname)

def save_queue(result):
    i, j, J, fname = result
    output = h5py.File(fname, 'a')
    output.create_dataset("J_omega{}_n{}".format(i, j), data=J, dtype=np.complex)
    output.close()
