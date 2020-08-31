from solutions.boundaryvalue import BoundaryValue
from solutions.util import j0
from scipy.interpolate import interp1d
import h5py
import numpy as np
from tqdm import tqdm

def process(n_grid, omega_grid, sigma_grid, i, j, p, fname):
    try:
        bv = BoundaryValue(n_grid[j], omega_grid[i], p)
        _, J_real, J_imag, _, _ = bv.solve()
        J = J_real + 1j*J_imag
    except Exception as e:
        print(e)
    return (i, j, J, fname)

def rsigmat_parallel(inputname, l, m, n, aux_variables, prim_variable, R, omega, d_omega, r, sigma, full_sigma, t, names, name, axis, fname):
    try:
        kappa_n = n[l] * np.pi / R

        # Load fourier coefficients for this n and omega
        Jnsigmaomega = h5py.File(inputname, 'r')
        Jdump = Jnsigmaomega['J_omega{}_n{}'.format(m, l)][:]
        J_interp = interp1d(full_sigma, Jdump)
        Jnsigmaomega.close()

        for i in range(len(aux_variables[0])):
            for j in range(len(aux_variables[1])):
                J = np.zeros(len(prim_variable), dtype=np.complex)
                H = np.zeros(len(prim_variable), dtype=np.complex)
                for k in range(len(prim_variable)):

                    # Figure out which index goes with which variable
                    iters = [i, j, k]
                    order = np.argsort(np.array(names+[name]))
                    iters_ord = [iters[s] for s in order]
                    r_index, sigma_index, t_index = iters_ord

                    # Eq 34
                    J[k] += d_omega / (2.*np.pi) * J_interp(sigma[sigma_index]) * j0(kappa_n, r[r_index]) * np.exp(-1j*omega[m]*t[t_index])
                    j0_prime = np.cos(kappa_n*r[r_index])/r[r_index] - np.sin(kappa_n*r[r_index])/kappa_n/r[r_index]**2.
                    H[k] += d_omega / (2.*np.pi) * J_interp(sigma[sigma_index]) * j0_prime * np.exp(-1j*omega[m]*t[t_index])
                iterator = iters_ord.pop(axis)
                Jsetname = 'J_{}{}_{}{}'.format(names[0], iters_ord[0], names[1], iters_ord[1])
                Hsetname = 'H_{}{}_{}{}'.format(names[0], iters_ord[0], names[1], iters_ord[1])
    except Exception as e:
       print(e)
    return (Jsetname, Hsetname, J, H, fname)
