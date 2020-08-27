import numpy as np
from solutions.util import j0
import h5py
from tqdm import tqdm
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from mpio import rsigmat_parallel, save_queue_rsigmat
import pdb

def evaluate_J_H(inputname, r, sigma, t, outputname, axis=1, mp=True):

    # Parallel processing setup
    cores = 16
    pool = Pool(processes=cores)

    # Load in some input variable arrays
    Jnsigmaomega = h5py.File(inputname, 'r')
    full_sigma = Jnsigmaomega['sigma'][:]
    omega = Jnsigmaomega['omega'][:]
    n = Jnsigmaomega['n'][:]
    d_omega = np.diff(omega)[0]
    R = r[-1]
    Jnsigmaomega.close()

    # Prepare output variable arrays
    aux_variables = [r, sigma, t]
    prim_variable = aux_variables.pop(axis)
    names = ['r', 'sigma', 't']
    name = names.pop(axis)

    # Create output file --- data will be filled in during loops
    output = h5py.File(outputname, 'w')
    output.create_dataset(names[0], data=aux_variables[0])
    output.create_dataset(names[1], data=aux_variables[1])
    output.create_dataset(name, data=prim_variable)
    output.close()

    if mp:
        for i in range(len(aux_variables[0])):
            for j in range(len(aux_variables[1])):
                result = pool.apply_async(rsigmat_parallel, args=(inputname, i, j, len(aux_variables[1]), prim_variable, n, R, omega, d_omega, r, sigma, full_sigma, t, names, name, axis, outputname), callback=save_queue_rsigmat)    
        pool.close()
        pool.join()
    else:
        output = h5py.File(outputname, 'a')
        pb = tqdm(total=len(t)*len(r)*len(n)*len(omega)*len(sigma))
        for l in range(len(n)):
            kappa_n = n[l] * np.pi / R
            for m in range(len(omega)):

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
                            pb.update()
                        iterator = iters_ord.pop(axis)
                        Jsetname = 'J_{}{}_{}{}'.format(names[0], iters_ord[0], names[1], iters_ord[1])
                        Hsetname = 'H_{}{}_{}{}'.format(names[0], iters_ord[0], names[1], iters_ord[1])
                        try:
                            output.create_dataset(Jsetname, data=J)
                            output.create_dataset(Hsetname, data=H)
                        except:
                            output[Jsetname][:] += J
                            output[Hsetname][:] += H
        pb.close()


if __name__ == "__main__":
    r = [1e11, ]
    t = [10., ]
    sigma_eval = np.linspace(-1e8, 1e8, 64)
    inputname = '/LyraShared/bcm2vn/outputs/lya_analytic/n16_sigma100000_omega128.hdf5'
    outputname = '/LyraShared/bcm2vn/outputs/lya_analytic/r{}_sigma{}_t{}.hdf5'.format(len(r), len(sigma_eval), len(t))
    evaluate_J_H(inputname, r, sigma_eval, t, outputname, axis=1, mp=False)
