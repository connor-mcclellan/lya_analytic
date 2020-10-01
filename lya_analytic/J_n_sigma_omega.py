import numpy as np
from solutions.util import Params, Line
from solutions.boundaryvalue import BoundaryValue
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from mpio import process
import h5py
import math as m


# Constants
c = 29979245800.0

# Parallel processing setup
pool = Pool(processes=8)

# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e5)

# Diffusion time
tdiff = p.R / c * (p.a * p.tau0)**(1./3) # Diffusion time
dt = 0.1*tdiff/4.#TODO: Do a run with dt = 0.1*tdiff/4

# Number of omega points in grid
N_omegas = 512 # TODO: Do a run with 512 omegas, with dt = 0.1*tdiff/4. Reduce sigma max if needed
N_ns = 8

# Create grids
omega_grid = np.linspace(0, 2*np.pi/dt, N_omegas)
#omega_grid = np.logspace(np.log(1e-3), np.log10(2*np.pi/dt), N_omegas-1)
#omega_grid = np.insert(omega_grid, 0, 0.)
n_grid = np.arange(1, N_ns+1, 1)
sigma_grid = p.sigma_grid

# Create output hdf5 file
fname = '/LyraShared/bcm2vn/outputs/lya_analytic/n{}_sigma{}_omega{}_highfreq_lorprofile.hdf5'.format(N_ns, len(sigma_grid), N_omegas)
#fname = './outputs/n{}_widesigma{}_logomega{}.hdf5'.format(N_ns, len(sigma_grid), N_omegas)

pb = tqdm(total=len(omega_grid)*len(n_grid))
def save_queue(result):
    pb.update()
    i, j, J, fname = result
    output = h5py.File(fname, 'a')
    output.create_dataset("J_omega{}_n{}".format(i, j), data=J, dtype=np.complex)
    output.close()

# Calculate J_n_sigma_omega for all grid points
for i in range(len(omega_grid)):
    for j in range(len(n_grid)):
#        process(n_grid, omega_grid, sigma_grid, i, j, p, fname)
        result = pool.apply_async(process, args=(n_grid, omega_grid, sigma_grid, i, j, p, fname), callback=save_queue)
pool.close()
pool.join()
pb.close()

output = h5py.File(fname, 'a')
output.create_dataset("n", data=n_grid)
output.create_dataset("sigma", data=sigma_grid)
output.create_dataset("omega", data=omega_grid)

# Set attributes 
for key, value in p.__dict__.items():
    try:
#        print(key, value)
        output.attrs[key] = value
    except:
#        print('attribute failed')
        pass
output.attrs['tdiff'] = tdiff
output.attrs['dt'] = dt
output.close()


