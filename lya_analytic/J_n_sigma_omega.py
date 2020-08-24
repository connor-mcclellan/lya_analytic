import numpy as np
from solutions.util import Params, Line
from solutions.boundaryvalue import BoundaryValue
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from mpio import process
import h5py

# Constants
c = 29979245800.0

# Parallel processing setup
pool = Pool(processes=8)

# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e6)

# Diffusion time
tdiff = p.R / c * (p.a * p.tau0)**(1./3) # Diffusion time
dt = 0.1*tdiff

# Number of omega points in gridh
N_omegas = 2
N_ns = 4

# Create grids
omega_grid, d_omega = np.linspace(0, 2*np.pi/dt, N_omegas, retstep=True)
n_grid = np.arange(1, N_ns+1, 1)
sigma_grid = p.sigma_grid

# Create output hdf5 file
fname = './outputs/n{}_sigma{}_omega{}.hdf5'.format(N_ns, len(sigma_grid), N_omegas)

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


