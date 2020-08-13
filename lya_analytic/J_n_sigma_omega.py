import numpy as np
from solutions.util import Params, Line
from solutions.boundaryvalue import BoundaryValue
import time
import pdb
from multiprocessing import Pool, cpu_count
from mpio import process

# Constants
c = 29979245800.0

# Parallel processing setup
pool = Pool(processes=(cpu_count() - 1))

# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e6)

# Diffusion time
tdiff = p.R / c * (p.a * p.tau0)**(1./3) # Diffusion time
dt = 0.1*tdiff

# Number of omega points in grid
N = 64

# Create grids
omega_grid, d_omega = np.linspace(0, 2*np.pi/dt, N, retstep=True)
n_grid = np.arange(1, 9, 1)
sigma_grid = p.sigma_grid

## Set up empty 4D array for data 
#J = np.zeros((len(n_grid), len(sigma_grid), len(omega_grid), 2))

# Calculate J_n_sigma_omega for all grid points
#computation_times = []
for i in range(len(omega_grid)):
    start = time.time()
    for j in range(len(n_grid)):
        result = pool.apply_async(process, args=(n_grid, omega_grid, i, j, p))
#    end = time.time()
#    computation_times.append(end-start)

    # Estimate time until finished
#    if len(computation_times) >= 2:
#        z = np.polyfit(omega_grid[:len(computation_times)], computation_times, 2)
#        p = np.poly1d(z)
#        projected = np.sum(p(omega_grid))
#        remaining = projected - np.sum(computation_times)
#        projectedstr = time.strftime('%H:%M:%S', time.gmtime(projected))
#        remainingstr = time.strftime('%H:%M:%S', time.gmtime(remaining))
#        print("\nProjected duration: {}".format(projectedstr))
#        print("Projected time remaining: {} \n".format(remainingstr))

np.save("./outputs/n8_sigma1e6_omega64/n_grid", n_grid)
np.save("./outputs/n8_sigma1e6_omega64/sigma_grid", sigma_grid)
np.save("./outputs/n8_sigma1e6_omega64/omega_grid", omega_grid)

pool.close()
pool.join()
