import numpy as np
from solutions.util import Params, Line, j0
from solutions.boundaryvalue import BoundaryValue
import time
import pdb

lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1e6, energy=1., 
           sigma_source=0., n_points=1e2)

# Internals
omega_grid, d_omega = np.linspace(0, 100, 100, retstep=True)
n_grid = np.arange(1, 16, 1)
sigma_grid = p.sigma_grid

J = np.zeros((len(n_grid), len(sigma_grid), len(omega_grid), 2))

# Calculate J_n_sigma_omega for all internal variables
for i in range(len(omega_grid)):
    start = time.time()
    for j in range(len(n_grid)):

        # Get solution for this fourier coefficient
        bv = BoundaryValue(n_grid[j], omega_grid[i], p)
        _, J_real, J_imag, _, _ = bv.solve()
        pdb.set_trace()
        # Assign coefficient's value in 4D array
        J[j, :, i, 0] = J_real
        J[j, :, i, 1] = J_imag
        
    end = time.time()
    print('Estimated Completion: {} s'.format((end-start)*len(omega_grid)))

np.save("./outputs/J", J)

'''
# Externals
r_grid = np.linspace(0, 9, 10)
t_grid = np.linspace(0, 1e4, int(1e6))


for t in t_grid:
    for r in r_grid:
        # Begin internals
        n_sum = 0.
        for n in n_grid:
            omega_integral = 0.
            for omega in omega_grid:
                J_n_sigma_omega = # ...
                kappa_n = # ...

                # Eq 34
                omega_integral += d_omega / (2.*np.pi) * J_n_sigma_omega * j0(kappa_n / r) * np.exp(-1j*omega*t)
            n_sum += omega_integral
'''
