from solutions.util import Params, Line, j0
import numpy as np
import h5py
import matplotlib.pyplot as plt

def J(r, sigma, R, delta, L, k, sigma_source):
    return np.sqrt(6)/32./np.pi**2.*L/R/r/delta*(np.sin(np.pi*r/R)/(np.cosh(np.pi*delta*np.abs(sigma-sigma_source)/k/R) - np.cos(np.pi*r/R)))

# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e5)

r = 1e6

J_analytic = J(r, p.sigma_grid, p.R, p.delta, p.energy, p.k, p.sigma_source)
J_numerical = np.zeros(len(p.sigma_grid), dtype=np.complex)
for n in range(1, 17):
    inputfile = h5py.File('./outputs/n16_sigma100000_omega128.hdf5', 'r')
    Jdump = inputfile['J_omega0_n{}'.format(n-1)][:]
    inputfile.close()

    kappa_n = n*np.pi / p.R
    J_numerical += Jdump * j0(kappa_n, r)

plt.figure()
plt.plot(p.sigma_grid, J_analytic.real, label='J analytic')
plt.plot(p.sigma_grid, J_numerical.real, '--', label='J numerical')
plt.title('r = {}'.format(r))
plt.xlabel('sigma')
plt.ylabel('J')
plt.yscale('log')
plt.legend()
plt.show()
    
    
