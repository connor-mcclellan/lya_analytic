from solutions.util import Params, Line, j0
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pdb

# Eq 30
#def J(r, sigma, R, delta, L, k, sigma_source):
#    return np.sqrt(6)/32./np.pi**2.*L/R/r/delta*(np.sin(np.pi*r/R)/(np.cosh(np.pi*delta*np.abs(sigma-sigma_source)/k/R) - np.cos(np.pi*r/R)))

def J(n, sigma, energy, delta, R, kappa_n, k):
    return np.sqrt(6.)/16./np.pi*n*energy/delta/R**2.*np.exp(-kappa_n*delta*np.abs(sigma)/k)


# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e5)

fig = plt.figure()
colors = pl.cm.jet(np.linspace(0, 1, 17))
for n in range(1, 17):
    inputfile = h5py.File('./outputs/n16_sigma100000_omega128.hdf5', 'r')
    Jdump = inputfile['J_omega0_n{}'.format(n-1)][:]
    inputfile.close()

    kappa_n = n*np.pi / p.R
    J_analytic = J(n, p.sigma_grid, p.energy, p.delta, p.R, kappa_n, p.k)
    J_numerical = Jdump

    plt.plot(p.sigma_grid, J_numerical.real, '-', alpha=0.25, color=colors[n])
    plt.plot(p.sigma_grid, J_analytic.real, 'k--', alpha=0.25)

sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=17))
cbar = fig.colorbar(sm)
cbar.ax.set_ylabel('n', rotation=90)

plt.xlabel('sigma')
plt.ylabel('J')
plt.yscale('log')
plt.legend()
plt.show()


plt.plot(np.linspace(-4e3, 4e3, int(1e4)), J(1, np.linspace(-4e3, 4e3, int(1e4)), p.energy, p.delta, p.R, kappa_n))
plt.show()
    
    
