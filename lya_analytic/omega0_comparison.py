from solutions.util import Params, Line, j0
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pdb

'''
Compares 16 fourier coefficients against the analytic solution for omega=0
'''


# Eq 30
#def J(r, sigma, R, delta, L, k, sigma_source):
#    return np.sqrt(6)/32./np.pi**2.*L/R/r/delta*(np.sin(np.pi*r/R)/(np.cosh(np.pi*delta*np.abs(sigma-sigma_source)/k/R) - np.cos(np.pi*r/R)))

def J(n, sigma, energy, delta, R, kappa_n, k):
    return np.sqrt(6.)/16./np.pi*n*energy/delta/R**2.*np.exp(-kappa_n*delta*np.abs(sigma)/k)


# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e4)

fig = plt.figure()
colors = pl.cm.jet(np.linspace(0, 1, 8))
for n in range(1, 9):
#    inputfile = h5py.File('./outputs/Jnso/n8_sigma10000_omega128_rk.hdf5', 'r')
    inputfile = h5py.File('./outputs/Jnso/n8_sigma10000_omega128_rk_lowtol.hdf5', 'r')
    Jdump = inputfile['J_omega0_n{}'.format(n-1)][:]
    inputfile.close()

    print(n)

    kappa_n = n*np.pi / p.R
    J_analytic = J(n, p.sigma_grid, p.energy, p.delta, p.R, kappa_n, p.k)
    J_numerical = Jdump

    # Scatterplot
    plt.plot(p.sigma_grid, np.abs(J_numerical.real - J_analytic.real)/J_analytic.real, '.', alpha=1, color=colors[n-1], ms=2)

    # lines
#    plt.plot(p.sigma_grid, J_numerical.real, '-', alpha=0.5, color=colors[n-1])
#    plt.plot(p.sigma_grid, J_analytic.real, 'k--', alpha=0.25, label='Analytic')

sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=8))
cbar = fig.colorbar(sm)
cbar.ax.set_ylabel('n', rotation=90)

plt.title('Fractional Error in Numerical Solve for $\omega=0$ (rk)')
plt.xlabel('sigma')
plt.ylabel('J')
plt.yscale('log')
plt.legend()
plt.show()


#plt.plot(np.linspace(-4e3, 4e3, int(1e4)), J(1, np.linspace(-4e3, 4e3, int(1e4)), p.energy, p.delta, p.R, kappa_n, p.k))
#plt.show()
    
    
