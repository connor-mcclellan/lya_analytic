import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('../outputs/exit_photons_tau10.dat', skiprows=1)
data[:, 0] = np.round(data[:, 0], 5)


fig, ax = plt.subplots(1, 1, dpi=180)
n, bins, patches = ax.hist(data[:, 7], bins=100, color='k', histtype='step', density=True)
ax.set_xlabel(r'$\nu$ (Hz)')
ax.set_ylabel(r'$n$ (Normalized)')
ax.set_title(r'Ly$\alpha$ spectrum, n=1e6 photons, $\tau=10$')
plt.tight_layout()
plt.savefig('../plots/spec_tau10.pdf')


