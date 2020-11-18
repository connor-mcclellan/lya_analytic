import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('coords.dat', skiprows=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.hist(data[:, 0], bins=50, density=True, histtype='step')
ax2.hist(data[:, 1], bins=50, density=True, histtype='step')

ax1.plot(np.linspace(0, 5, 50), 2.*np.linspace(0, 5, 50)/(5.0**2.), 'r--', label=r'$2r/R^2$')
ax2.plot(np.linspace(0, 2.0*np.pi, 50), np.ones(50)/2.0/np.pi, 'r--', label=r'$1/2\pi$')

ax1.set_xlabel(r'$r$')
ax1.set_ylabel('n (normalized)')
ax1.legend()

ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel('n (normalized)')
ax2.legend()

plt.suptitle(r'Probability Distribution in $r$ and $\theta$ (N=1e4)')
plt.subplots_adjust(top=0.935,
bottom=0.074,
left=0.046,
right=0.99,
hspace=0.2,
wspace=0.114)
plt.show()
