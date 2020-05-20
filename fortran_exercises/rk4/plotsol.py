import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('sol.dat')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(data[:,0]/(2*np.pi), data[:,1], 'c-', label='Numerical y', alpha=0.5)
ax1.plot(data[:,0]/(2*np.pi), data[:,3], 'm-', label='Analytic y', alpha=0.5)
ax2.plot(data[:,0]/(2*np.pi), data[:,2], 'c--', label='Numerical dy/dx', alpha=0.5)
ax2.plot(data[:,0]/(2*np.pi), data[:,4], 'm--', label='Analytic dy/dx', alpha=0.5)
plt.xlabel(r't/2$\pi$')
ax1.set_ylabel('y')
ax2.set_ylabel('y')
ax1.legend()
ax2.legend()
ax1.set_ylim((-2, 2))
ax2.set_ylim((-2, 2))
ax1.set_yticks([-1, 0, 1])
ax2.set_yticks([-1, 0, 1])
fig.subplots_adjust(hspace=0, wspace=0)
plt.suptitle('Step size = 1e-4')
plt.show()

