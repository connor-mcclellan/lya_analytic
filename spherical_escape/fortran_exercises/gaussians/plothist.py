import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('gauss.dat', skiprows=1)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.hist(data[:], bins=50, density=True, histtype='step')

def gauss(mu, sigma, x):
   return 1.0 / sigma / np.sqrt(2.0*np.pi) * np.exp(-0.5 * ((x-mu)/sigma)**2.0)
    
mu, sigma, offset = 0.0, 1.0, 2.0
x = np.linspace(-4., 4., 50)
ax1.plot(x, 0.5*gauss(mu+offset, sigma, x), 'r--')
ax1.plot(x, 0.5*gauss(mu-offset, sigma, x), 'r--')

ax1.set_xlabel(r'$x$')
ax1.set_ylabel('n (normalized)')


plt.suptitle(r'Double Gaussian Probability Distribution (N=1e6, $\mu$=0, $\sigma$=1, Offset=2.0)')
plt.subplots_adjust()
plt.show()
