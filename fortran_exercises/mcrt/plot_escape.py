import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np
import scipy
from scipy.stats import lognorm
from prob_ct_tau import prob_ct_tau


tau = 10.


data = np.loadtxt('exit_photons.dat', skiprows=1)
data[:, 0] = np.round(data[:, 0], 5)
data[:, 6] = data[:, 6]/(1./tau * (tau + 2./3.)**2. * 3./np.pi**2.)

fig, ax = plt.subplots(1, 1, dpi=180)
n, bins, patches = ax.hist(data[:, 6], bins=50, color='k', histtype='step', density=True)
bincenters = 0.5*(bins[1:]+bins[:-1])
ax.set_xlabel('Distance')
ax.set_ylabel('n (normalized)')
ax.set_yscale('log')
ax.set_title(r'Total Distance Traveled, $\tau = {}$, $n = 10^5$'.format(int(tau)))

#bincenters = bincenters[n >= 3]
#n = n[n >= 3]

#coef, res, _, _, _ = np.polyfit(bincenters, np.log(n), 1, full=True)
#poly1d = np.poly1d(coef)
#ax.plot(bincenters, np.exp(poly1d(bincenters)), '--r', label='Exponential')
#print(coef, res)

prob = np.zeros(np.shape(bincenters))
for i in range(len(bincenters)):
    prob[i] = prob_ct_tau(bincenters[i], tau)

ax.plot(bincenters, prob, 'b--', label='Series Solution')

#shape, loc, scale = lognorm.fit(data[:, 6], loc=1)
#pdf = lognorm.pdf(bincenters, shape, loc, scale)
#ax.plot(bincenters, pdf, 'r--', label='Log Normal')

plt.legend()
plt.savefig('./escape/fit_tau{}.pdf'.format(int(tau)))
#plt.show()
