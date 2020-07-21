import numpy as np
import matplotlib.pyplot as plt
from solutions import fits
from solutions.prob_ct_tau import prob_ct_tau
import matplotlib
from astropy.utils.console import ProgressBar
matplotlib.rc('text', usetex=True)
import pdb

array = np.load('./outputs/1m_bin_time.npy')
tc = array[0]
ydata = array[1]
theory = array[2]
tau0=1e7

'''
t0_fit = []
print('Refining adaptive series solution...')
pb = ProgressBar(len(tc))
for i in range(len(tc)):
    t0_bounds = (6., 13.)
    stepsize = 0.5
    while stepsize > 1e-4:
        t0_array = np.arange(min(t0_bounds), max(t0_bounds), stepsize)
        prob_array = np.zeros(len(t0_array))
        for j in range(len(t0_array)):
            prob_array[j] = prob_ct_tau(tc[i]/t0_array[j], tau0)/t0_array[j]
        closest_index = np.argmin(np.abs(2.3*prob_array*tc[i] - ydata[i]))
        try:
            t0_bounds = (t0_array[closest_index-1], t0_array[closest_index+1])
        except:
            t0_bounds = (t0_array[closest_index-1], t0_array[closest_index])
        stepsize = stepsize*0.1
    t0_fit.append(t0_array[closest_index])
    pb.update()


prob = np.zeros(len(tc))
for i in range(len(tc)):
    prob[i] = prob_ct_tau(tc[i]/t0_fit[i], tau0)/t0_fit[i]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
ax1.plot(tc, 2.3*prob*tc, '--', label='Adaptive Series Solution', alpha=0.25)
ax1.plot(tc, ydata, 'k.', ms=2, label='Data')
ax1.set_ylim((1e-4, 1e1))
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$2.3tP(t)$')
ax1.set_title(r'$\tau_0={}, n={}$'.format(tau0, 1e6))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend()

# Plot and fit adaptive solution t0s
ax2.plot(tc, t0_fit, 'k.', ms=2, label='$t_0$')

# Polynomial fit
z = np.polyfit(tc, t0_fit, 5)
p = np.poly1d(z)
#ax2.plot(tc, p(tc), '--', label='Polyfit', alpha=0.5)

ax2.set_xscale('log')
ax2.set_xlabel('t')
ax2.set_ylabel(r'$t_0$')
ax2.legend()
plt.show()
'''


#plt.figure()
#plt.plot(tc, ydata, 'k.', ms=2, label='Data')
#prob = np.zeros(len(tc))
#t0 = np.median(p(tc))
#for i in range(len(tc)):
##    prob[i] = prob_ct_tau(tc[i]/p(tc[i]), tau0)/p(tc[i])
#    prob[i] = prob_ct_tau(tc[i]/t0, tau0)/t0
#plt.plot(tc, 2.3*prob*tc, '--', label='$t_0={:.3f}$', alpha=0.25)
#plt.ylim((1e-4, 1e1))
#plt.xlabel(r'$t$')
#plt.ylabel(r'$2.3tP(t)$')
#plt.title(r'$\tau_0={}, n={}$'.format(tau0, 1e6))
#plt.yscale('log')
#plt.xscale('log')
#plt.legend()
#plt.show()

t0 = np.linspace(8., 11., 3)
tau0 = 1e7#np.linspace(20, 100, 5)
prob = np.zeros((len(t0), len(tc)))
for j in range(len(t0)):
    for i in range(len(tc)):
        prob[j, i] = prob_ct_tau(tc[i]/t0[j], tau0)/t0[j]
    plt.plot(tc, 2.3*prob[j]*tc, '--', label=r'$t_0={:.2f}$'.format(t0[j]), alpha=0.25)
plt.plot(tc, ydata, 'k.', ms=2, label='Data')
plt.plot(tc, 15.*np.exp(-tc/10.75), '--', label=r'$Ce^{-t/t_{\mathrm{diff}}}$', alpha=0.5)
plt.ylim((1e-4, 1e1))
plt.xlabel(r'$t$')
plt.ylabel(r'$2.3tP(t)$')
plt.title(r'$\tau_0={}, n={}$'.format(tau0, 1e6))
plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.show()

