import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('../outputs/fit_output_nphot1e6.dat')

fig = plt.figure(dpi=180)
ax = np.ndarray.flatten(fig.subplots(2, 1, sharex=True))

ax[0].scatter(a[:, 0], a[:, 1], color='r', s=3, label='mu', alpha=0.5)
ax[0].set_ylabel('mu')
ax[1].scatter(a[:, 0], a[:, 2], color='b', s=3, label='sigma', alpha=0.5)
ax[1].set_ylabel('sigma')


plt.xlabel('tau')
plt.suptitle('Log Normal Parameters v. Optical Depth (n_phot=1e6)')
plt.savefig('../outputs/tau_output_nphot1e6.pdf')
