import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('exit_photons.dat', skiprows=1)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(polar=True))

fig = plt.figure(figsize=(10, 4), dpi=80)

xlabels = [r'$r$', r'$\theta$', r'$\phi$']
titles = [r'$r$ Position', r'$\theta$ Position', r'$\phi$ Position']
scale = [1., 180./np.pi, 180./np.pi]
############
axs = fig.subplots(2, 3)
axs = np.ndarray.flatten(axs)

for i in range(3):
  ax = axs[i]
  ax.hist(scale[i]*data[:, i], bins=50, color='k', histtype='step', density=True)
  ax.set_xlabel(xlabels[i])
  ax.set_ylabel(r'n')
  ax.set_title(titles[i])

r = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]

xhat = data[:, 3]
yhat = data[:, 4]
zhat = data[:, 5]

rhat = np.sin(theta)*np.cos(phi)*xhat   \
       + np.sin(theta)*np.sin(phi)*yhat \
       + np.cos(theta)*zhat

exit_theta = np.arccos(zhat)
exit_phi = np.arctan2(yhat, xhat)
exit_phi[exit_phi < 0.0] += 2.0*np.pi

dat = [rhat, exit_theta, exit_phi]
xlabels = [r'$r$', r'$\theta$', r'$\phi$']
titles = [r'$\hat{r}$', r'Exit $\theta$', r'Exit $\phi$']
scale = [1., 180./np.pi, 180./np.pi]

for i in range(3):
  ax = axs[i+3]
  ax.hist(scale[i]*dat[i], bins=50, color='k', histtype='step', density=True)
  ax.set_xlabel(xlabels[i])
  ax.set_ylabel(r'n')
  ax.set_title(titles[i])

plt.suptitle(r'$\chi=10^{-2}$, $R=20$, $b_{max}=20$')
plt.tight_layout()
plt.subplots_adjust(top=0.878)
plt.show()

plt.figure()
plt.hist(data[:, 6], bins=50, color='k', histtype='step', density=True)
plt.xlabel('n steps')
plt.ylabel('n')
plt.tight_layout()
plt.show()
