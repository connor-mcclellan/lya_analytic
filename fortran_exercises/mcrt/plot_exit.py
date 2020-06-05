import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('exit_photons.dat', skiprows=1)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(polar=True))

fig = plt.figure(figsize=(10, 4), dpi=80)

xlabels = [r'$r$', r'$\theta$', r'$\phi$', r'$\hat{r}$', r'$\hat{\theta}$', r'$\hat{\phi}$']
titles = [r'$r$ Position', r'$\theta$ Position', r'$\phi$ Position', r'$r$ Direction', r'$\theta$ Direction', r'$\phi$ Direction']
scale = [1., 180./np.pi, 180./np.pi, 1., 180./np.pi, 180./np.pi]
############
axs = fig.subplots(2, 3)
axs = np.ndarray.flatten(axs)

for i in range(6):
  ax = axs[i]
  ax.hist(scale[i]*data[:, i], bins=50, color='k', histtype='step', density=True)
  ax.set_xlabel(xlabels[i])
  ax.set_ylabel(r'n')
  ax.set_title(titles[i])

plt.tight_layout()
plt.show()

plt.figure()
plt.hist(data[:, 6], bins=50, color='k', histtype='step', density=True)
plt.xlabel('n steps')
plt.ylabel('n')
plt.tight_layout()
plt.show()
