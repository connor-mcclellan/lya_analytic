import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('exit_photons.dat', skiprows=1)
data[:, 0] = np.round(data[:, 0], 5)


#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(polar=True))

fig = plt.figure(figsize=(10, 6), dpi=180)

xlabels = [r'$\theta$ ($\pi$ radians)', r'$\phi$ ($\pi$ radians)']
titles = [r'$\theta$ Position', r'$\phi$ Position']
scale = [1./np.pi, 1./np.pi]
############
axs = fig.subplots(2, 3)
axs = np.ndarray.flatten(axs)

for i in range(1, 3):
  ax = axs[i-1]
  ax.hist(scale[i-1]*data[:, i], bins=50, color='k', histtype='step', density=False)
  print(i, 1./np.sqrt(len(data[:, i])/50))
  ax.set_xlabel(xlabels[i-1])
  ax.set_ylabel(r'n')
  ax.set_title(titles[i-1])

axs[2].hist(data[:, 6], bins=20, color='k', histtype='step', density=False)
axs[2].set_xlabel('Distance')
axs[2].set_ylabel('n')
axs[2].set_yscale('log')
axs[2].set_title('Total Distance Traveled')

r = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]

dat = [data[:, 3], data[:, 4]]
xlabels = [r'$\theta$ ($\pi$ radians)', r'$\phi$ ($\pi$ radians)']
titles = [r'Exit $\theta$', r'Exit $\phi$']
scale = [1./np.pi, 1./np.pi]

for i in range(2):
  ax = axs[i+3]
  ax.hist(scale[i]*dat[i], bins=50, color='k', histtype='step', density=False)
  ax.set_xlabel(xlabels[i])
  ax.set_ylabel(r'n')
  ax.set_title(titles[i])

axs[5].hist(data[:, 5], bins=20, color='k', histtype='step', density=False)
plt.xlabel('n steps')
plt.ylabel('n')

plt.suptitle(r'Escape from Sphere, $\tau=100$')
plt.tight_layout()
plt.subplots_adjust(top=0.878)
plt.savefig('escape_tau100.pdf')
