import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('vecs.dat')

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(polar=True))

fig = plt.figure(figsize=(10, 4), dpi=80)


############
ax1 = fig.add_subplot(131, projection='polar')
ax1.scatter(data[:, 2], data[:, 0]*np.sin(data[:, 1]), s=0.25, alpha=0.25, c='k')
ax1.set_xlabel(r'$\phi$')
ax1.set_xticks([np.pi/4., 3.*np.pi/4., 5.*np.pi/4., 7.*np.pi/4.])
ax1.set_ylabel(r'Impact Parameter')
ax1.set_title('Incident Photon Positions On x-y Plane)')

############
ax2 = fig.add_subplot(132)
ax2.hist(data[:, 1], bins=50, color='k', histtype='step', density=True)
x = np.linspace(0, 1, 50)
theta = np.arcsin(2.0/5.0*x)
ax2.plot(theta, 2.*5.**2.*np.sin(theta)*np.cos(theta)/2.0**2., 'r--', label=r'$P(\theta)$')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel('n')
ax2.set_title(r'$\theta$ Position of Incident Photons')
ax2.legend()

############
ax3 = fig.add_subplot(133)
x = data[:, 0] * np.sin(data[:, 1]) * np.cos(data[:, 2])
z = data[:, 0] * np.cos(data[:, 1])
xhat = np.sin(data[:, 1]) * np.cos(data[:, 2]) * data[:, 3] \
       + np.cos([data[:, 1]]) * np.cos(data[:, 2]) * data[:, 4] \
       - np.sin(data[:, 2]) * data[:, 5]
zhat = np.cos(data[:, 1]) * data[:, 3] - np.sin(data[:, 1]) * data[:, 4]
ax3.quiver(x[:50], z[:50], xhat[0][:50], zhat[:50], units='xy', angles='xy', scale_units='xy', scale=1., alpha=0.25)

xcirc = np.linspace(-5, 5, 100)
ycirc = np.sqrt(25. - xcirc**2.)
ax3.plot(xcirc, ycirc, 'r--', label='Gas Surface')

ax3.set_xlabel('x-axis')
ax3.set_ylabel('z-axis')
ax3.set_xlim((-5, 5))
ax3.set_ylim((0, 5))
ax3.set_title('Photon Direction Vectors at Incidence')
ax3.legend()

plt.tight_layout()
plt.show()
