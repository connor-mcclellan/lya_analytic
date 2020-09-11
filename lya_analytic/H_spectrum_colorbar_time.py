import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np

a = h5py.File('./outputs/r1_sigma1001_t100.hdf5', 'r')

sigma = a['sigma'][:]
time = a['t'][:]


fig = plt.figure()
colors = pl.cm.jet(np.linspace(0, 1, len(time)))

for i in range(len(time)):
    data = a['H_r0_t{}'.format(i)][:]
    plt.plot(sigma, data.real, '-', alpha=0.25)

sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=min(time), vmax=max(time)))
cbar = fig.colorbar(sm)
cbar.ax.set_ylabel('time', rotation=90)

plt.xlabel('$\sigma$')
plt.ylabel('$H$')
plt.yscale('log')
plt.show()
