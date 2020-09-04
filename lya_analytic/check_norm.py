import h5py
import numpy as np

n_r, n_sigma, n_t = 1, 1000, 1000
a = h5py.File('./outputs/r{}_sigma{}_t{}.hdf5'.format(n_r, n_sigma, n_t), 'r')

r = a['r'][:]
sigma = a['sigma'][:]
t = a['t'][:]

H_t = np.zeros(len(t))
d_sigma = np.diff(sigma)[0]
for i in range(len(sigma)):
    Hdump = a['H_r0_sigma{}'.format(i)][:]
    H_t += d_sigma * Hdump * 4. * np.pi * r**2.

H = 0.
dt = np.diff(t)[0]
for j in range(len(t)):
    H += dt * H_t

print("summed H = {}".format(H))
