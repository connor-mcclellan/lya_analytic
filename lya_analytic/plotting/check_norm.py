import h5py
import numpy as np
import pdb

'''
Calculates the sum of the surface flux over frequency and time.
'''

n_r, n_sigma, n_t = 1, 101, 100
a = h5py.File('./outputs/r{}_sigma{}_t{}.hdf5'.format(n_r, n_sigma, n_t), 'r')

r = a['r'][:]
sigma = a['sigma'][:]
t = a['t'][:]

H_sigma = np.zeros(len(sigma))
dt = np.diff(t)[0]

for j in range(len(t)):
    H_dump = a['H_r0_t{}'.format(j)][:]
    H_sigma += dt * np.abs(H_dump.real)

H = 0.
d_sigma = np.diff(sigma)[0]
for i in range(len(sigma)):
    H += d_sigma * H_sigma[i] * 4. * np.pi * r**2.


print("summed H = {}".format(H))
