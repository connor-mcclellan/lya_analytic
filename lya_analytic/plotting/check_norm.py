import h5py
import numpy as np
import pdb
from solutions.util import Params, Line

'''
Calculates the sum of the surface flux over frequency and time.
'''

# Physical parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e5)


n_r, n_sigma, n_t = 1, 1001, 100
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
    H += d_sigma * H_sigma[i] * 4. * np.pi * r**2. * p.delta**2. * p.phi(sigma[i]) * np.sqrt(3./2)


print("summed H = {}".format(H))
