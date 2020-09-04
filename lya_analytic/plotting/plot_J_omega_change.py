import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

data = h5py.File('../outputs/n16_sigma100000_omega128.hdf5', 'r')

n = data['n']
omega = data['omega']
sigma = data['sigma']

for k in range(len(n)):

    plt.figure()
    colors = pl.cm.jet(np.linspace(0, 1, len(omega)))
    for l in range(len(omega)):

        # Load in data for this n and omega
        J = data['J_omega{}_n{}'.format(l, k)][:]

        plt.plot(sigma, np.abs(J.real), '-', alpha=0.25, color=colors[l])
#        plt.plot(sigma, J.imag, '--', alpha=0.25, color=colors[l])

    plt.title('J Coefficients for n={}'.format(n[k]))
    plt.xlabel('sigma')
    plt.ylabel('J(n, sigma, omega)')
    plt.yscale('log')
    plt.show()
