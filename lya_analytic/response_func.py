import h5py
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np

'''
Plots the J(n, sigma, omega)



'''

a = h5py.File('./outputs/Jnso/n8_sigma10000_omega64_rk_debug.hdf5', 'r')

sigma_indices = [4999, 5000, 6000]

n = a['n'][:]
omega = a['omega'][:]
sigma = a['sigma'][:]

for j in range(len(sigma_indices)):
    sigma_index = sigma_indices[j]
    sigma_val = sigma[sigma_index]
    response = []
    for i in range(len(omega)):
        c = np.abs(a['J_omega{}_n0'.format(i)][sigma_index])
        response.append(c)

    val, exp = '{:.1e}'.format(a['sigma'][sigma_index]).split('e+')
    plt.plot(omega, response, marker='o', ms=2, alpha=0.5, label='$\sigma='+val+'\\times 10^{'+str(int(exp))+'}$')

plt.ylabel('$|J (n=0, \sigma, \omega)|$')
plt.yscale('log')
plt.xlabel('$\omega$')
plt.legend()
plt.show()
