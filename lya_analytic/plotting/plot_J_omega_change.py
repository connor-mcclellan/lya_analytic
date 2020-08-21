import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

path = './outputs/n8_sigma1e6_omega64/'
n = np.load(path+'n_grid.npy')
omega = np.load(path+'omega_grid.npy')
sigma = np.load(path+'sigma_grid.npy')

# Sum & discretized integral
for k in range(len(n)):

    plt.figure()
    colors = pl.cm.jet(np.linspace(0, 1, len(omega)))
    for l in range(len(omega)):

        # Load in data for this n and omega
        Jdump = np.load(path+'J_omega{}_n{}.npy'.format(l, k))
        J_real = Jdump[:, 0]
        J_imag = Jdump[:, 1]

        plt.plot(sigma, J_real, '-', alpha=0.25, color=colors[l])
        plt.plot(sigma, J_real, '--', alpha=0.25, color=colors[l])

    plt.title('J Coefficients for n={}'.format(n[k]))
    plt.xlabel('sigma')
    plt.ylabel('J(n, sigma, omega)')
    plt.show()
