import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True

data = h5py.File('../outputs/n16_sigma100000_omega128.hdf5', 'r')

n = data['n']
omega = data['omega']
sigma = data['sigma']


for l in range(len(omega)):  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    colors = pl.cm.jet(np.linspace(0, 1, len(n)))
    for k in range(len(n)):
        # Load in data for this n and omega
        J = data['J_omega{}_n{}'.format(l, k)][:]
        ax1.plot(sigma, np.abs(J.real), '-', alpha=0.25, color=colors[k])
        ax2.plot(sigma, np.abs(J.imag), '-', alpha=0.25, color=colors[k])
    sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=len(n)))
    cbar = fig.colorbar(sm)
    cbar.ax.set_ylabel('n', rotation=90)

    ax1.set_title('Real part')
    ax2.set_title('Imag part')

    plt.suptitle('J Coefficients for $\omega={:.4f}$'.format(omega[l]))
    ax1.set_xlabel('$\sigma$')
    ax2.set_xlabel('$\sigma$')
    ax1.set_ylabel('$J(n, \sigma, \omega)$')
    ax2.set_ylabel('$J(n, \sigma, \omega)$')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim(bottom=1e-52, top=1e-32)
    ax2.set_ylim(bottom=1e-52, top=1e-32)
    plt.subplots_adjust(top=0.897,
bottom=0.091,
left=0.073,
right=0.953,
hspace=0.2,
wspace=0.103)
   # plt.show()
    plt.savefig('../plots/animations/n_each_omega_frame{:03d}.png'.format(l))
    plt.close()
