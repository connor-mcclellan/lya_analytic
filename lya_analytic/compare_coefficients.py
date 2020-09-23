import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np
import argparse

'''
Plots any number of J_n_sigma_omega Fourier coefficient outputs on the same 
graph, to see how changes to the sigma grids affect the amplitudes of the 
Fourier coefficients. The omega grids must be the same. A hard limit on n has
been set to 4, which can be changed below.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--inputfiles', '-i', type=str, nargs='+', help='file outputs to plot')
args = parser.parse_args()

a = h5py.File('./outputs/'+args.inputfiles[0], 'r')
omega = a['omega'][:]
a.close()

linestyles = ['-', '--', '-.', ':']

for l in range(len(omega)):  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    for j, inputfile in enumerate(args.inputfiles):
        a = h5py.File('./outputs/'+inputfile, 'r')
        n = a['n'][:4]
        sigma = a['sigma'][:]

        colors = pl.cm.jet(np.linspace(0, 1, len(n)))

        # TO DO:
        # if this_file_omega not equal to omega: abort with error message

        for k in range(len(n)):
            # Load in data for this n and omega
            J = a['J_omega{}_n{}'.format(l, k)][:]
            ax1.plot(sigma, np.abs(J.real), linestyles[j], alpha=0.25, color=colors[k])
            ax2.plot(sigma, np.abs(J.imag), linestyles[j], alpha=0.25, color=colors[k])
        a.close()

    sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=4)) # hardcoded n limit
    cbar = fig.colorbar(sm)
    cbar.ax.set_ylabel('n', rotation=90)

    ax1.set_title('Real part')
    ax2.set_title('Imag part')

    lines = [matplotlib.lines.Line2D([0], [0], color='k', linestyle=ls) for ls in linestyles]
    escaped_names = [text.translate(str.maketrans({"_": r"\_"})) for text in args.inputfiles]
    plt.legend(lines, escaped_names)
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
    plt.show()
   # plt.savefig('./plots/animations/compare_coeffs_frame{:03d}.png'.format(l))
    plt.close()
