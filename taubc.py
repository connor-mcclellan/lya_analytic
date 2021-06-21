# Produces a single-panel plot that shows Monte Carlo subtracted from
# Djikstra's H0 solution, which should equal H_bc. H_bc is overplotted.
# Three different datasets are shown varying line center optical depth \tau_0.
# Axes are scaled by (a\tau)^1/3 to show agreement with analytic estimate.

from pa_plots import bin_x, comparison_plot, get_input_info
from solutions.util import read_bin, voigtx_fast, Line, Params, scinot
import pickle
import numpy as np
import astropy.constants as c
import pdb
import matplotlib.pyplot as plt


# Line parameters object
lya = Line(1215.6701, 0.4164, 6.265e8)
color = ['xkcd:darkish red', 'xkcd:darkish green', 'xkcd:golden rod', 'xkcd:blue', 'xkcd:grey']
alpha = 0.8

if __name__ == "__main__":

    filenames = [
       'tau0_100000.0_xinit_0.0_temp_10000.0_probabs_0.0',
       '1M tau0_1000000.0_xinit_0.0_temp_10000.0_probabs_0.0',
       '1M tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0',
    ]

    data_dir = '/home/connor/Documents/lya_analytic/data/'
    outputs = []

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for filename in filenames:
        # Load input parameters for the monte carlo data
        tau0, temp, xinit, prob_abs, radius = get_input_info(data_dir + filename + '/input')
        vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
        delta = lya.nu0 * vth / c.c.cgs.value
        a = lya.gamma / (4.0 * np.pi * delta)

        # Create parameters object for binning data
        p = Params(line=lya, temp=temp, tau0=tau0,
               energy=1., R=radius, sigma_source=0., n_points=1e4)
        L = 1.0

        # Plot title
        mytitle = r'$\tau_0=${}'.format(scinot(tau0))+'\n'+r'$x_{{\rm init}}={:.1f}$'.format(xinit)+'\n'+'$T=${}'.format(scinot(temp))

        # Load the monte carlo outputs
        mu, x, time = np.load(data_dir + filename + '/mu_x_time.npy')

        # Bin the data 
        xuniform, hp_xuniform, hsp_xuniform, hh_xuniform, xc, count, err, x0, xinit, ymin, ymax, phix_xc, hp_interp, hsp_interp, hh_interp, a, tau0 = bin_x(x, 64, mytitle, filename, tau0, xinit, temp, radius, L, delta, a, p, mcgrid=True)
        h0_minus_mc = hsp_xuniform - count

        ax.errorbar(xc, h0_minus_mc, yerr=err, fmt='.', label=r'$H_0 - \rm MC$', alpha=alpha, c=color[2])
        ax.plot(xuniform, hh_xuniform, '-.', alpha=alpha, c=color[1], label=r'$H_{\rm bc}$')
    plt.legend()
    plt.show()
        
