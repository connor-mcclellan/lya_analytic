from pa_plots import bin_x, comparison_plot, get_input_info
from solutions.util import read_bin, Line, Params, scinot
import pickle
import numpy as np
import astropy.constants as c
import pdb

if __name__ == "__main__":

    filenames = [
#      '1M tau0_1000000.0_xinit_0.0_temp_10000.0_probabs_0.0',
#      '1M tau0_1000000.0_xinit_6.0_temp_10000.0_probabs_0.0',
#      '1M tau0_1000000.0_xinit_12.0_temp_10000.0_probabs_0.0',
       '1M tau0_100000.0_xinit_0.0_temp_10000.0_probabs_0.0',
       '1M tau0_1000000.0_xinit_0.0_temp_10000.0_probabs_0.0',
       '1M tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0',
    ]

    data_dir = '/home/connor/Documents/lya_analytic/steadystate/data/'
    generate_new = True
    vary_tau = True
    outputs = []

    for filename in filenames:
        lya = Line(1215.6701, 0.4164, 6.265e8)
        tau0, temp, xinit, prob_abs, radius = get_input_info(data_dir + filename + '/input')
        vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
        delta = lya.nu0 * vth / c.c.cgs.value
        a = lya.gamma / (4.0 * np.pi * delta)

        p = Params(line=lya, temp=temp, tau0=tau0,
               energy=1., R=radius, sigma_source=0., n_points=1e4)
        L = 1.0

        mytitle = r'$\tau_0=${}'.format(scinot(tau0))+'\n'+r'$x_{{\rm init}}={:.1f}$'.format(xinit)+'\n'+'$T=${}'.format(scinot(temp))

        if generate_new:
            mu, x, time = np.load(data_dir + filename + '/mu_x_time.npy')  
            binx_output = bin_x(x, 64, mytitle, filename, tau0, xinit, temp, radius, L, delta, a, p)
            if vary_tau:
                pickle.dump(binx_output, open('binx_output_tau{:.1f}.p'.format(tau0), 'wb'))
            else:
                pickle.dump(binx_output, open('binx_output_xinit{:.1f}.p'.format(xinit), 'wb'))
            outputs.append(binx_output)
        else:
            if vary_tau:
                binx_output = pickle.load(open('binx_output_tau{:.1f}.p'.format(tau0), 'rb'))
            else:
                binx_output = pickle.load(open('binx_output_xinit{:.1f}.p'.format(xinit), 'rb'))
            outputs.append(binx_output)

    comparison_plot(*outputs, tauax=vary_tau, divergent=False)
