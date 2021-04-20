from pa_plots import bin_x, comparison_plot, get_input_info
from solutions.util import read_bin, voigtx_fast, Line, Params, scinot
import pickle
import numpy as np
import astropy.constants as c

if __name__ == "__main__":

    filenames = [
      '1M tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0',
      '1M tau0_10000000.0_xinit_6.0_temp_10000.0_probabs_0.0',
      '1M tau0_10000000.0_xinit_12.0_temp_10000.0_probabs_0.0',
    ]

    data_dir = '/home/connor/Documents/lya_analytic/data/'
    generate_new = True

    lya = Line(1215.6701, 0.4164, 6.265e8)
    p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e4)
    L = 1.0
    outputs = []

    for filename in filenames:
        tau0, temp, xinit, prob_abs, radius = get_input_info(data_dir + filename + '/input')
        vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
        delta = lya.nu0 * vth / c.c.cgs.value
        a = lya.gamma / (4.0 * np.pi * delta)
    
        mytitle = r'$\tau_0=${}'.format(scinot(tau0))+'\n'+r'$x_{{\rm init}}={:.1f}$'.format(xinit)+'\n'+'$T=${}'.format(scinot(temp))

        if generate_new:
            mu, x, time = np.load(data_dir + filename + '/mu_x_time.npy')  
            binx_output = bin_x(x, 64, mytitle, filename, tau0, xinit, temp, radius, L, delta, a, p)
            pickle.dump(binx_output, open('binx_output_xinit{:.1f}.p'.format(xinit), 'wb'))
            outputs.append(binx_output)
        else:
            binx_output = pickle.load(open('binx_output_xinit{:.1f}.p'.format(xinit), 'rb'))
            outputs.append(binx_output)

    comparison_plot(*outputs)
