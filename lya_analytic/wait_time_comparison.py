import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import h5py

# Temporarily hardcode the frequency point to check. This will be changed
sigma_index = 0

parser = argparse.ArgumentParser('Make a plot from H_r_sigma_t data comparing the wait time distribution of several different outputs.')
parser.add_argument('-i', '--inputfiles', type=str, nargs='+')
parser.add_argument('-o', '--omegas', type=str, nargs='+')
args = parser.parse_args()

fig = plt.figure()
colors = pl.cm.jet(np.linspace(0, 1, len(args.inputfiles)))

# Load in the flux data for one of the files
for i, inputfile in enumerate(args.inputfiles):
    a = h5py.File(inputfile, 'r')
    sigma = a['sigma'][:]
    time = a['t'][:]

    # TO DO: Integrate over frequency to produce true wait time distribution
    # Currently, this is a representative sample at only a few frequencies
    H = a['H_r0_sigma{}'.format(sigma_index)][:]

    # Plot a labeled line here
    plt.plot(time, H.real, '-', marker='s', markersize=2, alpha=0.5, color=colors[i], label='$N_{{\omega}}={}$'.format(args.omegas[i]))

val, exp = '{:.1e}'.format(a['sigma'][sigma_index]).split('e+')

plt.xlabel('$t$')
plt.ylabel('$H(r=10^{11}, \sigma='+val+'\\times 10^{'+str(int(exp))+'}, t)$')
#plt.yscale('log')
plt.title('Surface Flux v. Time by Number of $\omega$ Points')
plt.legend()
plt.show()
