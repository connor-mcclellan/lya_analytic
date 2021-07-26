from pathlib import Path
from util import construct_sol, waittime, get_Pnm, scinot
from constants import fundconst,lymanalpha
from mc_visual import mc_wait_time
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})
import numpy as np
import pdb

fc=fundconst()
la=lymanalpha()

colorlegend = []

def monte_carlo(tau0, xinit, c):
    print('Monte Carlo...')
    mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_{:.1f}_xinit_{:.1f}_temp_10000.0_probabs_0.0/'.format(tau0, xinit)
    (bincenters, n), _, _ = mc_wait_time(mc_dir)
    plt.scatter(bincenters, n, marker='s', s=2, label=r'MC $x_s={:.1f}$, $\tau_0={}'.format(xinit, scinot(tau0).split(r'\times ')[1]), alpha=0.7, color=c)
    return min(bincenters)

def efunc(path, nmax, mmax, c, xmin=1):
    print('Loading eigenfunctions...')
    directory = Path(path).resolve()
    Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax, mmax)
    Pnmsoln = get_Pnm(ssoln, intJsoln, p)
    tlc = p.radius/fc.clight
    tau0 = p.tau0
    xinit = p.xsource
    pdb.set_trace()
    # Eigenfunctions wait time
    print('Calculating wait time...')
    t = tlc * np.arange(xmin,140.0,0.1)
    P = waittime(Jsoln, ssoln, intJsoln, t, p)
    plt.plot(t/tlc, P*tlc, alpha=0.7, label=r'Eigenfunction expansion, $x_s={:.1f}$, $\tau_0={}'.format(xinit, scinot(tau0).split(r'\times ')[1]), c=c)

    # Fitting function
    expfalloff = -ssoln[0, 0] * Pnmsoln[0, 0] * np.exp(ssoln[0, 0] * t)
    tdiff = tlc * np.cbrt(p.a * p.tau0)
    plt.plot(t/tlc, np.exp(-(tdiff/t/tlc)**2) * expfalloff * tlc, '--', lw=2, alpha=0.7, label=r'Fitting function, $x_s={:.1f}$, $\tau_0={}'.format(xinit, scinot(tau0).split(r'\times ')[1]), c=c)
    colorlegend.append(Patch(facecolor=c, label=r'$x_s={:.1f}$'.format(xinit)))


#colors = ['r', 'g', 'b', 'purple', 'limegreen', 'c']
#for i, xinit in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]):
#    monte_carlo(1e6, xinit, colors[i])
#    colorlegend.append(Patch(facecolor=colors[i], label=r'$x_s={:.1f}$'.format(xinit)))

#xmin = monte_carlo(1e6, 0.0, '#696969')
monte_carlo(1e7, 0.0, 'k')
#efunc('./data/tau1e6_xinit0', 20, 500, '#696969', xmin=xmin)
#efunc('./data/tau1e6_xinit6', 20, 500, 'k', xmin=xmin)
efunc('./data/210521_m500', 20, 500, 'k')

formatlegend = [Line2D([1], [0], color='k', label='Eigenfunctions'), Line2D([1], [1], ls='--', color='k', label='Fitting function'), Line2D([1], [0], color='k', ls='None', marker='s', ms=2, label='Monte Carlo')]
fmtlegend = plt.legend(handles=formatlegend, loc='upper left', bbox_to_anchor=(0.7, 0.8), fontsize='x-small', frameon=False)
clegend = plt.legend(handles=colorlegend, loc='lower left', bbox_to_anchor=(0.7, 0.8), fontsize='small', frameon=False)
plt.gca().add_artist(clegend)
plt.gca().add_artist(fmtlegend)

# Plotting
#plt.title('$x_s=0$')
#plt.title(r'$\tau_0=10^6$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$ct/R$')
plt.ylabel('$(R/c)\, P(t)$')
#plt.ylim(1e-6, 5e-1)
plt.show()
