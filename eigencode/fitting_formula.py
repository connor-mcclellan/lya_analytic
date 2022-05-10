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
import matplotlib.pylab as pl
colors = pl.cm.inferno(np.linspace(0.0, .65, 10))

fc=fundconst()
la=lymanalpha()

colorlegend = []

def monte_carlo(tau0, xinit, c):
    print('Monte Carlo...')
    mc_dir = '/home/connor/Documents/lya_analytic/steadystate/data/1M tau0_{:.1f}_xinit_{:.1f}_temp_10000.0_probabs_0.0/'.format(tau0, xinit)
    (bincenters, n), _, _ = mc_wait_time(mc_dir)
    plt.scatter(bincenters, n, marker='s', s=10, label=r'MC $x_s={:.1f}$, $\tau_0={}'.format(xinit, scinot(tau0).split(r'\times ')[1]), color=c, linewidths=0.6, edgecolors='white', zorder=5+tau0)
    return min(bincenters)

def efunc(path, nmax, mmax, c, xmin=1):
    print('Loading eigenfunctions...')
    directory = Path(path).resolve()
    Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax, mmax)
    Pnmsoln = get_Pnm(ssoln, intJsoln, p)
    tlc = p.radius/fc.clight
    tau0 = p.tau0
    xinit = p.xsource

    # Eigenfunctions wait time
    print('Calculating wait time...')
    t = tlc * np.arange(xmin,140.0,0.1)
    P = waittime(Jsoln, ssoln, intJsoln, t, p)
    plt.plot(t/tlc, P*tlc, alpha=0.8, zorder=tau0, label=r'Eigenfunction expansion, $x_s={:.1f}$, $\tau_0={}'.format(xinit, scinot(tau0).split(r'\times ')[1]), c=c)

    # Fitting function
    expfalloff = -ssoln[0, 0] * Pnmsoln[0, 0] * np.exp(ssoln[0, 0] * t)
    tdiff = tlc * np.cbrt(p.a * p.tau0)
    plt.plot(t/tlc, np.exp(-(tdiff/t/tlc)**2) * expfalloff * tlc, '--', zorder=tau0, alpha=0.8, lw=3, label=r'Fitting function, $x_s={:.1f}$, $\tau_0={}'.format(xinit, scinot(tau0).split(r'\times ')[1]), c=c)
    colorlegend.append(Patch(facecolor=c, label=r'$\tau_0={}'.format(scinot(tau0).split(r'\times ')[1])))


#colors = ['r', 'g', 'b', 'purple', 'limegreen', 'c']
#for i, xinit in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]):
#    monte_carlo(1e6, xinit, colors[i])
#    colorlegend.append(Patch(facecolor=colors[i], label=r'$x_s={:.1f}$'.format(xinit)))

c1 = '#f02626'
c2 = '#000e38'

xmin1 = monte_carlo(1e6, 0.0, c1)
xmin2 = monte_carlo(1e7, 0., c2)

efunc('./data/tau1e6_xinit0', 20, 500, c1, xmin=xmin1)
efunc('./data/210521_m500', 20, 500, c2, xmin=xmin2)


formatlegend = [Line2D([1], [0], color='k', label='Eigenfunctions'), Line2D([1], [1], ls='--', color='k', label='Fitting function'), Line2D([1], [0], color='k', ls='None', marker='s', ms=2, label='Monte Carlo')]
fmtlegend = plt.legend(handles=formatlegend, loc='upper left', bbox_to_anchor=(0.75, 0.85), fontsize='x-small', frameon=False)
clegend = plt.legend(handles=colorlegend, loc='lower left', bbox_to_anchor=(0.75, 0.85), fontsize='small', frameon=False)
plt.gca().add_artist(clegend)
plt.gca().add_artist(fmtlegend)

# Plotting
#plt.title('$x_s=0$')
#plt.title(r'$\tau_0=10^6$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$ct/R$')
plt.ylabel('$(R/c)\, P(t)$')
plt.ylim(1e-6, 5e-1)
plt.show()
