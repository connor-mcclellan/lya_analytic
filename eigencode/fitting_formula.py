from pathlib import Path
from util import construct_sol, waittime, get_Pnm, scinot
from constants import fundconst,lymanalpha
from mc_visual import mc_wait_time
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np
import pdb

fc=fundconst()
la=lymanalpha()

def monte_carlo(tau0, xinit):
    print('Monte Carlo...')
    mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_{:.1f}_xinit_{:.1f}_temp_10000.0_probabs_0.0/'.format(tau0, xinit)
    (bincenters, n), _, _ = mc_wait_time(mc_dir)
    plt.scatter(bincenters, n, marker='+', label=r'MC $\tau_0={}'.format(scinot(tau0).split(r'\times ')[1]), alpha=0.7)

def efunc(path, nmax, mmax):
    print('Loading eigenfunctions...')
    directory = Path(path).resolve()
    Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax, mmax)
    Pnmsoln = get_Pnm(ssoln, intJsoln, p)
    tlc = p.radius/fc.clight
    tau0 = p.tau0

    # Eigenfunctions wait time
    print('Calculating wait time...')
    t = tlc * np.arange(0.1,140.0,0.1)
    P = waittime(Jsoln, ssoln, intJsoln, t, p)
    plt.plot(t/tlc, P*tlc, label=r'Eigenfunction expansion, $\tau_0={}'.format(scinot(tau0).split(r'\times ')[1]))

    # Fitting function
    expfalloff = -ssoln[0, 0] * Pnmsoln[0, 0] * np.exp(ssoln[0, 0] * t)
    tdiff = tlc * np.cbrt(p.a * p.tau0)
    plt.plot(t/tlc, np.exp(-(t/tdiff)**2.)*expfalloff * tlc, '--', label=r'Fitting function, $\tau_0={}'.format(scinot(tau0).split(r'\times ')[1]))



monte_carlo(1e6, 0.0)
monte_carlo(1e7, 0.0)
efunc('./data/tau1e6_xinit0', 20, 500)
efunc('./data/210521_m500', 20, 500)


# Plotting
#plt.title('$x_s=0$')
plt.title(r'$\tau_0=10^6$')
plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$ct/R$')
plt.ylabel('$(R/c)\, P(t)$')
plt.ylim(1e-6, 5e-1)
plt.legend()
plt.show()
