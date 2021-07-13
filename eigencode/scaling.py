import matplotlib.pyplot as plt
import numpy as np
from mc_visual import mc_wait_time
import matplotlib
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from wait_time import *
from util import get_Pnm
import pdb
from pathlib import Path

dirs = [
  './data/lowestorder_tau1e5',
  './data/lowestorder_tau1e6',
  './data/lowestorder_tau1e7',
  './data/lowestorder_tau1e8',
  './data/lowestorder_tau1e9',
]

data = np.zeros((4, len(dirs)))
for k, directory in enumerate(dirs):

    directory = Path(directory).resolve()
    Jsoln, ssoln, intJsoln, p = construct_sol(directory, 1, 1)
    Pnm = get_Pnm(ssoln, intJsoln, p)[0][0]
    eigenfreq = ssoln[0][0]

    taustr = str(directory).split('_tau')[-1].split('.npy')[0]
    if taustr in ['1e5', '1e6', '1e7']:
        mcdir = '/home/connor/Documents/lya_analytic/data/1M tau0_'+str(float(taustr))+'_xinit_0.0_temp_10000.0_probabs_0.0/'
        tdata, _, poly = mc_wait_time(mcdir)
        t, n = tdata
        tlc = p.radius / fc.clight
        plt.scatter(t, n, label='MC', marker='+', s=3)
        plt.plot(t, np.exp(poly[1]) * np.exp(poly[0]*t), label='Fit', alpha=0.7)
        plt.plot(t, -eigenfreq * tlc *Pnm*np.exp(tlc * eigenfreq * t), label='Lowest order eigenfrequency', alpha=0.7)
        plt.yscale('log')
        plt.title(r"$\tau_0=${}".format(taustr))
        plt.legend()
        plt.show()

    data[0][k] = p.a*p.tau0 # Optical depth
    data[1][k] = 1/(-poly[0]) # Monte Carlo exponential fit
    data[2][k] = fc.clight/p.radius * 1/(-eigenfreq) # Lowest order eigenfrequency
    data[3][k] = p.a

norm = data[2][-1]/(fc.clight/p.radius*(data[0][-1])**(1./3))
plt.figure(figsize=(5, 3))
plt.plot(data[0]/data[3], norm*fc.clight/p.radius*(data[0])**(1./3), alpha=0.5, label=r'$t \propto (a\tau_0)^{1/3}$', ls=':', c='k')
plt.plot(data[0]/data[3], data[2], '-', marker='s', ms=3, alpha=0.5, label=r'$\gamma_{00}^{\ \ -1}$')
plt.scatter(data[0][:3]/data[3][:3], data[1][:3], label='Monte Carlo', marker='+', c='k')

#plt.plot(data[0], fc.clight/p.radius*(data[0])**(1./2), alpha=0.5, label=r'$t = (a\tau)^{1/2}$', ls=':', c='k')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylabel('$ct/R$')
plt.xlabel(r'$\tau_0$')
plt.tight_layout()

plt.show()
