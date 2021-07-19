from constants import fundconst
from parameters import Parameters
import numpy as np
import matplotlib.pyplot as plt
fc=fundconst()
import pdb
from pathlib import Path
from util import construct_sol, gamma
from glob import glob
import matplotlib
import matplotlib.pylab as pl
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})
import pickle   

directory = Path('./data/coarse_test').resolve()
nmin = 18
Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax=20, mmax=10, nmin=nmin)
colors = pl.cm.viridis(np.linspace(0, 1, 20))
fig = plt.figure()

### TEMP ###
#p = pickle.load(open(directory/'parameters.p', 'rb'))
#fnames = sorted(glob(str(directory/'*.npy')))
#nmin, mmin = np.array(fnames[0].split('/n')[-1].split('.npy')[0].split('_m')).astype(int)
#nmax, mmax = np.array(fnames[-1].split('/n')[-1].split('.npy')[0].split('_m')).astype(int)
#ssoln = np.zeros((nmax, mmax))

#for n in range(nmin, nmax+1):
 #   for m in range(mmin, mmax+1):
 #       data = np.load(directory/"n{:03d}_m{:03d}.npy".format(n, m), allow_pickle=True).item()
 #       ssoln[n-1, m-1] = data['s']        

linewidths = np.logspace(np.log10(7), np.log10(2), 20)


for n in range(nmin, p.nmax+1):

#    gamma_sweep = -n**(-4/3)*ssoln[n-1][:mmax]
#    gamma_analytic = gamma(n, np.arange(mmin, mmax+1), p)

    gamma_sweep = -n**(-4/3)*ssoln[n-nmin-1][:p.mmax]
    plt.plot(np.arange(1, p.mmax+1), p.radius/fc.clight/(p.a*p.tau0)**(1/3.)/gamma_sweep, '-', lw=linewidths[n-1], c=colors[n-1])#, label='$\gamma$ sweep')

n=nmin
gamma_analytic = n**(-4/3)*gamma(n, np.arange(1, p.mmax+1), p)
plt.plot(np.arange(1, p.mmax+1), p.radius/fc.clight/(p.a*p.tau0)**(1/3.)/gamma_analytic, 'k--', lw=1, label='Analytic $\gamma_{nm}$')

sm = plt.cm.ScalarMappable(cmap=pl.cm.viridis, norm=plt.Normalize(vmin=1, vmax=20)) 
cbar = fig.colorbar(sm) 
cbar.ax.set_ylabel('n for numerical $\gamma_{nm}$', rotation=90)

bounds = ['1', '5', '10', '15', '20']
cbar.set_ticks(np.array(bounds).astype(float))
cbar.set_ticklabels(bounds)

plt.ylabel(r'$Rc^{-1} (a\tau_0)^{-1/3} n^{-4/3} \gamma_{nm}^{\ \ -1}$')
plt.xlabel('m')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.legend(frameon=False)
plt.show()

############



'''
for n in range(1, p.nmax+1):
    gamma_analytic = n**(-4/3)*gamma(n, np.arange(1, p.mmax+1), p)
    gamma_sweep = -n**(-4/3)*ssoln[n-1][:p.mmax]

    plt.plot(np.arange(1, p.mmax+1), 1/gamma_analytic, '--', alpha=0.5)#, label='$\gamma_{nm}$ analytic')
    plt.plot(np.arange(1, p.mmax+1), 1/gamma_sweep, '-', alpha=0.5)#, label='$\gamma$ sweep')

plt.ylabel('$n^{4/3}t_{nm}(s)$')
plt.xlabel('m')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.show()
'''


