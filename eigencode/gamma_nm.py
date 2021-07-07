from constants import fundconst
from parameters import Parameters
import numpy as np
import matplotlib.pyplot as plt
fc=fundconst()
import pdb
from pathlib import Path
from util import construct_sol
from glob import glob
import pickle

def gamma(n, m, p): 
     return 2**(-1/3) * np.pi**(13/6)*n**(4/3)*(m-7/8)**(2/3)*fc.clight/p.radius/(p.a * p.tau0)**(1/3)    

directory = Path('./data/210521_m500').resolve()
nmin = 1
Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax=20, mmax=500)

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

for n in range(nmin, p.nmax+1):
    gamma_analytic = n**(-4/3)*gamma(n, np.arange(1, p.mmax+1), p)
#    gamma_sweep = -n**(-4/3)*ssoln[n-1][:mmax]
#    gamma_analytic = gamma(n, np.arange(mmin, mmax+1), p)
    gamma_sweep = -n**(-4/3)*ssoln[n-1][:p.mmax]

    plt.plot(np.arange(1, p.mmax+1), 1/gamma_analytic, '--', alpha=0.5)#, label='$\gamma_{nm}$ analytic')
    plt.plot(np.arange(1, p.mmax+1), 1/gamma_sweep, '-', alpha=0.5)#, label='$\gamma$ sweep')

plt.ylabel('$t_{nm}(s)$')
plt.xlabel('m')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.legend()
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


