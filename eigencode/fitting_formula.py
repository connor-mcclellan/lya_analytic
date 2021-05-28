from pathlib import Path
from util import construct_sol, waittime, get_Pnm
from constants import fundconst,lymanalpha
from mc_visual import mc_wait_time
import matplotlib.pyplot as plt
import numpy as np
import pdb

fc=fundconst()
la=lymanalpha()

# Monte Carlo
print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1m_tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, s=1, label='Monte Carlo')

# Eigenfunctions
print('Loading eigenfunctions...')
directory = Path('./data/210521_m500').resolve()
Jsoln, ssoln, intJsoln, p = construct_sol(directory, 20, 500)
tlc = p.radius/fc.clight

# Eigenfunctions wait time
print('Calculating wait time...')
t = tlc * np.arange(0.1,140.0,0.1)
P = waittime(Jsoln, ssoln, intJsoln, t, p)
plt.plot(t/tlc, P*tlc, label='efunctions')

# Exponential using lowest order eigenfrequency
print('Plotting exponential falloff...')
Pnmsoln = get_Pnm(ssoln, intJsoln, p)
expfalloff = -ssoln[0, 0] * Pnmsoln[0, 0] * np.exp(ssoln[0, 0] * t) * p.Delta
plt.plot(t/tlc, expfalloff*tlc, '--', label=r'$s_{00}P_{00}e^{s_{00} t}$')

# Plotting
plt.yscale('log')
plt.xlabel(r'$ct/R$')
plt.ylabel('$(R/c)\, P(t)$')
plt.legend()
plt.show()
