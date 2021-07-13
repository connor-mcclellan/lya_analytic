from pathlib import Path
from util import construct_sol, waittime, get_Pnm
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

# Monte Carlo
'''
print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, marker='+', label=r'MC $\tau_0=10^7$', alpha=0.7)


print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_1000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, marker='+', label=r'MC $\tau_0=10^6$', alpha=0.7)


print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_100000.0_xinit_0.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, marker='+', label=r'MC $\tau_0=10^5$', alpha=0.7)
'''
print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_1000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, marker='+', label=r'MC $x_s=0.0$', alpha=0.7)

print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_1000000.0_xinit_6.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, marker='+', label=r'MC $x_s=6.0$', alpha=0.7)

'''
print('Monte Carlo...')
mc_dir = '/home/connor/Documents/lya_analytic/data/1M tau0_1000000.0_xinit_12.0_temp_10000.0_probabs_0.0/'
(bincenters, n), _, _ = mc_wait_time(mc_dir)
plt.scatter(bincenters, n, marker='+', label=r'$x_s=12.0$', alpha=0.7)


# Eigenfunctions
print('Loading eigenfunctions...')
directory = Path('./data/210521_m500').resolve()
Jsoln, ssoln, intJsoln, p = construct_sol(directory, 20, 500)
tlc = p.radius/fc.clight

# Eigenfunctions wait time
print('Calculating wait time...')
t = tlc * np.arange(0.1,140.0,0.1)
P = waittime(Jsoln, ssoln, intJsoln, t, p)

plt.plot(t/tlc, P*tlc, label=r'Eigenfunction expansion, $\tau_0=10^7$')
'''

# Eigenfunctions
print('Loading eigenfunctions...')
directory = Path('./data/tau1e6_xinit0').resolve()
Jsoln, ssoln, intJsoln, p = construct_sol(directory, 12, 500)
tlc = p.radius/fc.clight

# Eigenfunctions wait time
print('Calculating wait time...')
t = tlc * np.arange(0.1,140.0,0.1)
P = waittime(Jsoln, ssoln, intJsoln, t, p)
#plt.plot(t/tlc, P*tlc, label=r'Eigenfunction expansion, $\tau_0=10^6$')
plt.plot(t/tlc, P*tlc, label=r'Eigenfunction expansion, $x_s=0.0$')


# Eigenfunctions
print('Loading eigenfunctions...')
directory = Path('./data/tau1e6_xinit6').resolve()
Jsoln, ssoln, intJsoln, p = construct_sol(directory, 12, 500)
tlc = p.radius/fc.clight

# Eigenfunctions wait time
print('Calculating wait time...')
t = tlc * np.arange(0.1,140.0,0.1)
P = waittime(Jsoln, ssoln, intJsoln, t, p)
#plt.plot(t/tlc, P*tlc, label=r'Eigenfunction expansion, $\tau_0=10^6$')
plt.plot(t/tlc, P*tlc, label=r'Eigenfunction expansion, $x_s=6.0$')

# Exponential using lowest order eigenfrequency
#print('Plotting exponential falloff...')
#Pnmsoln = get_Pnm(ssoln, intJsoln, p)
expfalloff = -ssoln[0, 0] * Pnmsoln[0, 0] * np.exp(ssoln[0, 0] * t)
#plt.plot(t/tlc, expfalloff*tlc, '--', label=r'$\gamma_{00}P_{00}e^{-\gamma_{00} t}$')

#tdiff = tlc * np.cbrt(p.a * p.tau0)
#plt.plot(t/tlc, np.exp(-(t/tdiff)**2.)*expfalloff * tlc, '--'), label=r'$Fitting $')


# Try fitting an inverse gamma distribution
#def invgamma(x, alpha, beta):
#    return beta**alpha/gamma(alpha) * (1/x)**(alpha+1) * np.exp(-beta/x)

#def gammafunc(x, alpha, beta):
#    return beta**alpha * x**(alpha - 1) * np.exp(-beta*x) / gamma(alpha)


#fit1 = gammafunc(t, -tdiff*ssoln[0, 0], -ssoln[0, 0])

#pdb.set_trace()
#for a in np.linspace(0.5, 1.5, 10):
#a = .98
#b = 1.2

#fit2 = gammafunc(t, -a*tdiff*ssoln[0, 0], -b*ssoln[0, 0])

#scale = 5
#stored_err = np.sum(np.abs((P[P>0] - fit2[P>0])))#/P[P>0]))
#while True:
#    randa = (np.random.random()-0.5)*scale
#    randb = (np.random.random()-0.5)*scale
#    fit2 = gammafunc(t, -(a+randa)*tdiff*ssoln[0, 0], -(b+randb)*ssoln[0, 0])
#    err = np.sum(np.abs((P[P>0] - fit2[P>0])))#/P[P>0]))
#    if np.abs(err - stored_err) < 1e-6:
#        break
#    if err < stored_err and np.random.random() > 0.3:
#        a += randa
#        b += randb
#        scale = 0.5*scale
#        stored_err = err
#        print(scale, err)


#print(a, b)
#fit2 = gammafunc(t, -a*tdiff*ssoln[0, 0], -b*ssoln[0, 0])
#plt.plot(t/tlc, fit1*tlc, alpha=0.5, label='a=1')
#plt.plot(t/tlc, fit2*tlc, alpha=0.5, label=r'$\Gamma(a, b)$', ls='-.')

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
