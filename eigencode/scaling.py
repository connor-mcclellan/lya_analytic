import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex']=True
from wait_time import *

data = np.array([
  [1e5, 1e6, 1e7, 1e8],
  [1.1, 3.2, 8.0, 16.8],
  [3.1343, 9.8665, 25.174, 54.448],
])

filenames = [
  './eigenmode_data_xinit0.0_tau1e5.npy',
  './eigenmode_data_xinit0.0_tau1e6.npy',
  './eigenmode_data_xinit0.0_tau1e7.npy',
  './eigenmode_data_xinit0.0_tau1e8.npy',
]

data = np.zeros((4, len(filenames)))

for k, filename in enumerate(filenames):
    array = np.load(filename, allow_pickle=True, fix_imports=True)
    energy, temp, tau0, radius, alpha_abs, prob_dest, xsource, nmax, nsigma, nomega, tdiff, sigma, ssoln, Jsoln = array

    ### For deprecated file format adjustment
    nmax -= 1
    ssoln = ssoln[1:]
    Jsoln = Jsoln[1:]
    ###

    p = parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
    Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)
    times = p.radius/fc.clight * np.arange(0.1,140.0,0.1)

    P = 0.0*times
    for i in range(times.size):
      t=times[i]
      for n in range(1,p.nmax+1):
        for m in range(0,20):
          P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)

    data[0][k] = p.a*tau0 # Optical depth
    data[1][k] = fc.clight/p.radius * times[np.argmax(P)] # Peak of distribution
    data[2][k] = fc.clight/p.radius * 1/(-ssoln[0][0]) # Lowest order eigenfrequency
    data[3][k] = p.a

plt.plot(data[0], data[1], marker='o', alpha=0.5, label='Wait Time Peak')
plt.plot(data[0], data[2], marker='o', alpha=0.5, label=r'$t_{00} = (-\s_{00})^{-1}$')

plt.plot(data[0], fc.clight/p.radius*(data[0])**(1./3), alpha=0.5, label=r'$t = (a\tau)^{1/3}$', ls='--', c='k')
plt.plot(data[0], fc.clight/p.radius*(data[0])**(1./2), alpha=0.5, label=r'$t = (a\tau)^{1/2}$', ls=':', c='k')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylabel('$ct/R$')
plt.xlabel(r'$a\tau$')
plt.title('Optical Depth Wait Time Scaling')
plt.show()
