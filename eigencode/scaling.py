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

data = np.zeros((3, len(filenames)))

for i, filename in enumerate(filenames):
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
    wait_time_dist = wait_time_vs_time(ssoln,Pnmsoln,times,p)

    data[0][i] = tau0 # Optical depth
    data[1][i] = fc.clight/p.radius * times[np.argmax(wait_time_dist)]) # Peak of distribution
    data[2][i] = 1/(-ssoln[0][0])) # Lowest order eigenfrequency


plt.plot(data[0], data[1], marker='o', alpha=0.5, label='Wait Time Peak')
plt.plot(data[0], (data[1][-1]/data[2][-1])*data[2], marker='o', alpha=0.5, label=r'$t_c$, Characteristic Timescale')
scale = data[1][-1]/data[0][-1]**(1./3)

plt.plot(data[0], scale*data[0]**(1./3), alpha=0.5, label=r'$ct/R \propto \tau^{1/3}$', ls='--', c='k')
plt.plot(data[0], (data[1][-1]/data[2][-1])*(data[2][0]/data[0][0]**(1./2))*data[0]**(1./2), alpha=0.5, label=r'$ct/R \propto \tau^{1/2}$', ls=':', c='k')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylabel('$ct/R$')
plt.xlabel(r'$\tau$')
plt.title('Optical Depth Wait Time Scaling')
plt.show()
