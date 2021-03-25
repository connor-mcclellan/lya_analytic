import matplotlib.pyplot as plt
import numpy as np
from mc_visual import mc_wait_time
import matplotlib
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from wait_time import *


filenames = [
  './data/eigenmode_data_xinit0.0_tau1e5.npy',
  './data/eigenmode_data_xinit0.0_tau1e6.npy',
  './data/eigenmode_data_xinit0.0_tau1e7.npy',
  './data/eigenmode_data_xinit0.0_tau1e8.npy',
]

data = np.zeros((4, len(filenames)))
mcpoints = []

for k, filename in enumerate(filenames):
    array = np.load(filename, allow_pickle=True, fix_imports=True)
    energy, temp, tau0, radius, alpha_abs, prob_dest, xsource, nmax, nsigma, nomega, tdiff, sigma, ssoln, Jsoln = array

    taustr = filename.split('_tau')[-1].split('.npy')[0]
    if taustr in ['1e7', '1e6']:
        mcdir = '/home/connor/Documents/lya_analytic/data/1M tau0_'+str(float(taustr))+'_xinit_0.0_temp_10000.0_probabs_0.0/'
        tdata, _, _ = mc_wait_time(mcdir)
        t, n = tdata
        tpeak = t[np.argmax(n)]
        taufloat = float(taustr)
        mcpoints.append((taufloat, tpeak))

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
          if any(np.isnan(Pnmsoln[n-1,m,:])):
            continue
          else:
            P[i]=P[i] - np.sum(Pnmsoln[n-1,m,:])*ssoln[n-1,m]*np.exp(ssoln[n-1,m]*t)


    data[0][k] = p.a*tau0 # Optical depth
    data[1][k] = fc.clight/p.radius * times[np.argmax(P)] # Peak of distribution
    data[2][k] = fc.clight/p.radius * 1/(-ssoln[0][0]) # Lowest order eigenfrequency
    data[3][k] = p.a

norm = data[2][-1]/(fc.clight/p.radius*(data[0][-1])**(1./3))
plt.plot(data[0]/data[3], norm*fc.clight/p.radius*(data[0])**(1./3), alpha=0.5, label=r'$t \propto (a\tau_0)^{1/3}$', ls=':', c='k')

plt.plot(data[0]/data[3], data[1], '-', marker='^', ms=3, alpha=0.5, label='Wait Time Peak')
plt.plot(data[0]/data[3], data[2], '-', marker='o', ms=3, alpha=0.5, label=r'$t_{10} = -(s_{10})^{-1}$')

mcpoints = np.array(mcpoints)
plt.scatter(mcpoints[:, 0], mcpoints[:, 1], label='Monte Carlo', marker='+')

#plt.plot(data[0], fc.clight/p.radius*(data[0])**(1./2), alpha=0.5, label=r'$t = (a\tau)^{1/2}$', ls=':', c='k')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylabel('$ct/R$')
plt.xlabel(r'$\tau$')
plt.title('Characteristic Wait Time Scaling with Optical Depth')
plt.show()
