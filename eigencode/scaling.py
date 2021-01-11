import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex']=True

data = np.array([
  [1e5, 1e6, 1e7, 1e8],
  [1.1, 3.2, 8.0, 16.8],
  [3.1343, 9.8665, 25.174, 54.448],
])

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
