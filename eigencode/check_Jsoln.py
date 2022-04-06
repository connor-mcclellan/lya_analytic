import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from pathlib import Path
from util import construct_sol
import numpy as np
colors = pl.cm.jet(np.linspace(0, 1, 500))
import pdb

ns=[15]
fig, axs = plt.subplots(len(ns), 1, sharex=True, figsize=(10, 8))

for i, n in enumerate(ns):
    try:
        ax = axs[i]
    except:
        ax = axs
    directory = Path('./data/tau1e6_xinit0').resolve()
    sol = construct_sol(directory, 20, 500)
    Jdata = sol[0][n-1, :]
    p = sol[3]
    voffset = 0#np.max(Jdata)
    print(voffset)

    for i, data in enumerate(Jdata): 
#        fig.patch.set_visible(False)
#        ax.axis('off')
        ax.plot(np.cbrt(sol[3].sigma/sol[3].c1), data+i*0.05*voffset, lw=0.5, alpha=1, color=colors[i]) 
#        ax.annotate('$n=${}'.format(n), (0.85, 0.9), xycoords='axes fraction')
#        ax.set_ylabel('$J_{nm}(x)$') 

plt.xlabel('$x$') 


cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=500))
cbar = fig.colorbar(sm, cax=cbar_ax) 
cbar.ax.set_ylabel('$m$', rotation=90)

#plt.subplots_adjust(top=0.964,
#bottom=0.067,
#left=0.073,
#right=0.858,
#hspace=0.164,
#wspace=0.2)
#axs[2].set_xlim((0, 60))

#plt.yscale('log') 
plt.tight_layout()
plt.show()
