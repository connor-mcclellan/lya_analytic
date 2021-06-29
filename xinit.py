import solutions.ftsoln as ftsoln
from solutions.util import voigtx
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

delta = 105691974558.58401
a = 0.0004717036097750442

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

### PLOT THEORY LINES
for i, x in enumerate([0., 2., 4., 6., 8., 10., 12.]):
    tau0, xinit, temp, radius, L = (1e7, x, 1e4, 1e11, 1.)
    x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(tau0, xinit, temp, radius, L)
    norm = 4.0 * np.pi * radius**2 * delta * 4.0 * np.pi / L
    yscale = xinit/tau0
    ax.plot(x_ft, norm*Hh_ft/yscale, alpha=0.6, label=r'$x_s=${}'.format(xinit))
    
    
plt.ylabel('P(x)')
plt.xlabel('x')
plt.legend()
plt.show()
