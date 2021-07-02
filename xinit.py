import solutions.ftsoln as ftsoln
from solutions.util import voigtx
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.rcParams['text.usetex'] = True

delta = 105691974558.58401
a = 0.0004717036097750442
load=False
colors = ['xkcd:grey', 'xkcd:darkish green', 'xkcd:golden rod', 'xkcd:blue']

def sigma(x, a):
    return np.sqrt(2.0/3.0)*np.pi*x**3/(3.0*a)

def get_x_from_sigma(sigma, a):
    return np.cbrt(sigma*(3.0*a)*np.sqrt(3.0/2.0)/np.pi)

if __name__ == "__main__":
    colorlegend=[]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ### PLOT THEORY LINES
    for j, tau0 in enumerate([1e6, 1e7]):
        for i, sigmas in enumerate([0, tau0, 2*tau0, 3*tau0]):
            x = get_x_from_sigma(sigmas, a)
            tau0, xinit, temp, radius, L = (tau0, x, 1e4, 1e11, 1.)
            x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(tau0, xinit, temp, radius, L)
            norm = 4.0 * np.pi * radius**2 * delta * 4.0 * np.pi / L
            tauscale = (a*tau0)**(1/3.)
            ls = '-' if tau0==1e6 else '--'
            ax.plot(x_ft/tauscale, norm*Hh_ft*tauscale**2, alpha=0.6, ls=ls)
            colorlegend.append(Patch(facecolor=colors[i], label=r'${:d}\tau_0'.format(i)))
    formatlegend = [Line2D([], [], color='k', label=r'$\tau_0=10^6$'), Line2D([], [], linestyle='--', color='k', label=r'$\tau_0=10^7$')]
    clegend = plt.legend(handles=colorlegend, loc='lower right')
    plt.legend(handles=formatlegend, loc='upper right')
    plt.gca().add_artist(clegend)
    plt.ylabel(r'$H_{\rm bc}(x)$')
    plt.xlabel('x')
    plt.legend()
    plt.show()
