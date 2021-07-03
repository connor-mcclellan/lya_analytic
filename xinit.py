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
colors = ['#afafaf', '#696969', '#000000']

def sigma(x, a):
    return np.sqrt(2.0/3.0)*np.pi*x**3/(3.0*a)

def get_x_from_sigma(sigma, a):
    return np.cbrt(sigma*(3.0*a)*np.sqrt(3.0/2.0)/np.pi)

if __name__ == "__main__":
    colorlegend=[]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ### PLOT THEORY LINES
    for j, tau0 in enumerate([1e5, 1e7, 1e9]):
        for i, sigmas in enumerate([0, tau0, 2*tau0]):
            x = get_x_from_sigma(sigmas, a)
            tau0, xinit, temp, radius, L = (tau0, x, 1e4, 1e11, 1.)
            x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(tau0, xinit, temp, radius, L)
            norm = 4.0 * np.pi * radius**2 * delta * 4.0 * np.pi / L
            tauscale = (a*tau0)**(1/3.)
            if tau0==1e5:
                lw = 1.5 
            elif tau0==1e7:
                lw = 2.25
            else:
                lw = 3
            ax.plot(x_ft/tauscale, norm*Hh_ft*tauscale**2, alpha=1, lw=lw, c=colors[i], zorder=100-30*i)
            if j==0:
#                ax.axvline(x/tauscale, color=colors[i], lw=1)
                colorlegend.append(Patch(facecolor=colors[i], label=r'$\sigma_s = {:d}\times \tau_0$'.format(i)))
    formatlegend = [Line2D([1], [0], color='k', lw=1.5, label=r'$\tau_0=10^5$'), Line2D([1], [1], lw=2.25, color='k', label=r'$\tau_0=10^7$'), Line2D([1], [0], color='k', lw=3, label=r'$\tau_0=10^9$')]
    fmtlegend = ax.legend(handles=formatlegend, loc='upper left')
    clegend = ax.legend(handles=colorlegend, loc='lower left')
    plt.gca().add_artist(clegend)
    plt.gca().add_artist(fmtlegend)
    plt.ylabel(r'$(a\tau_0)^{2/3}P(x)$')
    plt.xlabel(r'$(a\tau_0)^{-1/3}x$')
    plt.show()
