import solutions.ftsoln as ftsoln
from solutions.util import voigtx, find_doppler_boundary
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pylab as pl
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})

generate_new = False

delta = 105691974558.58401
a = 0.0004717036097750442
#colors = ['#adadad', '#696969', '#000000']
colors = pl.cm.inferno(np.linspace(0.0, .75, 5))
powers = [5, 6, 7, 8, 9]

def sigma(x, a):
    return np.sqrt(2.0/3.0)*np.pi*x**3/(3.0*a)

def get_x_from_sigma(sigma, a):
    return np.cbrt(sigma*(3.0*a)*np.sqrt(3.0/2.0)/np.pi)

if __name__ == "__main__":
    colorlegend=[]
    fig, axs = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
    ### PLOT THEORY LINES

    xdop = find_doppler_boundary(3.0, a)
    print("Peak at dop. boundary when tau0={}".format(xdop**3./a))

    if generate_new:
        xdata = []
        ydata = []
        for j, tau0 in enumerate([1e5, 1e6, 1e7, 1e8, 1e9]):
            for i, sigmas in enumerate([0,0.855 * tau0, 2*0.855 * tau0]):
                x = get_x_from_sigma(sigmas, a)
                tau0, xinit, temp, radius, L = (tau0, x, 1e4, 1e11, 1.)
                x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(tau0, xinit, temp, radius, L)
                norm = 4.0 * np.pi * radius**2 * delta * 4.0 * np.pi / L
                tauscale = (a*tau0)**(1/3.)
                xdata.append(x_ft/tauscale)
                ydata.append(norm*Hh_ft*tauscale**2)
        xdata = np.array(xdata)
        np.save("xdata.npy", xdata)
        ydata = np.array(ydata)
        np.save("ydata.npy", ydata)
    else:
        xdata = np.load("xdata.npy")
        ydata = np.load("ydata.npy")

    for j, tau0 in enumerate([1e5, 1e6, 1e7, 1e8, 1e9]):
        for i, sigmas in enumerate([0,0.855 * tau0, 2*0.855 * tau0]):
            ax = axs[i]
            dind = j*3 + i
            #lw = 7.5 - (j/4) * 5.5
            lw=1.5
            ax.plot(xdata[dind], ydata[dind], lw=lw, ls='-', c=colors[j])
            if i==0:
                colorlegend.append(Patch(facecolor=colors[j], label=r'$\tau_0=10^{}$'.format(powers[j])))
            if j==0:
                ax.text(0.85, 0.1, r'$\sigma_s = {:d}\times \tau_0$'.format(i), transform=ax.transAxes)

            tauscale = (a*tau0)**(1/3.)
            ax.axvline(xdop/tauscale, color=colors[j], lw=1.6, ls="--", alpha=1)
            ax.set_ylim((-3.99, 1.99))
            ax.set_xlim((-.05, 2))
            ax.set_ylabel(r'$(a\tau_0)^{2/3}P_{bc}(x)$')
    clegend = axs[0].legend(handles=colorlegend, loc='upper left', bbox_to_anchor=[1.02, 0.8], frameon=False)
    boundarylegend = [Line2D([1], [0], color='k', lw=2, ls="--", label=r'$x_{\rm cw}$')]
    blegend = axs[0].legend(handles=boundarylegend, loc='upper left', bbox_to_anchor=[1.02, 0.025], frameon=False)
    #formatlegend = [Line2D([1], [0], color='k', lw=1, label=r'$\tau_0=10^5$'), Line2D([1], [1], lw=2, ls='-', color='k', label=r'$\tau_0=10^7$'), Line2D([1], [0], color='k', lw=3, label=r'$\tau_0=10^9$')]
    #fmtlegend = axs[0].legend(handles=formatlegend, loc='upper left', bbox_to_anchor=[1.02, 0.2], frameon=False)
    axs[0].add_artist(blegend)
    axs[0].add_artist(clegend)

    plt.subplots_adjust(top=0.97,
bottom=0.11,
left=0.11,
right=0.80,
hspace=0.0,
wspace=0.0)
    plt.xlabel(r'$(a\tau_0)^{-1/3}x$')
    plt.show()
