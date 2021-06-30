import solutions.ftsoln as ftsoln
from solutions.util import voigtx
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

delta = 105691974558.58401
a = 0.0004717036097750442

def sigma(x, a):
    return np.sqrt(2.0/3.0)*np.pi*x**3/(3.0*a)

def get_x_from_sigma(sigma, a):
    return np.cbrt(sigma*(3.0*a)*np.sqrt(3.0/2.0)/np.pi)

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ### PLOT THEORY LINES
    for j, tau0 in enumerate([1e6, 1e7]):
        for i, sigmas in enumerate([tau0, 2*tau0, 3*tau0, 4*tau0]):
            x = get_x_from_sigma(sigmas, a)
            tau0, xinit, temp, radius, L = (tau0, x, 1e4, 1e11, 1.)
            x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(tau0, xinit, temp, radius, L)
            norm = 4.0 * np.pi * radius**2 * delta * 4.0 * np.pi / L
            tauscale = (a*tau0)**(1/3.)
            ls = '-' if tau0==1e6 else '--'
            ax.plot(x_ft/tauscale, norm*Hh_ft*tauscale**2, alpha=0.6, label=r'$x_s=${:.2f}'.format(xinit), ls=ls)
            
    plt.ylabel('P(x)')
    plt.xlabel('x')
    plt.legend()
    plt.show()
