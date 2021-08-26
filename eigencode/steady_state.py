# Compares the time-dependent spectrum against the steady state
# solution for each spatial mode.

from wait_time import get_Pnm
from parameters import Parameters
from util import construct_sol, line_profile
from scipy.interpolate import interp1d, CubicSpline
from pathlib import Path
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})
import matplotlib.pyplot as plt
import numpy as np
import pdb


def steady_state_partial_sum(sigma, p):
    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    phi = line_profile(sigma, p)
    for n in range(1, p.nmax+1):
        spec[n-1] = (
                     - np.sqrt(6) * np.pi / 3. / p.k / phi
                    * p.energy / p.radius * n * (-1)**n 
                    * np.exp(-n * np.pi * p.Delta / p.k / p.radius * np.abs(sigma - p.sigmas))
                    )
    return np.cbrt(sigma/p.c1), spec


def dijkstra(sigma, p):
    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    phi = line_profile(sigma, p)
    H0 = (
         np.sqrt(6) * p.energy / (3. * p.k * phi) / 32 / np.pi
         / p.radius**3 / (np.cosh(np.pi * p.Delta / p.k / p.radius * (sigma - p.sigmas)) + 1)
         )
    F = 4 * np.pi * H0
    spec[0] = 4 * np.pi * p.radius**2 * F
    return np.cbrt(sigma/p.c1), spec


def time_integrated(sigma, p, Jsoln, ssoln):
    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    phi = line_profile(sigma, p)

    for n in range(1, p.nmax+1):
        for m in range(1, p.mmax+1):
            spec[n-1] += (
                         16. * np.pi**2 * p.radius * p.Delta
                         / (3.0 * p.k * p.energy * phi) * (-1)**n
                         * Jsoln[n-1, m-1, :] / ssoln[n-1, m-1]
                         )
    return np.cbrt(sigma/p.c1), spec


if __name__ == '__main__':
    directory = Path('./data/tau1e7_xinit12').resolve()
    Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax=20, mmax=500)

    x_t, tdep_spec = time_integrated(p.sigma, p, Jsoln, ssoln)
#    p.nmax = 201
#    p.mmax = 500
    x_s, steady_state = steady_state_partial_sum(p.sigma, p)
    x_d, dijkstra = dijkstra(p.sigma, p)

    
    for n in range(p.nmax-1, p.nmax):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_t, np.sum(tdep_spec[:n], axis=0), '-', c='gray', marker='s', ms=2, lw=1, alpha=0.5, label='Time-integrated')
        ax.plot(x_d, dijkstra[0], '-', c='purple', alpha=0.7, lw=3, label=r'Steady State'.format(n))
        ax.plot(x_s, np.sum(steady_state[:n], axis=0), '--', c='c', lw=3, label='Steady State (Partial Sum)')
#        plt.ylim(-0.001, 0.05)
#        plt.xlim(8, 30)
        plt.xlabel('x')
        plt.ylabel('P(x)')
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()
