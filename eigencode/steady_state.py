# Compares the time-dependent spectrum against the steady state
# solution for each spatial mode.

from wait_time import get_Pnm
from efunctions import line_profile
from parameters import Parameters
from util import construct_sol
from scipy.interpolate import interp1d, CubicSpline
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pdb


def generate_xuniform(sigma, p):
    sigma_to_x = np.cbrt(sigma/p.c1)
    xuniform_l = np.linspace(np.min(sigma_to_x), -0.1, int(len(sigma_to_x)/2))
    xuniform_r = np.linspace(0.1, np.max(sigma_to_x), int(len(sigma_to_x)/2))
    xuniform = np.concatenate([xuniform_l, xuniform_r])
    return xuniform, sigma_to_x


def fluence(sigma, p, Jsoln=None, ssoln=None, dijkstra=False):
    '''
    Calculates the fluence or luminosity by spatial eigenmode.
    '''
    #xuniform, sigma_to_x = generate_xuniform(sigma, p)
    #sigma_xuniform = xuniform**3 * p.c1
    #phi_xuniform = line_profile(sigma_xuniform, p)

    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    #spec_xuniform = np.zeros((p.nmax, np.shape(xuniform)[0]))
    phi = line_profile(sigma, p)

    if Jsoln is None and dijkstra is False:
        # STEADY STATE SOLUTION
        for n in range(1, p.nmax+1):

            spec[n-1] = (                   ## FACTOR OF 2???
                        -np.sqrt(6) * np.pi / 3. / p.k / p.Delta / phi
                        * p.energy / p.radius * n * (-1)**n 
                        * np.exp(-n * np.pi * p.Delta / p.k / p.radius * np.abs(sigma))
                        )

    elif Jsoln is None and dijkstra is True:
        H0 = (
             np.sqrt(6) * p.energy / (3. * p.k * phi) / 32 / np.pi / p.Delta
             / p.radius**3 / (np.cosh(np.pi * p.Delta / p.k / p.radius * sigma) + 1)
             )
        F = 4 * np.pi * H0
        spec[0] = 4 * np.pi * p.radius**2 * F

    else:
        # TIME DEPENDENT SOLUTION
        for n in range(1, p.nmax+1):
            for m in range(1, p.mmax+1):
                spec[n-1] += (
                             16. * np.pi**2 * p.radius
                             / (3.0 * p.k * p.energy * phi) * (-1)**n
                             * Jsoln[n-1, m-1, :] / ssoln[n-1, m-1]
                             )

    return np.cbrt(sigma/p.c1), spec

if __name__ == '__main__':
    directory = Path('./data/210519-1213_integrator_precision').resolve()
    Jsoln, ssoln, intJsoln, p = construct_sol(directory, nmax=8, mmax=100)

    directory = Path('./data/210507_all').resolve()
    Jsoln2, ssoln2, intJsoln2, p2 = construct_sol(directory, nmax=8, mmax=100)

    x_t2, tdep_spec2 = fluence(p2.sigma, p2, Jsoln=Jsoln2, ssoln=ssoln2)
    x_t, tdep_spec = fluence(p.sigma, p, Jsoln=Jsoln, ssoln=ssoln)
    x_s, steady_state = fluence(p.sigma, p)
    x_d, dijkstra = fluence(p.sigma, p, dijkstra=True)

    for n in range(1, p.nmax):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_s, np.abs(np.sum(steady_state[:n], axis=0)), '-', marker='s', ms=1, alpha=0.7, label='steady state'.format(n))
        ax.plot(x_d, np.abs(dijkstra[0]), '-', marker='s', ms=1, alpha=0.7, label=r'dijkstra'.format(n))
        ax.plot(x_t, np.abs(np.sum(tdep_spec[:n], axis=0)), 'r-', marker='o', ms=1, alpha=0.7, label=r'High integrator precision, $n < 8$, $m < 100$'.format(n))
        ax.plot(x_t2, np.abs(np.sum(tdep_spec2[:n], axis=0)), 'm--', marker='^', ms=1, alpha=0.7, label=r'Regular, $n < 8$, $m < 100$'.format(n))
        plt.yscale('log')
        plt.ylim(1e-16, 1e-10)
        plt.xlim(0, 30)
        plt.title('abs val of sum to n={}'.format(n))
        plt.xlabel('x')
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig('timedep_v_steadystate_n{:03d}.pdf'.format(n))
        plt.close()

    '''
    import matplotlib.pylab as pl
    fig, ax = plt.subplots(1, 1)
    colors = pl.cm.jet(np.linspace(0,1,p.nmax))

    for n in range(10, p.nmax, 25):
        #y = np.sum(steady_state[:n], axis=0)
        y = np.abs(steady_state[n-1])
        if n%2==0:
            ls='--'
        else:
            ls='-'
        ax.plot(x_s, y, ls, lw=1, marker='o', ms=1, c=colors[n-1], alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=1, vmax=p.nmax))
    cbar=plt.colorbar(sm)
    cbar.ax.set_ylabel('n')
    plt.yscale('log')
#    plt.xscale('log')
    plt.xlabel('$x$')
    plt.tight_layout()
    plt.show()
    ''' 



    
