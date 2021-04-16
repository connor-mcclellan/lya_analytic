# Compares the time-dependent spectrum against the steady state
# solution for each spatial mode.

from wait_time import get_Pnm
from efunctions import line_profile
from parameters import Parameters, make_sigma_grids
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import pdb

def generate_xuniform(sigma, p):
    sigma_to_x = np.cbrt(sigma/p.c1)
    xuniform_l = np.linspace(np.min(sigma_to_x), -0.1, int(len(sigma_to_x)/2))
    xuniform_r = np.linspace(0.1, np.max(sigma_to_x), int(len(sigma_to_x)/2))
    xuniform = np.concatenate([xuniform_l, xuniform_r])
    return xuniform, sigma_to_x


def fluence(sigma, p, Jsoln=None, dijkstra=False, mmax=20):
    '''
    Calculates the fluence or luminosity by spatial eigenmode.
    '''
    xuniform, sigma_to_x = generate_xuniform(sigma, p)
    sigma_xuniform = xuniform**3 * p.c1
    phi_xuniform = line_profile(sigma_xuniform, p)

    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    #spec_xuniform = np.zeros((p.nmax, np.shape(xuniform)[0]))
    phi = line_profile(sigma, p)

    if Jsoln is None:
        # STEADY STATE SOLUTION
        for n in range(1, p.nmax+1):

#            spec[n-1:] =  (
            spec[n-1] = (                   ## FACTOR OF 2???
                        -np.sqrt(6) * np.pi / 2 / 3. / p.k / p.Delta / phi
                        * p.energy / p.radius * n * (-1)**n 
                        * np.exp(-n * np.pi * p.Delta / p.k / p.radius * np.abs(sigma))
                        )
            #spec_interp = CubicSpline(sigma_to_x, spec[n-1] * phi) # Interpolate the SUM instead, not here for each mode
            #spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform

    elif dijkstra:
        H0 = (
             np.sqrt(6) * p.energy / (3. * p.k * phi) / 32 / np.pi / p.Delta
             / p.radius**3 / (np.cosh(np.pi * p.Delta / p.k / p.R * sigma) + 1)
             )
        F = 4 * np.pi * H0
        spec[0] = 4 * np.pi * p.radius**2 * F

    else:
        # TIME DEPENDENT SOLUTION
        for n in range(1, p.nmax+1):
            for m in range(mmax):
                spec[n-1] += (
                             16. * np.pi**2 * p.radius
                             / (3.0 * p.k * p.energy * phi) * (-1)**n
                             * Jsoln[n-1, m, :] / ssoln[n-1, m]
                             )
            #spec_interp = interp1d(sigma_to_x, spec[n-1] * phi)
            #spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform


    return sigma_to_x, spec
#    return sigma, spec

if __name__ == '__main__':
    energy=1.e0
    temp=1.e4
    tau0=1.e7
    radius=1.e11
    alpha_abs=0.0
    prob_dest=0.0
    xsource=0.0
    nmax=6
    nsigma=512
    nomega=10
    p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)

    sigma = np.array(sorted(np.concatenate(list(p.sigma_master.values()))))
    x_s, steady_state = fluence(sigma, p)
    x_d, dijkstra = fluence(sigma, p, dijkstra=True)


    for n in range(1, p.nmax+1):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_s, np.abs(np.sum(steady_state[:n], axis=0)), '-', marker='s', ms=1, alpha=0.7, label='steady state'.format(n))
        ax.plot(x_d, np.abs(dijkstra[0]), '-', marker='s', ms=1, alpha=0.7, label='dijkstra'.format(n))
        plt.yscale('log')
        plt.ylim(1e-16, 1e-12)
        plt.xlim(0, 30)
        plt.title('abs val of sum to n={}'.format(n))
        plt.xlabel('x')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #plt.savefig('timedep_v_steadystate_n{}.pdf'.format(n))
