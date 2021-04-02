# Compares the time-dependent spectrum against the steady state
# solution for each spatial mode.

from wait_time import get_Pnm
from efunctions import line_profile
from parameters import Parameters
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import pdb
nsolnmax=20


def generate_xuniform(sigma, p):
    sigma_to_x = np.cbrt(sigma/p.c1)
    xuniform_l = np.linspace(np.min(sigma_to_x), -0.1, int(len(sigma_to_x)/2))
    xuniform_r = np.linspace(0.1, np.max(sigma_to_x), int(len(sigma_to_x)/2))
    xuniform = np.concatenate([xuniform_l, xuniform_r])
    return xuniform, sigma_to_x


def fluence(sigma, p, Jsoln=None, mmax=20):
    '''
    Calculates the fluence or luminosity by spatial eigenmode.
    '''
    xuniform, sigma_to_x = generate_xuniform(sigma, p)
    sigma_xuniform = xuniform**3 * p.c1
    phi_xuniform = line_profile(sigma_xuniform, p)

    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    spec_xuniform = np.zeros((p.nmax, np.shape(xuniform)[0]))
    phi = line_profile(sigma, p)

    if Jsoln is None:
        # STEADY STATE SOLUTION
        for n in range(1, p.nmax+1):

#            spec[n-1:] =  (
            spec[n-1] = (
                        -np.sqrt(6) * np.pi / 3. / p.k / p.Delta / phi
                        * p.energy / p.radius * n * (-1)**n 
                        * np.exp(-n * np.pi * p.Delta / p.k / p.radius * np.abs(sigma))
                        )
            spec_interp = CubicSpline(sigma_to_x, spec[n-1] * phi) # Interpolate the SUM instead, not here for each mode
            spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform

    else:
        # TIME DEPENDENT SOLUTION
        for n in range(1, p.nmax+1):
            for m in range(mmax):
                spec[n-1] += (
                             16. * np.pi**2 * p.radius
                             / (3.0 * p.k * p.energy * phi) * (-1)**n
                             * Jsoln[n-1, m, :] / ssoln[n-1, m]
                             )
            spec_interp = interp1d(sigma_to_x, spec[n-1] * phi)
            spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform

    return xuniform, spec_xuniform
#    return sigma, spec

if __name__ == '__main__':
    filename = './data/eigenmode_data_xinit0_tau1e7_n6_m20.npy'
    array = np.load(filename, allow_pickle=True, fix_imports=True, )
    energy = array[0]
    temp = array[1]
    tau0 = array[2]
    radius = array[3]
    alpha_abs = array[4]
    prob_dest = array[5]
    xsource = array[6]
    nmax = array[7]
#    nmax = 1010
    nsigma = array[8]
    nomega = array[9]
    tdiff = array[10]
    sigma = array[11]
    ssoln = array[12]
    Jsoln = array[13]
    p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
#    Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)

    x_t, tdep_spec = fluence(sigma, p, Jsoln=Jsoln)
    x_s, steady_state = fluence(sigma, p)


    for n in range(1, p.nmax+1):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_t, np.abs(np.sum(tdep_spec[:n], axis=0)), 'r--', alpha=0.7, label='time dependent, mmax=20')
#        ax.plot(x_t_10, np.abs(tdep_spec_10[n-1]), 'm--', alpha=0.7, label='time dependent, mmaxx=10')
        ax.plot(x_s, np.abs(np.sum(steady_state[:n], axis=0)), '-', alpha=0.7, label='steady state n={}'.format(n))
        plt.yscale('log')
        plt.ylim(1e-21, 1e-11)
        plt.xlim(0, 40)
        plt.title('abs val of sum to n={}'.format(n))
        plt.xlabel('x')
        plt.legend()
        plt.tight_layout()
#        plt.show()
        plt.savefig('timedep_v_steadystate_n{}.pdf'.format(n))

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



    
