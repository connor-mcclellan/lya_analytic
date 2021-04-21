# Compares the time-dependent spectrum against the steady state
# solution for each spatial mode.

from wait_time import get_Pnm
from efunctions import line_profile
from parameters import Parameters
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


def fluence(sigma, p, Jsoln=None, dijkstra=False):
    '''
    Calculates the fluence or luminosity by spatial eigenmode.
    '''
    xuniform, sigma_to_x = generate_xuniform(sigma, p)
    sigma_xuniform = xuniform**3 * p.c1
    phi_xuniform = line_profile(sigma_xuniform, p)

    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    #spec_xuniform = np.zeros((p.nmax, np.shape(xuniform)[0]))
    phi = line_profile(sigma, p)

    if Jsoln is None and dijkstra is False:
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

    elif Jsoln is None and dijkstra is True:
        H0 = (
             np.sqrt(6) * p.energy / (3. * p.k * phi) / 32 / np.pi / p.Delta
             / p.radius**3 / (np.cosh(np.pi * p.Delta / p.k / p.R * sigma) + 1)
             )
        F = 4 * np.pi * H0
        spec[0] = 4 * np.pi * p.radius**2 * F

    else:
        # TIME DEPENDENT SOLUTION
        for n in range(1, p.nmax+1):
            for m in range(p.mmax):
                spec[n-1] += (
                             16. * np.pi**2 * p.radius
                             / (3.0 * p.k * p.energy * phi) * (-1)**n
                             * Jsoln[n-1, m, :] / ssoln[n-1, m]
                             )
            #spec_interp = interp1d(sigma_to_x, spec[n-1] * phi)
            #spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform


    return sigma_to_x, spec#xuniform, spec_xuniform
#    return sigma, spec

if __name__ == '__main__':
    filename = './data/eigenmode_data_xinit0_tau1e7_n6_m20_gammatest.npy'
    array = np.load(filename, allow_pickle=True, fix_imports=True, )
    energy = array[0]
    temp = array[1]
    tau0 = array[2]
    radius = array[3]
    alpha_abs = array[4]
    prob_dest = array[5]
    xsource = array[6]
    nmax = array[7]
    mmax = array[8]
    nsigma = array[9]
    tdiff = array[10]
    sigma = array[11]
    ssoln = array[12]
    Jsoln = array[13]
    p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax,mmax)
#    Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)


#    filename2 = './data/eigenmode_data_xinit0_tau1e7_n6_m20.npy'
    filename2 = './data/old/eigenmode_data_xinit0_tau1e7_n6_m20_rtolatol_test.npy'
    array2 = np.load(filename2, allow_pickle=True, fix_imports=True, )
    energy2 = array2[0]
    temp2 = array2[1]
    tau02 = array2[2]
    radius2 = array2[3]
    alpha_abs2 = array2[4]
    prob_dest2 = array2[5]
    xsource2 = array2[6]
    nmax2 = array[7]
    mmax2 = array[8]
    nsigma2 = array[9]
    tdiff2 = array2[10]
    sigma2 = array2[11]
    ssoln2 = array2[12]
    Jsoln2 = array2[13]
    p2 = Parameters(temp2,tau02,radius2,energy2,xsource2,alpha_abs2,prob_dest2,nsigma2,nmax2,mmax2)

    x_t2, tdep_spec2 = fluence(sigma2, p2, Jsoln=Jsoln2)
    x_t, tdep_spec = fluence(sigma, p, Jsoln=Jsoln)
    x_s, steady_state = fluence(sigma, p)
    x_d, dijkstra = fluence(sigma, p, dijkstra=True)


    for n in range(1, p.nmax):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_t, np.abs(np.sum(tdep_spec[:n], axis=0)), 'r-', marker='o', ms=1, alpha=0.7, label=r'gamma fixed, all $m < 100$'.format(n))
        ax.plot(x_t2, np.abs(np.sum(tdep_spec2[:n], axis=0)), 'm--', marker='^', ms=1, alpha=0.7, label=r'all $m < 20$'.format(n))
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



    
