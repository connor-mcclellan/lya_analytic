# Compares the time-dependent spectrum against the steady state
# solution for each spatial mode.

from wait_time import get_Pnm
from efunctions import parameters, line_profile
from scipy.interpolate import interp1d
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


def fluence(sigma, p, Pnmsoln=None):
    '''
    Calculates the fluence or luminosity by spatial eigenmode.
    '''
    xuniform, sigma_to_x = generate_xuniform(sigma, p)
    phi_xuniform = line_profile(xuniform**3 * p.c1, p)

    spec = np.zeros((p.nmax, np.shape(sigma)[0]))
    spec_xuniform = np.zeros((p.nmax, np.shape(xuniform)[0]))
    phi = line_profile(sigma, p)

    if Pnmsoln is None:
        for n in range(1, p.nmax+1):
            spec[n-1] = (
                        -np.sqrt(6) * np.pi / 3. / p.k / p.Delta / phi 
                        * p.energy / p.radius * n * (-1)**n 
                        * np.exp(-n * np.pi * p.Delta / p.k / p.radius * sigma)
                        )
            spec_interp = interp1d(sigma_to_x, spec[n-1] * phi)
            spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform

    else:
        for n in range(1, p.nmax+1):
            for m in range(nsolnmax):
                spec[n-1] += (
                             16. * np.pi**2 * p.radius * p.Delta 
                             / (3.0 * p.k * p.energy) * (-1)**n 
                             * Pnmsoln[n-1, m, :] / ssoln[n-1, m] / phi
                             )
                spec_interp = interp1d(sigma_to_x, spec[n-1] * phi)
                spec_xuniform[n-1] = spec_interp(xuniform) / phi_xuniform

    return xuniform, spec_xuniform


if __name__ == '__main__':
    filename = './data/eigenmode_data_xinit0.0_tau1e7_nmax6_nsolnmax20.npy'
    array = np.load(filename, allow_pickle=True, fix_imports=True, )
    energy = array[0]
    temp = array[1]
    tau0 = array[2]
    radius = array[3]
    alpha_abs = array[4]
    prob_dest = array[5]
    xsource = array[6]
    nmax = array[7]
    nsigma = array[8]
    nomega = array[9]
    tdiff = array[10]
    sigma = array[11]
    ssoln = array[12]
    Jsoln = array[13]
    p = parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
    Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)

    x_t, tdep_spec = fluence(sigma, p, Pnmsoln=Pnmsoln)
    x_s, steady_state = fluence(sigma, p)

    for n in range(1, p.nmax+1):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_t, np.abs(tdep_spec[n-1]), 'r--', alpha=0.7, label='time dependent')
        ax.plot(x_s, np.abs(steady_state[n-1]), 'b--', alpha=0.7, label='steady state')
        plt.yscale('log')
        plt.title('n={}'.format(n))
        plt.tight_layout()
        plt.show()
        



    
