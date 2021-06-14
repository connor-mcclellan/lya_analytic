import numpy as np
from glob import glob
from scipy.io import FortranFile
from scipy.special import wofz
from scipy.interpolate import interp1d
import pdb

# Constants (CGS)
c = 29979245800.0
esu = 4.803204712570263e-10
m_e = 9.1093837015e-28
k_B = 1.380649e-16
m_p = 1.67262192369e-24

def midpoint_diff(t):
  midpoints = 0.5*(t[1:]+t[:-1])
  dt = np.diff(midpoints)
  dt = np.insert(dt, 0, midpoints[1] - midpoints[0])
  dt = np.append(dt, midpoints[-1] - midpoints[-2])
  return dt

def read_bin(path):
    filenames = sorted(glob(path + '*.bin'))
    for filename in filenames:
        f = FortranFile(filename)

        # Get each row from the FortranFile object
        ntrials = f.read_ints()
        new_mu = f.read_reals(np.float32)
        new_x = f.read_reals(np.float32)
        new_time = f.read_reals(np.float32)
        f.close()

        # Add new data to array if the arrays already exists, or create them

        try:
            mu = np.append(mu, new_mu[new_mu > 0.])
            x = np.append(x, new_x[new_mu > 0.])
            time = np.append(time, new_time[new_mu > 0.])
        except BaseException:
            mu = new_mu[new_mu > 0.]
            x = new_x[new_mu > 0.]
            time = new_time[new_mu > 0.]

    return mu, x, time


def voigtx_fast(a, x):
    return a / np.pi / (0.01 + x**2)

  #np.exp(-x**2) / np.sqrt(np.pi) + a / np.pi / (0.01 + x**2)

def voigtx(a, x):
    z = x + a*1j
    H = np.real(wofz(z))
    line_profile = H/np.sqrt(np.pi)
    return line_profile

def tanf(x, tau):
    return np.tan(x) - x / (1. - 1.5 * tau)

def scinot(num):
    ''' 
    Formats numbers in scientific notation for LaTeX.
    '''
    numstrs = '{:.1E}'.format(num).split('E+')
    return r'${} \times 10^{{{}}}$'.format(numstrs[0], int(numstrs[1]))


def j0(kappa, r):
    return np.sin(kappa*r)/(kappa*r)
