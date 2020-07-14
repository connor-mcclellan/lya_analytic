import numpy as np
from glob import glob
from scipy.io import FortranFile
import astropy.constants as c
import astropy.units as u

beta = np.sqrt(2.0 / 3.0) * np.pi / 3.0


class Line(object):

    def __init__(self, lambda0, osc_strength, gamma):
        self.lambda0 = lambda0
        self.osc_strength = osc_strength
        self.gamma = gamma
        self.nu0 = (c.c.cgs / (lambda0 * u.angstrom)).to('Hz').value
        self.strength = (np.pi * c.e.esu**2. /
                         (c.m_e.cgs * c.c.cgs) * osc_strength).value


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
    return np.exp(-x**2) / np.sqrt(np.pi) + a / np.pi / (0.01 + x**2)

def tanf(x, tau):
    return np.tan(x) - x / (1. - 1.5 * tau)
