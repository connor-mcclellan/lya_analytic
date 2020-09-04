import numpy as np
from glob import glob
from scipy.io import FortranFile
from scipy.interpolate import interp1d

# Constants (CGS)
c = 29979245800.0
esu = 4.803204712570263e-10
m_e = 9.1093837015e-28
k_B = 1.380649e-16
m_p = 1.67262192369e-24

class Line(object):

    def __init__(self, lambda0, osc_strength, gamma):
        self.lambda0 = lambda0
        self.osc_strength = osc_strength
        self.gamma = gamma
        self.nu0 = c / (lambda0 * 1e-8)
        self.strength = (np.pi * esu**2. / (m_e * c) * osc_strength)

class Params(object):
      
    def __init__(self, line=None, temp=None, tau0=None, num_dens=None, energy=None, R=None, sigma_source=None, n_points=None, log=False):

        # Parameters
        self.line = line
        self.temp = temp
        self.tau0 = tau0
        self.num_dens = num_dens
        self.energy = energy
        self.R = R
        self.sigma_source = sigma_source
        self.n_points = n_points

        # Constants
        self.beta = np.sqrt(2.0 / 3.0) * np.pi / 3.0

        # Derived quantities
        self.sigma_max = 10.*self.tau0
        self.k = self.num_dens * np.pi * esu**2. * \
                 self.line.osc_strength / m_e / c  # eq A2
        self.vth = np.sqrt(2.0 * k_B * self.temp / m_p)
        self.delta = self.line.nu0 * self.vth / c
        self.a = self.line.gamma / (4.0 * np.pi * self.delta)
#        self.R = self.tau0 * np.sqrt(np.pi) * self.delta / self.k

        if log:
            # Assumes source sigma is 0 and domains are symmetric
            sourcelog = np.log10(self.sigma_source + 1.)
            limitlog = np.log10(self.sigma_max + 1.)
            left = -np.logspace(sourcelog, limitlog, int(self.n_points/2)) + 1.
            right = (np.logspace(sourcelog, limitlog, int(self.n_points/2)+1) - 1.)[1:]
            self.sigma_grid = np.concatenate([np.flip(left), right])

        else:
            self.sigma_grid = np.concatenate([np.linspace(-self.sigma_max, self.sigma_source, int(self.n_points / 2)), 
                                              np.linspace(self.sigma_source, self.sigma_max, int(self.n_points / 2)+1)[1:]])
        self.x_grid = np.cbrt(self.a / self.beta * self.sigma_grid)
        self.phi_grid = voigtx_fast(self.a, self.x_grid)/(np.sqrt(np.pi)*self.delta)
        self.phi = interp1d(self.sigma_grid, self.phi_grid)

        print("PARAMETER DICT")
        print('==============')
        self.print_block(self)

    def print_block(self, obj, indent=0):

        if indent == 0:
            print('{')
            indent += 1

        for key, value in obj.__dict__.items():
            if key[0] == '_':
                continue
            elif hasattr(value, '__dict__'):
                print('    '*indent + "'{}': {{".format(key))
                self.print_block(value, indent=indent+1)
            else:
                if type(value) == np.ndarray:
                    value = '[{} ... {}]'.format(value[0], value[-1])
                print('    '*indent+"'{}': {},".format(key, value))

        if indent == 1:
            print('    }')
        else:
            print('    '*(indent)+'},')


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
    return np.exp(-x**2) / np.sqrt(np.pi) #+ a / np.pi / (0.01 + x**2)

def tanf(x, tau):
    return np.tan(x) - x / (1. - 1.5 * tau)

def j0(kappa, r):
    return np.sin(kappa*r)/(kappa*r)
