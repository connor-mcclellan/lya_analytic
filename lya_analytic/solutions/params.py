import numpy as np
import astropy.constants as c
import pprint
from solutions.util import voigtx_fast
from scipy.interpolate import interp1d


class Params(object):
      
    def __init__(self, line=None, temp=None, tau0=None, num_dens=None, energy=None, sigma_source=None, n_points=None):

        # Parameters
        self.line = line
        self.temp = temp
        self.tau0 = tau0
        self.num_dens = num_dens
        self.energy = energy
        self.sigma_source = sigma_source
        self.n_points = n_points

        # Constants
        self.beta = np.sqrt(2.0 / 3.0) * np.pi / 3.0

        # Derived quantities
        self.sigma_max = 10.*self.tau0
        self.k = self.num_dens * np.pi * c.e.esu.value**2. * \
                 self.line.strength / c.m_e.cgs.value / c.c.cgs.value  # eq A2
        self.vth = np.sqrt(2.0 * c.k_B.cgs.value * self.temp / c.m_p.cgs.value)
        self.delta = self.line.nu0 * self.vth / c.c.cgs.value
        self.a = self.line.gamma / (4.0 * np.pi * self.delta)
        self.R = self.tau0 * np.sqrt(np.pi) * self.delta / self.k

        self.sigma_grid = np.concatenate([np.linspace(-self.sigma_max, self.sigma_source, int(self.n_points / 2)), 
                                          np.linspace(self.sigma_source, self.sigma_max, int(self.n_points / 2))[1:]])
        self.x_grid = self.a / self.beta * np.cbrt(self.sigma_grid)
        self.phi_grid = voigtx_fast(self.a, self.x_grid)
        self.phi = interp1d(self.sigma_grid, self.phi_grid)

        print("PARAMETER DICT")
        print('==============')
        pprint.pprint(self.__dict__)
