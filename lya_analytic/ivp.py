import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from solutions.util import voigtx_fast, Line
import astropy.constants as c

# Constants
R = 3.0
k = 1.
beta = np.sqrt(2.0 / 3.0) * np.pi / 3.0

# Parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
temp = 1e4

# Derived quantities
vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
delta = lya.nu0 * vth / c.c.cgs.value
a = lya.gamma / (4.0 * np.pi * delta)


class BoundaryValue(object):

    def __init__(self, n, omega, sigma_grid, sigma_source):
        self.n = n
        self.omega = omega
        self.sigma_grid = sigma_grid
        self.sigma_source = sigma_source
        self.kappa_n = self.n*np.pi/R
        self.sigma_source = sigma_source

    def left_real(self):
        '''
        Determines J_{-, r} and its derivative by setting J_{1, r} = 1 and 
        J_{1, i} = 0. Integration is performed from a large negative sigma to
        the source.
        '''

        bounds = (np.min(sigma_grid), self.sigma_source)
        ivs = (0., 0., 1., 0.)
        solution = integrate_J(self, bounds, ivs)
        return solution
#        self.J_left_real = J_left_real
#        self.J_left_real_prime = J_left_real_prime

    def right_real(self):

        # ...
        self.J_right_real = J_right_real
        self.J_right_real_prime = J_right_real_prime
        return

    def left_imag(self):
        # ...
        return

    def right_imag(self):
        # ...
        return

    def solve_coeff_matrix(self):

        # ...

        return J_1_real, J_1_imag, J_2_real, J_2_imag


def integrate_J(obj, bounds, ivs):
    sigma_grid = obj.sigma_grid
    atol, rtol = (3e-14, 3e-14)
    sigma_eval = sigma_grid[(sigma_grid <= bounds[1]) & (sigma_grid >= bounds[0])]
    dummy = 1
    sol = solve_ivp(mean_intensity, bounds, ivs, atol=atol, rtol=rtol, args=(obj, ))

    while len(sol.t) == 1:
        atol = atol * 10.
        rtol = rtol * 10.
        sol = solve_ivp(mean_intensity, bounds, ivs, atol=atol, rtol=rtol, args=(obj, ))

    return sol

def mean_intensity(sigma, dependent, *args):

    (x2, y2, x1, y1) = dependent
    (obj, ) = args
#    print(delta, phi(sigma), k, x1, y1)
#    print((delta*obj.kappa_n/k)**2. * x1 + 3.*obj.omega*delta**2.*phi(sigma)/k/c.c.cgs.value * y1)

    ### x2 = d(J_real)/d(sigma)
    ### y2 = d(J_imag)/d(sigma)
    ### x1 = J_real
    ### y1 = J_imag

    return [(delta*obj.kappa_n/k)**2. * x1 + 3.*obj.omega*delta**2.*phi(sigma)/k/c.c.cgs.value * y1, # dx2_dsigma
            (delta*obj.kappa_n/k)**2. * y1 - 3.*obj.omega*delta**2.*phi(sigma)/k/c.c.cgs.value * x1, # dy2_dsigma
            x2, # dx1_dsigma = x2
            y2  # dy1_dsigma = y2
           ]

# Grid values
sigma_grid = np.linspace(-100, 100, 1000000)
x_grid = a / beta * np.cbrt(sigma_grid)
phi_grid = voigtx_fast(a, x_grid)

# Line profile as a function of sigma
phi = interp1d(sigma_grid, phi_grid)

# Would we rather have x or sigma be uniform?
# if x uniform --- better spacing in solution for plots
# if sigma uniform --- more accurate interpolation near line center

bv = BoundaryValue(1., 0.5, sigma_grid, 0.)
sol = bv.left_real()
print(sol)

