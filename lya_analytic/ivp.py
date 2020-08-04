import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import solve
from solutions.util import voigtx_fast, Line
import astropy.constants as c
import pdb

# Constants
k = 1.
beta = np.sqrt(2.0 / 3.0) * np.pi / 3.0

# Physical Parameters
lya = Line(1215.6701, 0.4164, 6.265e8)
temp = 1e4
tau0 = 1e7
num_dens = 1e6  # essentially sets a timescale, which determines range of omega
energy = 1.

# Grid parameters
sigma_max = 100.
sigma_source = 0.
npoints = 100000

# Derived quantities
k = num_dens * np.pi * c.e.esu.value**2. * \
    lya.strength / c.m_e.cgs.value / c.c.cgs.value  # eq A2
vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
delta = lya.nu0 * vth / c.c.cgs.value
a_const = lya.gamma / (4.0 * np.pi * delta)
R = tau0 * np.sqrt(np.pi) * delta / k

tc = R / c.c.cgs.value * tau0  # Characteristic timescale
omega_c = c.c.cgs.value / R * (a_const * tau0)**(-1. / 3.)
# print(a_const**(-1./3)*(tau0)**(2./3))


class BoundaryValue(object):

    def __init__(self, n, omega, sigma_grid, sigma_source):
        self.n = n
        self.omega = omega
        self.sigma_grid = sigma_grid
        self.sigma_source = sigma_source
        self.kappa_n = self.n * np.pi / R
        self.sigma_source = sigma_source

    def integrate_J(self, bounds, ivs):
        atol, rtol = (3e-14, 3e-14)
        sigma_eval = self.sigma_grid[(self.sigma_grid <= np.max(
            bounds)) & (self.sigma_grid >= np.min(bounds))]

        # Ensure eval points are sorted in same direction as bounds
        if bounds[0] > bounds[1]:
            sigma_eval = np.flip(sigma_eval)

        sol = solve_ivp(_mean_intensity, bounds, ivs, atol=atol, rtol=rtol,
                        args=(self, ), t_eval=sigma_eval)

        while len(sol.t) == 1:
            atol = atol * 10.
            rtol = rtol * 10.
            sol = solve_ivp(_mean_intensity, bounds, ivs, atol=atol, rtol=rtol,
                            args=(self, ), t_eval=sigma_eval)
        return sol

    def left_real(self):
        '''
        Determines four coefficients by setting J_{1, r} = 1 and J_{1, i} = 0.
        Integration is performed from a large negative sigma to the source.
        '''
        bounds = (np.min(self.sigma_grid), self.sigma_source)
        ivs = (delta * self.kappa_n / k * 1., 0., 1., 0.)
        print('Integrating real part from large negative sigma...')
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.sigma_source)

        self.a = solution.y[2][at_source]  # J_real
        self.c = solution.y[3][at_source]  # J_imag
        self.e = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.g = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def right_real(self):
        '''
        Determines four coefficients by setting J_{2, r} = 1 and J_{2, i} = 0.
        Integration is performed from a large positive sigma to the source.
        '''
        bounds = (np.max(self.sigma_grid), self.sigma_source)
        ivs = (-delta * self.kappa_n / k * 1., 0., 1., 0.)
        print('Integrating real part from large positive sigma...')
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.sigma_source)

        self.A = solution.y[2][at_source]  # J_real
        self.C = solution.y[3][at_source]  # J_imag
        self.E = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.G = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def left_imag(self):
        '''
        Determines four coefficients by setting J_{1, r} = 0 and J_{1, i} = 1.
        Integration is performed from a large negative sigma to the source.
        '''
        bounds = (np.min(self.sigma_grid), self.sigma_source)
        ivs = (0., delta * self.kappa_n / k * 1., 0., 1.)
        print('Integrating imaginary part from large negative sigma...')
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.sigma_source)

        self.b = solution.y[2][at_source]  # J_real
        self.d = solution.y[3][at_source]  # J_imag
        self.f = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.h = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def right_imag(self):
        '''
        Determines four coefficients by setting J_{2, r} = 0 and J_{2, i} = 1.
        Integration is performed from a large positive sigma to the source.
        '''
        bounds = (np.max(self.sigma_grid), self.sigma_source)
        ivs = (0., -delta * self.kappa_n / k * 1., 0., 1.)
        print('Integrating imaginary part from large positive sigma...')
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.sigma_source)

        self.B = solution.y[2][at_source]  # J_real
        self.D = solution.y[3][at_source]  # J_imag
        self.F = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.H = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def solve_coeff_matrix(self):

        matrix = np.array([[self.a, self.b, -self.A, -self.B],
                           [self.c, self.d, -self.C, -self.D],
                           [-self.e, -self.f, self.A, self.B],
                           [self.g, self.h, -self.G, -self.H]])

        solution_vector = np.array([0.,
                                    0.,
                                    -np.sqrt(6.) / 8. * self.n**2. *
                                             energy / k / R**3.,
                                    0.])

        # Solve the matrix equation
        J_1_real, J_1_imag, J_2_real, J_2_imag = solve(matrix, solution_vector)

        self.J1r = J_1_real
        self.J1i = J_1_imag
        self.J2r = J_2_real
        self.J2i = J_2_imag

        return J_1_real, J_1_imag, J_2_real, J_2_imag

    def J_final(self):
        # Right-going piece
        lbounds = (np.min(self.sigma_grid), self.sigma_source)
        livs = (delta * self.kappa_n / k * self.J1r,
               delta * self.kappa_n / k * self.J1i,
               self.J1r,
               self.J1i)
        lsolution = self.integrate_J(lbounds, livs)

        # Left-going piece
        rbounds = (np.max(self.sigma_grid), self.sigma_source)
        rivs = (-delta * self.kappa_n / k * self.J1r,
               -delta * self.kappa_n / k * self.J1i,
               self.J1r,
               self.J1i)
        rsolution = self.integrate_J(rbounds, rivs)

        sigma = np.ndarray.flatten(np.array((lsolution.t, rsolution.t)))
        J_real = np.ndarray.flatten(np.array((lsolution.y[2], rsolution.y[2])))
        J_imag = np.ndarray.flatten(np.array((lsolution.y[3], rsolution.y[3])))

        sort = sigma.argsort()
        sigma = sigma[sort]
        J_real = J_real[sort]
        J_imag = J_imag[sort]

        return np.array([sigma, J_real, J_imag])


def _mean_intensity(sigma, dependent, *args):

    (x2, y2, x1, y1) = dependent
    (obj, ) = args
#    print('sigma=',sigma)

    # x2 = d(J_real)/d(sigma)
    # y2 = d(J_imag)/d(sigma)
    # x1 = J_real
    # y1 = J_imag

    return [(delta * obj.kappa_n / k)**2. * x1 + 3. * obj.omega * delta**2. * phi(sigma) / k / c.c.cgs.value * y1,  # dx2_dsigma
            (delta * obj.kappa_n / k)**2. * y1 - 3. * obj.omega * \
             delta**2. * phi(sigma) / k / c.c.cgs.value * x1,  # dy2_dsigma
            x2,  # dx1_dsigma = x2
            y2  # dy1_dsigma = y2
           ]


# Build the sigma grid, x grid, and line profile grid
sigma_grid = np.concatenate([np.linspace(-sigma_max, sigma_source, int(npoints / 2)), 
                             np.linspace(sigma_source, sigma_max, int(npoints / 2)[1:]])
x_grid=a_const / beta * np.cbrt(sigma_grid)
phi_grid=voigtx_fast(a_const, x_grid)

print('1/tc=', 1. / tc)
print('omega_c=', omega_c)

# Line profile as a function of sigma
phi=interp1d(sigma_grid, phi_grid)

# BoundaryValue Object and method calls --- will be cleaned up eventually
bv=BoundaryValue(1., 1 / tc, sigma_grid, 0.)
bv.left_real()
bv.right_real()
bv.left_imag()
bv.right_imag()
J1r, J1i, J2r, J2i=bv.solve_coeff_matrix()
J=bv.J_final()
pdb.set_trace()
