import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from solutions.util import voigtx_fast, Line
from solutions.params import Params
import time
import matplotlib.pyplot as plt
import astropy.constants as c


class BoundaryValue(object):

    def __init__(self, n, omega, p):
        self.n = n
        self.omega = omega
        self.p = p
        self.kappa_n = self.n * np.pi / self.p.R
        print('\nn = {}    omega = {}    sigma_s = {}'.format(self.n, self.omega, self.p.sigma_source))
        print('process        part      domain      rtol, atol        time')
        print('-----------------------------------------------------------')

    def integrate_J(self, bounds, ivs):
        start = time.time()
        atol, rtol = (3e-10, 3e-10)
        sigma_eval = self.p.sigma_grid[(self.p.sigma_grid <= np.max(bounds)) & (self.p.sigma_grid >= np.min(bounds))]

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
        print('    [{}, {}]'.format(rtol, atol), end='', flush=True)
        end = time.time()
        print('{:9.3f} s'.format(end - start))

        return sol

    def left_real(self):
        '''
        Determines four coefficients by setting J_{1, r} = 1 and J_{1, i} = 0.
        Integration is performed from a large negative sigma to the source.
        '''
        bounds = (np.min(self.p.sigma_grid), self.p.sigma_source)
        ivs = (self.p.delta * self.kappa_n / self.p.k * 1., 0., 1., 0.)
        print('real      (-)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.a = solution.y[2][at_source]  # J_real
        self.c = solution.y[3][at_source]  # J_imag
        self.e = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.g = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def right_real(self):
        '''
        Determines four coefficients by setting J_{2, r} = 1 and J_{2, i} = 0.
        Integration is performed from a large positive sigma to the source.
        '''
        bounds = (np.max(self.p.sigma_grid), self.p.sigma_source)
        ivs = (-self.p.delta * self.kappa_n / self.p.k * 1., 0., 1., 0.)
        print('               real      (+)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.A = solution.y[2][at_source]  # J_real
        self.C = solution.y[3][at_source]  # J_imag
        self.E = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.G = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def left_imag(self):
        '''
        Determines four coefficients by setting J_{1, r} = 0 and J_{1, i} = 1.
        Integration is performed from a large negative sigma to the source.
        '''
        bounds = (np.min(self.p.sigma_grid), self.p.sigma_source)
        ivs = (0., self.p.delta * self.kappa_n / self.p.k * 1., 0., 1.)
        print('               imag      (-)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.b = solution.y[2][at_source]  # J_real
        self.d = solution.y[3][at_source]  # J_imag
        self.f = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.h = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def right_imag(self):
        '''
        Determines four coefficients by setting J_{2, r} = 0 and J_{2, i} = 1.
        Integration is performed from a large positive sigma to the source.
        '''
        bounds = (np.max(self.p.sigma_grid), self.p.sigma_source)
        ivs = (0., -self.p.delta * self.kappa_n / self.p.k * 1., 0., 1.)
        print('               imag      (+)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.B = solution.y[2][at_source]  # J_real
        self.D = solution.y[3][at_source]  # J_imag
        self.F = solution.y[0][at_source]  # d(J_real)/d(sigma)
        self.H = solution.y[1][at_source]  # d(J_imag)/d(sigma)

    def solve_coeff_matrix(self):

        start = time.time()
        matrix = np.array([[self.a, self.b, -self.A, -self.B],
                           [self.c, self.d, -self.C, -self.D],
                           [-self.e, -self.f, self.A, self.B],
                           [self.g, self.h, -self.G, -self.H]])

        solution_vector = np.array([0.,
                                    0.,
                                    -np.sqrt(6.) / 8. * self.n**2. * self.p.energy / self.p.k / self.p.R**3.,
                                    0.])

        # Solve the matrix equation
        J_1_real, J_1_imag, J_2_real, J_2_imag = solve(matrix, solution_vector)
        end = time.time()
        print('                                    {:8.3f} s'.format(end-start))

        self.J1r = J_1_real
        self.J1i = J_1_imag
        self.J2r = J_2_real
        self.J2i = J_2_imag

        return J_1_real, J_1_imag, J_2_real, J_2_imag

    def J_final(self):
        # Right-going piece
        print('          (-)    ', end='', flush=True)
        lbounds = (np.min(self.p.sigma_grid), self.p.sigma_source)
        livs = (self.p.delta * self.kappa_n / self.p.k * self.J1r,
               self.p.delta * self.kappa_n / self.p.k * self.J1i,
               self.J1r,
               self.J1i)
        lsolution = self.integrate_J(lbounds, livs)

        # Left-going piece
        print('                         (+)    ', end='', flush=True)
        rbounds = (np.max(self.p.sigma_grid), self.p.sigma_source)
        rivs = (-self.p.delta * self.kappa_n / self.p.k * self.J1r,
               -self.p.delta * self.kappa_n / self.p.k * self.J1i,
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

    def solve(self):
        print('find ivs       ', end='', flush=True)
        self.left_real()
        self.right_real()
        self.left_imag()
        self.right_imag()
        print('solve matrix   ', end='', flush=True)
        self.solve_coeff_matrix()
        print('integrate      ', end='', flush = True)
        return self.J_final()


def _mean_intensity(sigma, dependent, *args):

    (x2, y2, x1, y1) = dependent
    (obj, ) = args
#    print('sigma=',sigma)

    # x2 = d(J_real)/d(sigma)
    # y2 = d(J_imag)/d(sigma)
    # x1 = J_real
    # y1 = J_imag

    return [(obj.p.delta * obj.kappa_n / obj.p.k)**2. * x1 + 3. * obj.omega * obj.p.delta**2. * obj.p.phi(sigma) / obj.p.k / c.c.cgs.value * y1,  # dx2_dsigma
            (obj.p.delta * obj.kappa_n / obj.p.k)**2. * y1 - 3. * obj.omega * obj.p.delta**2. * obj.p.phi(sigma) / obj.p.k / c.c.cgs.value * x1,  # dy2_dsigma
            x2,  # dx1_dsigma = x2
            y2  # dy1_dsigma = y2
           ]


if __name__ == '__main__':

    # Demonstrate how to use the boundary value integrator

    # Create params object
    lya = Line(1215.6701, 0.4164, 6.265e8)
    p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1e6, energy=1., 
               sigma_source=0., n_points=1e5)

    # Comparison of characteristic time and characteristic frequency
    tc = p.R / c.c.cgs.value * p.tau0  # Characteristic timescale
    omega_c = c.c.cgs.value / p.R * (p.a * p.tau0)**(-1. / 3.)

    print('1/tc=', 1. / tc)
    print('omega_c=', omega_c)

    # Plot some fourier coefficients
    plt.figure()
    print('\nSOLUTIONS')
    print('=========')
    for n in range(1, 6):
        bv = BoundaryValue(n, 0., p)
        J = bv.solve()
        plot = plt.plot(J[0], J[1], '-', alpha=(5-n)/5)
        plt.plot(J[0], J[2], '--', c=plot[-1].get_color(), alpha=(5-n)/5)

    plt.legend()
    #plt.yscale('log')
    plt.show()


