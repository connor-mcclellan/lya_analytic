from __future__ import print_function
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve

try:
    from util import voigtx_fast, Line, Params
    from rk import rk
except:
    from solutions.util import voigtx_fast, Line, Params
    from solutions.rk import rk

import time
import pdb
#import matplotlib.pyplot as plt


# Constants
c = 29979245800.0

class BoundaryValue(object):

    def __init__(self, n, omega, p, verbose=False):
        self.n = n
        self.omega = omega
        self.p = p
        self.verbose = verbose
        self.kappa_n = self.n * np.pi / self.p.R
        if self.verbose:
            print('\nn = {}    omega = {}    sigma_s = {}'.format(self.n, self.omega, self.p.sigma_source))
            print('process        part      domain      dxmax, dxmin      time')
            print('-----------------------------------------------------------')

    def integrate_J(self, bounds, ivs):

        start = time.time()
        dxmax, dxmin = (1e-2, 1e-6)
        sigma_eval = self.p.sigma_grid[(self.p.sigma_grid <= np.max(bounds)) & (self.p.sigma_grid >= np.min(bounds))]

        # Check to see if omega term dominates; it shouldn't out on the wing, because then there would be oscillations
        extremal_phi = self.p.phi(np.max(np.abs(np.array(bounds))))
        omega_term = 3. * self.omega * self.p.delta**2. * extremal_phi / self.p.k / c
        other_term = (self.p.delta * self.kappa_n / self.p.k)**2. 
        if omega_term >= other_term:
            raise ValueError("Omega term exceeds kappa term in wing. Aborting integration at n={}, omega={}, sigma bounds {}".format(self.n, self.omega, bounds))

        # Ensure eval points are sorted in same direction as bounds
        if bounds[0] > bounds[1]:
            sigma_eval = np.flip(sigma_eval)

        sol = rk(_mean_intensity, bounds, ivs, args=self, t_eval=sigma_eval, dx_max=dxmax, dx_min=dxmin, verbose=True)

        #sol_noeval = rk(_mean_intensity, bounds, ivs, args=self, dx_max=1e-1, dx_min=1e-4, verbose=True)
#        sol = solve_ivp(_mean_intensity, bounds, ivs, args=(self, ), t_eval=sigma_eval, rtol=rtol, atol=atol)

        #### BEGIN DEBUG CONVERGENCE TEST #####
#        sol1 = rk(_mean_intensity, bounds, ivs, args=self, t_eval=sigma_eval, dx_max=1e-2, dx_min=1e-5, verbose=True)
#
 #       start = time.time()
 #       sci = solve_ivp(_mean_intensity, bounds, ivs, t_eval=sigma_eval, args=(self, ), rtol=rtol, atol=atol)
 #       end = time.time()
 #       print('Scipy done in {:.2f} s'.format(end-start))#

        # Longer solution should be "correct"
        #long_sol = rk(_mean_intensity, bounds, ivs, args=self, t_eval=sigma_eval, dx_max=1e-4, dx_min=1e-10, verbose=True)

        #import matplotlib.pyplot as plt
        #def plotdiff(sol, label):
        #    t = np.array(sol.t, dtype=float)
        #    plt.scatter(sol.t[:-1], np.abs(sol.x[2][:-1]-long_sol.x[2][:-1])/long_sol.x[2][:-1], label=label, alpha=0.5, s=2)
        #pdb.set_trace()
        #plotdiff(sol, 'rk (1e-1, 1e-4)')
        #plotdiff(sol1, 'rk (1e-2, 1e-5)')
        #plt.scatter(sci.t, np.abs(sci.y[2]-long_sol.x[2])/long_sol.x[2], label='scipy', alpha=0.5, s=2)
#        plt.yscale('log')
        #plt.title('Fractional Difference with rk (1e-4, 1e-10)')
        #plt.legend()
        #pdb.set_trace()
        #### END DEBUG CONVERGENCE TEST #####

        end = time.time()
        if self.verbose:
            print('    [{}, {}]'.format(dxmax, dxmin), end='', flush=True)
            print('{:9.3f} s'.format(end - start))

        return sol

    def left_real(self):
        '''
        Determines four coefficients by setting J_{1, r} = 1 and J_{1, i} = 0.
        Integration is performed from a large negative sigma to the source.
        '''
        bounds = (np.min(self.p.sigma_grid), self.p.sigma_source)
        ivs = (self.p.delta * self.kappa_n / self.p.k * 1., 0., 1., 0.)
        if self.verbose:
            print('real      (-)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.a = solution.x[2][at_source]  # J_real
        self.c = solution.x[3][at_source]  # J_imag
        self.e = solution.x[0][at_source]  # d(J_real)/d(sigma)
        self.g = solution.x[1][at_source]  # d(J_imag)/d(sigma)

        return solution

    def right_real(self):
        '''
        Determines four coefficients by setting J_{2, r} = 1 and J_{2, i} = 0.
        Integration is performed from a large positive sigma to the source.
        '''
        bounds = (np.max(self.p.sigma_grid), self.p.sigma_source)
        ivs = (-self.p.delta * self.kappa_n / self.p.k * 1., 0., 1., 0.)
        if self.verbose:
            print('               real      (+)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.A = solution.x[2][at_source]  # J_real
        self.C = solution.x[3][at_source]  # J_imag
        self.E = solution.x[0][at_source]  # d(J_real)/d(sigma)
        self.G = solution.x[1][at_source]  # d(J_imag)/d(sigma)

        return solution

    def left_imag(self):
        '''
        Determines four coefficients by setting J_{1, r} = 0 and J_{1, i} = 1.
        Integration is performed from a large negative sigma to the source.
        '''
        bounds = (np.min(self.p.sigma_grid), self.p.sigma_source)
        ivs = (0., self.p.delta * self.kappa_n / self.p.k * 1., 0., 1.)
        if self.verbose:
            print('               imag      (-)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.b = solution.x[2][at_source]  # J_real
        self.d = solution.x[3][at_source]  # J_imag
        self.f = solution.x[0][at_source]  # d(J_real)/d(sigma)
        self.h = solution.x[1][at_source]  # d(J_imag)/d(sigma)

        return solution

    def right_imag(self):
        '''
        Determines four coefficients by setting J_{2, r} = 0 and J_{2, i} = 1.
        Integration is performed from a large positive sigma to the source.
        '''
        bounds = (np.max(self.p.sigma_grid), self.p.sigma_source)
        ivs = (0., -self.p.delta * self.kappa_n / self.p.k * 1., 0., 1.)
        if self.verbose:
            print('               imag      (+)    ', end='', flush=True)
        solution = self.integrate_J(bounds, ivs)
        at_source = (solution.t == self.p.sigma_source)

        self.B = solution.x[2][at_source]  # J_real
        self.D = solution.x[3][at_source]  # J_imag
        self.F = solution.x[0][at_source]  # d(J_real)/d(sigma)
        self.H = solution.x[1][at_source]  # d(J_imag)/d(sigma)

        return solution

    def solve_coeff_matrix(self):

        start = time.time()
        matrix = np.array([[self.a, self.b, -self.A, -self.B],
                           [self.c, self.d, -self.C, -self.D],
                           [-self.e, -self.f, self.E, self.F],
                           [self.g, self.h, -self.G, -self.H]])
        scales = np.mean(np.abs(matrix), axis=1)
        matrix = np.array([row/scales[i] for i, row in enumerate(matrix)], dtype=float)
        solution_vector = np.array([0.,
                                    0.,
                                    -np.sqrt(6.) / 8. * self.n**2. * self.p.energy / self.p.k / self.p.R**3./scales[2],
                                    0.], dtype=float)

        print(matrix)

        # Solve the matrix equation
        J_1_real, J_1_imag, J_2_real, J_2_imag = solve(matrix, solution_vector)
        end = time.time()
        if self.verbose:
            print('                                    {:8.3f} s'.format(end-start))
        return J_1_real, J_1_imag, J_2_real, J_2_imag

    def solve(self):
        if self.verbose:
            print('integrate      ', end='', flush=True)

        left_real = self.left_real()
        right_real = self.right_real()
        left_imag = self.left_imag()
        right_imag = self.right_imag()

        if self.verbose:
            print('solve matrix   ', end='', flush=True)
        J_1_real, J_1_imag, J_2_real, J_2_imag = self.solve_coeff_matrix()
        print("J's: ", J_1_real, J_1_imag, J_2_real, J_2_imag)
        # Centerpoint is duplicated --- remove it from one of the arrays with [:-1] before combining 
        sigma = np.concatenate((left_real.t[:-1], right_real.t))
        J_real = np.concatenate((J_1_real * left_real.x[2][:-1], J_2_real * right_real.x[2]))
        J_imag = np.concatenate((J_1_imag * left_imag.x[3][:-1], J_2_imag * right_imag.x[3]))
        J_prime_real = np.concatenate((J_1_real * left_real.x[0][:-1], J_2_real * right_real.x[0]))
        J_prime_imag = np.concatenate((J_1_imag * left_imag.x[1][:-1], J_2_imag * right_imag.x[1]))

        sort = sigma.argsort()
        sigma = sigma[sort]
        J_real = J_real[sort]
        J_imag = J_imag[sort]
        J_prime_real = J_prime_real[sort]
        J_prime_imag = J_prime_imag[sort]

        return np.array([sigma, J_real, J_imag, J_prime_real, J_prime_imag], dtype=float)


def _mean_intensity(sigma, dependent, *args):

    (x2, y2, x1, y1) = dependent
    (obj, ) = args

    # x2 = d(J_real)/d(sigma)
    # y2 = d(J_imag)/d(sigma)
    # x1 = J_real
    # y1 = J_imag    

    return np.array([(obj.p.delta * obj.kappa_n / obj.p.k)**2. * x1 + 3. * obj.omega * obj.p.delta**2. * obj.p.phi(float(sigma)) / obj.p.k / c * y1,  # dx2_dsigma
            (obj.p.delta * obj.kappa_n / obj.p.k)**2. * y1 - 3. * obj.omega * obj.p.delta**2. * obj.p.phi(float(sigma)) / obj.p.k / c * x1,  # dy2_dsigma
            x2,  # dx1_dsigma = x2
            y2  # dy1_dsigma = y2
           ])
if __name__ == '__main__':

    # Create params object
    lya = Line(1215.6701, 0.4164, 6.265e8)
    p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1e6, energy=1., 
               sigma_source=0., n_points=1e5, R=1e11)

    # Comparison of characteristic time and characteristic frequency
    tc = p.R / c * p.tau0  # Characteristic timescale
    omega_c = c / p.R * (p.a * p.tau0)**(-1. / 3.)

    print('1/tc=', 1. / tc)
    print('omega_c=', omega_c)
    print('R/c = ', p.R/c)

    # Plot some fourier coefficients
#    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    print('\nSOLUTIONS')
    print('=========')
    for n in range(7, 8):
        print('n: ', n)
        bv = BoundaryValue(n, 1.12, p)
        J = bv.solve()
#        plot = ax1.plot(J[0], J[1], '-', alpha=0.5, label='n={}'.format(n))
#        ax1.plot(J[0], J[2], '--', c=plot[-1].get_color(), alpha=0.5)
#        ax1.legend()
#        dplot = ax2.plot(J[0], J[3], '-', alpha=0.5, label='n={}'.format(n))
#        ax2.plot(J[0], J[3], '-', c=dplot[-1].get_color(), alpha=0.5)
#        ax2.legend(loc=1)

#        ax1.set_ylabel('J')
#        ax2.set_ylabel('dJ/dsigma')

#        ax1.set_xlim((-1e7, 1e7))
#        ax2.set_xlim((-1e7, 1e7))

    #plt.yscale('log')
#    plt.xlabel('Sigma')
#    plt.tight_layout()
#    plt.show()


