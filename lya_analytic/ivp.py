import numpy as np
from scipy.integrate import solve_ivp

# Constants
delta = 0.5
n = 1
R = 3.0
kappa_n = n*np.pi/R
k = 1.
omega = 5.
c = 2.99e10
phi = 0.2

# Initial conditions
#      x2, y2, x1, y1
ivs = (1., 2., 3., 4.)
sigma_bounds = (1e-2, 100.)

# x2 = d(J_real)/d(sigma)
# x1 = J_real
# y2 = d(J_imag)/d(sigma)
# y1 = J_imag

# Solver
def mean_intensity(sigma, dependent, *args):

    (x2, y2, x1, y1) = dependent

    return [(delta*kappa_n/k)**2. * x1 + 3.*omega*delta**2.*phi/k/c * y1, # dx2_dsigma
            (delta*kappa_n/k)**2. * y1 - 3.*omega*delta**2.*phi/k/c * x1, # dy2_dsigma
            x2, # dx1_dsigma = x2
            y2  # dy1_dsigma = y2
           ]

atol, rtol = (3e-14, 3e-14)
sol = solve_ivp(mean_intensity, sigma_bounds, ivs, atol=atol, rtol=rtol)

while len(sol.t) == 1:
    atol = atol * 10.
    rtol = rtol * 10.
    sol = solve_ivp(struct, sigma_bounds, ivs, atol=atol, rtol=rtol)

print(sol)
