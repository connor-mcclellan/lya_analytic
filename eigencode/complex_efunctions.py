import numpy as np
import matplotlib.pyplot as plt
from constants import fundconst, lymanalpha
from scipy.integrate import solve_ivp, odeint
from parameters import Parameters
from util import line_profile, get_sigma_bounds, gamma, dgamma, midpoint_diff
import warnings
import pdb
from glob import glob

fc = fundconst()
la = lymanalpha()

def func(sigma, y, n, omega, p):
    '''
    Differential equation for solve_ivp.

    Parameters
    ----------
    sigma : float
        Independent variable.
    y : array
        Array containing dependent variables J, dJ, and intJdsigma
    n : int
        Additional argument for calculating solution scale. Spatial eigenmode.
    omega : complex
        Frequency variable of the transform.
    p : `Parameters` object
        Contains physical parameters for the problem.

    Returns
    -------
    dydsigma : array
        The derivative of the dependent variables with respect to the 
        independent variable.
    '''
    J = y[0]
    dJ = y[1]
    phi = line_profile(sigma, p)
    kappan = n * np.pi / p.radius
    wavenum = kappan * p.Delta / p.k
    term1 = wavenum**2
    term2 = -3.0 * 1j * omega * p.Delta**2 * phi / (fc.clight * p.k)
    dydsigma = np.zeros(3, dtype=complex)
    dydsigma[0] = dJ
    dydsigma[1] = (term2 - term1) * J
    dydsigma[2] = J
    return dydsigma


def integrate(sigma_bounds, y_start, n, omega, p):
    '''
    Integrates the solution over a given range of sigma, given starting 
    conditions, mode number, and imaginary frequency.

    Parameters
    ----------
    sigma_bounds : tuple
        The lower and upper bounds of the integration range over sigma.
        If provided as (a, b) with a > b, the integrator will work from left
        to right. If given as (b, a) with a > b, the integrator will work from
        right to left.
    y_start : complex array
        Starting values for (J, dJ, intJdsigma) at the first value of 
        sigma_bounds
    n : int
        Spatial eigenmode.
    omega : complex
        Complex frequency.
    p : `Parameters` object
        Physical parameters of the problem.

    Returns
    -------
    sol.y.T : array
        The transposed solution for the integrator's choice of sigma points.
    sol.sol : scipy.integrate.OdeSolution
        Interpolants which can be used to evaluate the function at any 
        sigma.
    '''
    sol = solve_ivp(func, [sigma_bounds[0], sigma_bounds[1]], y_start, args=(
        n, omega, p), rtol=1e-8, atol=1e-8, dense_output=True)
    return sol.y.T, sol.sol


def one_omega_value(n, omega, p, plot=False):
    '''
    Solves for the function's response given n and omega.

    Parameters
    ----------
    n : int
        Spatial eigenmode.
    omega : complex
        Complex frequency.
    p : `Parameters` object
        Physical parameters of the problem.

    Returns
    -------
    J : array
        Solution values that have been interpolated onto the master sigma grid.
    dJ : array
        Derivative of solution at points on master sigma grid.
    intJdsigma : float
        Integral of J with respect to sigma over the entire spectrum.
    '''

    kappan = n * np.pi / p.radius
    wavenum = kappan * p.Delta / p.k

    # Construct grids for this value of n and s
    left, middle, right, sigma_vals = get_sigma_bounds(n, abs(omega), p)
    sigma_tp, sigma_efold, sigma_right = sigma_vals

    # rightward integration
    J = 1.0 + 0j
    dJ = wavenum * J
    y_start = np.array((J, dJ, 0.0 + 0j))
    soll, interp_left = integrate(left, y_start, n, omega, p)
    Jleft = soll[:, 0]
    dJleft = soll[:, 1]
    intJdsigmaleft = soll[:, 2]
    A = Jleft[-1]    # Set matrix coefficient equal to Jleft's rightmost value
    B = dJleft[-1]
    left_P = intJdsigmaleft[-1]

    # leftward integration
    J = 1.0 + 0j
    dJ = -wavenum * J
    y_start = np.array((J, dJ, 0.0 + 0j))
    solr, interp_right = integrate(right[::-1], y_start, n, omega, p)
    Jright = solr[:, 0]
    dJright = solr[:, 1]
    intJdsigmaright = solr[:, 2]
    C = Jright[-1]   # Set matrix coefficient equal to Jright's leftmost value
    D = dJright[-1]
    right_P = intJdsigmaright[-1]

    # middle integration
    # If source > 0, integrate rightward from 0 to source, matching at 0
    # Middlegrid is ordered increasingly, starting at 0 and going to source
    if p.sigmas > 0.:

        # Match initial conditions at 0 (rightward integration)
        J = Jleft[-1]
        dJ = dJleft[-1]
        y_start = np.array((J, dJ, left_P))

        # Find solution in middle region
        solm, interp_middle = integrate(middle, y_start, n, omega, p)
        Jmiddle = solm[:, 0]
        dJmiddle = solm[:, 1]
        intJdsigmamiddle = solm[:, 2]

        # Set coefficients of matrix equation at the source
        A = Jmiddle[-1]    # Overwrite previous matrix coefficients
        B = dJmiddle[-1]
        left_P = intJdsigmamiddle[-1]

        scale_right = - 1.0 / (D - B * (C / A)) * np.sqrt(6.0) / \
            8.0 * n**2 * p.energy / (p.k * p.radius**3)
        scale_left = C / A * scale_right
        scale_middle = scale_left

        scales = [scale_left, scale_middle, scale_right]
        interps = [interp_left, interp_middle, interp_right]

    # If source < 0, integrate leftward from 0 to source, matching at 0
    # Middlegrid is ordered decreasingly, starting at zero and going to source
    elif p.sigmas < 0.:

        # Match initial conditions at 0 (leftward integration)
        J = Jright[-1]
        dJ = dJright[-1]
        y_start = np.array((J, dJ, right_P))

        # Find solution in middle region
        sol, interp_middle = integrate(middle[::-1], y_start, n, omega, p)
        Jmiddle = sol[:, 0]
        dJmiddle = sol[:, 1]
        intJdsigmamiddle = sol[:, 2]

        # Set coefficients of matrix equation at the source
        C = Jmiddle[-1]   # Overwrite previous matrix coefficients
        D = dJmiddle[-1]
        right_P = intJdsigmamiddle[-1]

        scale_right = - 1.0 / (D - B * (C / A)) * np.sqrt(6.0) / \
            8.0 * n**2 * p.energy / (p.k * p.radius**3)
        scale_left = C / A * scale_right
        scale_middle = scale_right

        scales = [scale_left, scale_middle, scale_right]
        interps = [interp_left, interp_middle, interp_right]


    # If source = 0, do nothing
    else:
        scale_right = - 1.0 / (D - B * (C / A)) * np.sqrt(6.0) / \
            8.0 * n**2 * p.energy / (p.k * p.radius**3)
        scale_left = C / A * scale_right

        scales = [scale_left, scale_right]
        interps = [interp_left, interp_right]

    J, dJ = np.zeros(p.nsigma, dtype=complex), np.zeros(p.nsigma, dtype=complex)
    for i in range(len(scales)):
        mask = np.logical_and(
            p.sigma <= interps[i].t_max,
            p.sigma >= interps[i].t_min)
        inds = np.where(mask)

        J[inds], dJ[inds], _ = interps[i](p.sigma[mask]) * scales[i]
    intJdsigma = left_P * scale_left - right_P * scale_right

    if plot:
        sigmas = p.sigma
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(np.cbrt(sigmas/p.c1), J, marker='+', ms=3, alpha=0.5)
        ax1.axvline(np.cbrt(sigma_tp/p.c1), c='c', label=r'$\sigma_{\rm tp}$')
        ax1.axvline(np.cbrt(sigma_right/p.c1), c='m', label=r'$\sigma_{\rm right}$')
        ax1.axvline(p.xsource, c='limegreen', label=r'$\sigma_s$')
        ax1.legend(frameon=False)
        ax2.plot(np.cbrt(sigmas/p.c1), dJ, marker='+', ms=3, alpha=0.5)
        ax2.axvline(p.xsource, c='limegreen')
        ax2.axvline(np.cbrt(sigma_tp/p.c1), c='c')
        ax2.axvline(np.cbrt(sigma_right/p.c1), c='m')
        ax2.set_xlabel('x')
        ax2.set_ylabel('dJ(x)/dsigma')
        ax1.set_ylabel('J(x)')
        plt.suptitle('n={}, s={:.4f}'.format(n, s))
        plt.show()

    return J, dJ, intJdsigma


def omega_circ_integral(r, npoints=100):

    n = 1

    angles = np.linspace(0, 2*np.pi, npoints)
    points = r*(np.cos(angles) + 1j*np.sin(angles))
    domega = midpoint_diff(points)

    J_vals = []
    for omega in points:
        print("omega={}".format(omega), end='\r')
        J, dJ, intJdsigma = one_omega_value(n, omega, p)
        J_vals.append(J)

    J_vals = np.array(J_vals)
    integral = np.sum(np.expand_dims(domega, 1) * J_vals)
    return integral, J_vals, domega


if __name__ == "__main__":
    energy = 1.e0
    temp = 1.e4
    tau0 = 1.e7
    radius = 1.e11
    alpha_abs = 0.0
    prob_dest = 0.0
    xsource = 12.0
    nsigma = 1024

    from pathlib import Path
    from datetime import datetime
    import time
    import pickle
    import argparse

    # Create and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int)
    parser.add_argument('--mmax', type=int)
    parser.add_argument('--nmin', nargs='?', type=int, default=1)
    parser.add_argument('-p', '--path', nargs='?', type=str)
    args = parser.parse_args()

    nmin = args.nmin
    nmax = args.nmax
    mmax = args.mmax
    path = args.path

    # If no directory is provided, make one using the current date
    if path is None:
        datestr = datetime.today().strftime('%y%m%d-%H%M')
        output_dir = Path("./data/{}".format(datestr)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    p = Parameters(
        temp,
        tau0,
        radius,
        energy,
        xsource,
        alpha_abs,
        prob_dest,
        nsigma,
        nmax,
        mmax)
    pickle.dump(p, open(output_dir / 'parameters.p', 'wb'))

    radii = np.linspace(0.0000001, 0.4, 500)
    vals = []
    for r in radii:
        integral, J_circ, domega = omega_circ_integral(r)
        vals.append(integral)

    vals = np.array(vals)
   
    plt.plot(radii, vals.real, label='real')
    plt.plot(radii, vals.imag, label='imag')

    plt.title('circular integrals over J')
    plt.xlabel('|$\omega$|')
    plt.ylabel('$\sum J \Delta \omega$')

    plt.show()
