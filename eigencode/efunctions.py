import numpy as np
import matplotlib.pyplot as plt
from constants import fundconst, lymanalpha
from scipy.integrate import solve_ivp, odeint
from parameters import Parameters
from util import line_profile, get_sigma_bounds
import warnings
import pdb

# TODO:
# [X] Add in master sigma grid
# [X] Rewrite sol to use interpolants on master sigma grid
# [X] Speed improvements to sweep routine
# [X] Add int J dsigma as a more accurate normalization
# [ ] Change sweep step based on dispersion relation
# [ ] Insert zero at line center for Jsoln?
# [ ] rtol and atol convergence

fc = fundconst()
la = lymanalpha()




# Differential equation for solve_ivp


def func(sigma, y, n, s, p):
    J = y[0]
    dJ = y[1]
    phi = line_profile(sigma, p)
    kappan = n * np.pi / p.radius
    wavenum = kappan * p.Delta / p.k
    term1 = wavenum**2
    term2 = 3.0 * s * p.Delta**2 * phi / (fc.clight * p.k)
    dydsigma = np.zeros(3)
    dydsigma[0] = dJ
    dydsigma[1] = (term1 + term2) * J
    dydsigma[2] = J
    return dydsigma


def integrate(sigma_bounds, y_start, n, s, p):
    '''
    Returns interpolants which can be used to evaluate the function at any sigma
    '''
    sol = solve_ivp(func, [sigma_bounds[0], sigma_bounds[1]], y_start, args=(
        n, s, p), rtol=1e-10, atol=1e-10, dense_output=True)
    return sol.y.T, sol.sol


def one_s_value(n, s, p):
    '''Solve for response given n and s'''

    kappan = n * np.pi / p.radius
    wavenum = kappan * p.Delta / p.k

    # Construct grids for this value of n and s
    left, middle, right = get_sigma_bounds(n, s, p)

    # rightward integration
    J = 1.0
    dJ = wavenum * J
    y_start = np.array((J, dJ, 0.0))
    sol, interp_left = integrate(left, y_start, n, s, p)
    Jleft = sol[:, 0]
    dJleft = sol[:, 1]
    intJdsigmaleft = sol[:, 2]
    A = Jleft[-1]    # Set matrix coefficient equal to Jleft's rightmost value
    B = dJleft[-1]
    left_P = intJdsigmaleft[-1]

    # leftward integration
    J = 1.0
    dJ = -wavenum * J
    y_start = np.array((J, dJ, 0.0))
    sol, interp_right = integrate(right[::-1], y_start, n, s, p)
    Jright = sol[:, 0]
    dJright = sol[:, 1]
    intJdsigmaright = sol[:, 2]
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
        sol, interp_middle = integrate(middle, y_start, n, s, p)
        Jmiddle = sol[:, 0]
        dJmiddle = sol[:, 1]
        intJdsigmamiddle = sol[:, 2]

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
        sol, interp_middle = integrate(middle[::-1], y_start, n, s, p)
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

    J, dJ = np.zeros(p.nsigma), np.zeros(p.nsigma)
    for i in range(len(scales)):
        mask = np.logical_and(
            p.sigma <= interps[i].t_max,
            p.sigma >= interps[i].t_min)
        inds = np.where(mask)
        J[inds], dJ[inds], _ = interps[i](p.sigma[mask]) * scales[i]
    intJdsigma = left_P * scale_left - right_P * scale_right

    return J, dJ, intJdsigma


def solve(s1, s2, s3, n, p):
    '''
    Iterate to find the eigenfrequency sres and eigenvector Jres(sigma)
    given three frequencies s1, s2, s3.
    '''

    if s1 > s3:
        s1, s3 = s3, s1

    J1, dJ1, n1 = one_s_value(n, s1, p)
    J2, dJ2, n2 = one_s_value(n, s2, p)
    J3, dJ3, n3 = one_s_value(n, s3, p)

    err = 1.e20
    while err > 1.e-3:
        ratio1 = (n2 - n1) / (n2 - n3)
        ratio2 = ratio1 * (s3 - s2) / (s1 - s2)
        sguess = (s1 * ratio2 - s3) / (ratio2 - 1.0)
        Jguess, dJguess, nguess = one_s_value(n, sguess, p)

        print("\nguess:")
        print("s:    {:.6f}    {:.6f}    {:.6f}".format(
            sguess / s1, sguess / s2, sguess / s3))
        print("n:    {:.1e}    {:.1e}    {:.1e}".format(
            nguess / n1, nguess / n2, nguess / n3))

        if (sguess - s1) * (sguess - s2) < 0.0:
            s3, J3, n3 = s2, J2, n2
        else:
            s1, J1, n1 = s2, J2, n2
        s2, J2, n2 = sguess, Jguess, nguess
        err = np.abs((s3 - s1) / s2)

    sres = s2
    Jres = (J3 - J1) * (s3 - sres) * (s1 - sres) / (s1 - s3)
    nres = (n3 - n1) * (s3 - sres) * (s1 - sres) / (s1 - s3)
    return sres, Jres, nres


def sweep(p, nmin=1, output_dir=None):
    # loop over n and s=-i\omega. when you find a maximum in the size of the response, call the solve function
    # tabulate s(n,m) and J(n,m,sigma).

    gamma_const = fc.clight / p.radius / \
        (p.a * p.tau0)**0.333 * np.pi**(13.0 / 6.0) / 2.0**0.333

    for n in range(nmin, p.nmax + 1):
        print("n=", n)
        nsoln = 1
        s = -0.000001
        if nmin == 1:
            s_increment = -0.01
        else:
            s_increment = -0.25 * gamma_const * \
                n**(4.0 / 3.0) * 0.667 * (1 + 1.0 / 8.0)**(-1.0 / 3.0)

        norm = []
        while nsoln < p.mmax + 1:

            J, dJ, intJdsigma = one_s_value(n, s, p)
            norm.append(np.abs(intJdsigma))
            print("nsoln,n,s,response=", nsoln, n, s, norm[-1])
            if len(norm) > 2 and norm[-3] < norm[-2] and norm[-1] < norm[-2]:
                sres, Jres, intJdsigmares = solve(
                    s - 2 * s_increment, s - s_increment, s, n, p)
                out = {"s": sres, "J": Jres, "Jint": intJdsigmares}
                np.save(
                    output_dir /
                    'n{:03d}_m{:03d}.npy'.format(
                        n,
                        nsoln),
                    out)

                nsoln = nsoln + 1
                s_increment = -0.25 * gamma_const * \
                    n**(4.0 / 3.0) * 0.667 * (nsoln + 1.0 / 8.0)**(-1.0 / 3.0)
                print("\nds={}".format(s_increment))
            s += s_increment
    return


def check_s_eq_0(p):
    n = 1
    s = -0.00001
    kappan = n * np.pi / p.radius
    wavenum = kappan * p.Delta / p.k
    J, dJ, intJdsigma = one_s_value(n, s, p)
    plt.figure()
    plt.plot(p.sigma, J, 'b-')
    analytic = np.sqrt(6.0 / np.pi) / 16.0 * p.tau0 * n * p.energy / \
        (p.k * p.radius**3) * np.exp(-wavenum * np.abs(p.sigma))
    plt.plot(p.sigma, analytic, 'r--')
    plt.yscale('log')
    plt.show()
    plt.close()


if __name__ == "__main__":
    energy = 1.e0
    temp = 1.e4
    tau0 = 1.e7
    radius = 1.e11
    alpha_abs = 0.0
    prob_dest = 0.0
    xsource = 0.0
    nmin = 71
    nmax = 80
    mmax = 100
    nsigma = 1024

    from pathlib import Path
    from datetime import datetime
    import time
    import pickle
    datestr = datetime.today().strftime('%y%m%d-%H%M')
    output_dir = Path("./data/{}".format(datestr)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
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
    sweep(p, nmin=nmin, output_dir=output_dir)
    stop = time.time()
    with open(output_dir / 'time.txt', 'w') as f:
        f.write(str(stop - start))
