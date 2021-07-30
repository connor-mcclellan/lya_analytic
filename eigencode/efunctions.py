import numpy as np
import matplotlib.pyplot as plt
from constants import fundconst, lymanalpha
from scipy.integrate import solve_ivp, odeint
from parameters import Parameters
from util import line_profile, get_sigma_bounds, gamma, dgamma
import warnings
import pdb

# TODO:
# [X] Add in master sigma grid
# [X] Rewrite sol to use interpolants on master sigma grid
# [X] Speed improvements to sweep routine
# [X] Add int J dsigma as a more accurate normalization
# [X] Change sweep step based on dispersion relation
# [ ] Insert zero at line center for Jsoln?

fc = fundconst()
la = lymanalpha()

def func(sigma, y, n, s, p):
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
    s : float
        Sweep variable; imaginary frequency. Negative for damped, real 
        solutions.
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
    term2 = 3.0 * s * p.Delta**2 * phi / (fc.clight * p.k)
    dydsigma = np.zeros(3)
    dydsigma[0] = dJ
    dydsigma[1] = (term1 + term2) * J
    dydsigma[2] = J
    return dydsigma


def integrate(sigma_bounds, y_start, n, s, p):
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
    y_start : array
        Starting values for (J, dJ, intJdsigma) at the first value of 
        sigma_bounds.
    n : int
        Spatial eigenmode.
    s : float
        Imaginary frequency.
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
        n, s, p), rtol=1e-10, atol=1e-10, dense_output=True)
    return sol.y.T, sol.sol


def one_s_value(n, s, p, plot=False):
    '''
    Solves for the function's response given n and s.

    Parameters
    ----------
    n : int
        Spatial eigenmode.
    s : float
        Imaginary frequency.
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
    left, middle, right = get_sigma_bounds(n, s, p)

    # rightward integration
    J = 1.0
    dJ = wavenum * J
    y_start = np.array((J, dJ, 0.0))
    soll, interp_left = integrate(left, y_start, n, s, p)
    Jleft = soll[:, 0]
    dJleft = soll[:, 1]
    intJdsigmaleft = soll[:, 2]
    A = Jleft[-1]    # Set matrix coefficient equal to Jleft's rightmost value
    B = dJleft[-1]
    left_P = intJdsigmaleft[-1]

    # leftward integration
    J = 1.0
    dJ = -wavenum * J
    y_start = np.array((J, dJ, 0.0))
    solr, interp_right = integrate(right[::-1], y_start, n, s, p)
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
        solm, interp_middle = integrate(middle, y_start, n, s, p)
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

    if plot:
        sigmas = p.sigma
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(np.cbrt(sigmas/p.c1), dJ, marker='+', ms=3, alpha=0.5)
        ax2.plot(np.cbrt(sigmas/p.c1), np.abs(dJ), marker='+', ms=3, alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_yscale('log')
        ax1.set_ylabel('dJ(x)/dsigma')
        ax2.set_ylabel('|dJ(x)/dsigma|')
        plt.suptitle('n={}, s={:.4f}'.format(n, s))
#        plt.savefig('Jres_n={}_s={:08.3f}.pdf'.format(n, s))
        plt.show()
#        plt.close()

    return J, dJ, intJdsigma


def solve(s1, s2, s3, n, p):
    '''
    Iterate to find the eigenfrequency sres and eigenvector Jres(sigma)
    given three frequencies s1, s2, s3.

    Parameters
    ----------
    s1 : float
        Imaginary frequency at the leftmost point.
    s2 : float
        Imaginary frequency at the middle point.
    s3 : float
        Imaginary frequency at the rightmost point.
    n : int
        Spatial eigenmode.
    p : `Parameters` object
        Physical parameters of the problem.

    Returns
    -------
    sres : float
        Eigenfrequency of the resonance which has been solved for.
    Jres : array
        Eigenfunction at the resonance which has been solved for.
    nres : float
        Integrated response with respect to sigma.
    '''

    if s1 > s3:
        s1, s3 = s3, s1

    J1, dJ1, n1 = one_s_value(n, s1, p)
    J2, dJ2, n2 = one_s_value(n, s2, p)
    J3, dJ3, n3 = one_s_value(n, s3, p)

    n_iter = 0
    err = 1e20
    while n_iter < 10 and err > 1e-6:
        ratio1 = (n2 - n1) / (n2 - n3)
        ratio2 = ratio1 * (s3 - s2) / (s1 - s2)
        sguess = (s1 * ratio2 - s3) / (ratio2 - 1.0)
        Jguess, dJguess, nguess = one_s_value(n, sguess, p)

#        print("\nguess:")
#        print("s:    {:.6f}    {:.6f}    {:.6f}".format(
#            sguess / s1, sguess / s2, sguess / s3))
#        print("n:    {:.1e}    {:.1e}    {:.1e}".format(
#            nguess / n1, nguess / n2, nguess / n3))

        err = np.abs(s2 - sguess)
        if (sguess - s1) * (sguess - s2) < 0.0:
            s3, J3, n3 = s2, J2, n2
        else:
            s1, J1, n1 = s2, J2, n2
        s2, J2, n2 = sguess, Jguess, nguess
        n_iter += 1

    print("\nres: {:.7f}    err: {:.7f}".format(s2, err))
    sres = s2
    Jres = (J3 - J1) * (s3 - sres) * (s1 - sres) / (s1 - s3)
    nres = (n3 - n1) * (s3 - sres) * (s1 - sres) / (s1 - s3)

    one_s_value(n, sres, p)#, plot=True)
#    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#    ax1.plot(np.cbrt(p.sigma/p.c1), Jres, lw=0.5, alpha=0.75)
#    ax2.plot(np.cbrt(p.sigma/p.c1), Jres, lw=0.5, alpha=0.75)

    return sres, Jres, nres

def sweep(p, nmin=1, output_dir=None):
    '''
    Sweeps over n and s=-i\omega to find maxima in the size of the response.

    Parameters
    ----------
    p : `Parameters` object
        Physical parameters of the problem.
    nmin : int, optional  
        Starting value of n, built in for parallelization purposes. Default
        value is 1.
    output_dir : `pathlib.Path` object
        Base directory within which outputs will be stored. Default is None 
        (current working directory). If outputs already exist in the output
        directory, the sweep will pick up from the last eigenvalue found.

    Returns
    -------
    None. sres, Jres, and intJdsigmares output is saved to file.
    '''

    middle_sweep_res = 0.05
    early_sweep_res = 0.0001
    n_sweep_buffers = 15

    for n in range(nmin, p.nmax + 1):
        print("n=", n)

        # Set starting s based on what eigenmode solution number we're starting
        # on. If no previous solution has been calculated, start close to 0.
        try:
            data_fname = sorted(glob(str(output_dir/'n{:03d}_*.npy'.format(n))))[-1]
            data = np.load(data_fname, allow_pickle=True).item()
            nsoln = int(data_fname.split('.npy')[0].split('_m')[-1]) + 1
            s = data['s'] - middle_sweep_res * dgamma(n, nsoln, p)
        except:
            nsoln = 1
            s = - gamma(n, nsoln, p) + 10 * n_sweep_buffers * early_sweep_res * dgamma(n, nsoln, p) if n!=1 else -0.00000001
            s = min(s, -0.00000001)

        # Set starting sweep increment in s based on the dispersion relation.
        if n == 1 and nsoln == 1:
            s_increment = - early_sweep_res
            sweep_resolution = early_sweep_res
        elif nsoln < 10:
            s_increment = - early_sweep_res * dgamma(n, nsoln, p)
            sweep_resolution = early_sweep_res
        else:
            s_increment = - middle_sweep_res * dgamma(n, nsoln, p)
            sweep_resolution = middle_sweep_res
        print("\nNEXT RESONANCE: ", - gamma(n, nsoln, p))
        print("INCREMENT: ", s_increment)

        # Sweep, check resonance, save outputs
        norm = []
        nsweeps = 0
        print("  m  n  s         f(s)      sweep#")

        while nsoln < p.mmax + 1:
            nsweeps += 1
            J, dJ, intJdsigma = one_s_value(n, s, p)
            norm.append(np.abs(intJdsigma))
            print("{} {} {:.6f} {:.3e} {}".format(str(nsoln).rjust(3), str(n).rjust(3), s, norm[-1], nsweeps), end="\r")
            if len(norm) > 2 and norm[-3] < norm[-2] and norm[-1] < norm[-2]:
                sres, Jres, intJdsigmares = solve(s - 2 * s_increment, s - s_increment, s, n, p)
                out = {"s": sres, "J": Jres, "Jint": intJdsigmares}
                np.save(output_dir/'n{:03d}_m{:03d}.npy'.format(n, nsoln), out)
                #pdb.set_trace()
                nsoln = nsoln + 1

                # Set starting sweep increment in s based on the dispersion relation.
                if n == 1 and nsoln == 1:
                    s_increment = - early_sweep_res
                    sweep_resolution = early_sweep_res
                elif nsoln < 10:
                    s_increment = - early_sweep_res * dgamma(n, nsoln, p)
                    sweep_resolution = early_sweep_res
                else:
                    s_increment = - middle_sweep_res * dgamma(n, nsoln, p)
                    sweep_resolution = middle_sweep_res

                s_increment = - sweep_resolution * dgamma(n, nsoln, p)
                nsweeps = 0
                print("\nNEXT RESONANCE: ", gamma(n, nsoln, p))
                print("INCREMENT: ", s_increment)
                print("  m  n  s        f(s)      sweep#")
                s = - gamma(n, nsoln, p) - n_sweep_buffers * s_increment
            s += s_increment
    return

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
    from glob import glob
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

    ###
    # Command line call should look like this:
        # python efunctions.py --nmax 20 --mmax 500 -p ./data/sample_run
    # Then, to continue up to m=1000 for all the same n, you would do this:
        # python efunctions.py --nmax 20 --mmax 1000 -p ./data/sample_run
    # The code will start at n=1, m=501 and sweep through the remaining modes.
    ###

    # If no directory is provided, make one using the current date
    if path is None:
        datestr = datetime.today().strftime('%y%m%d-%H%M')
        output_dir = Path("./data/{}".format(datestr)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)


    # Set up the parameters object and the clock and begin sweeping
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
