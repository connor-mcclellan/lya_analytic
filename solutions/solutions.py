from util import Line, read_bin, voigtx_fast, Params
from scipy import linalg
from scipy.special import spherical_in
import astropy.units as u
import astropy.constants as c
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})


def monte_carlo(x, bins=64):

    # Plot histogram to get n and bin positions, then clear it
    n, bins, patches = plt.hist(x, bins=bins)
    plt.cla()

    # Calculate centers of bins and normalization factor
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    N = np.sum(n)
    norm = 1. / N / (bins[1:] - bins[:-1])
    err = 2. * np.sqrt(n / N**2.) * N

    return bincenters, n * norm, err * norm


def hsp_analytic(x, line, p, temp=1e4, radius=1e11,
                 L=1., tau0=1e7, xi=0.0, normed=True):

    # Quantities derived from constants
    vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)

    # Quantities derived from arguments
    delta = line.nu0 * vth / c.c.cgs.value
    a = line.gamma / (4.0 * np.pi * delta)
    print('a: ', a)
    phix = voigtx_fast(a, x)
    sigma = p.beta * x**3. / a
    sigma0 = line.strength / (np.sqrt(np.pi) * delta)
    sigmai = p.beta * xi**3. / a
    numden = tau0 / (sigma0 * radius)
    kx = numden * line.strength / delta
    z = (sigma - sigmai) / (kx * radius)

    # Equations
    Jprefac = np.sqrt(6.0) / (16.0 * np.pi**3.) * kx**2. * L / delta

    print('Jprefac: ', Jprefac)
    print('sigma: ', sigma)
    print('z: ', z)

    Hsp_analytic = Jprefac / (3.0 * phix) / (kx * radius)**3. * \
        (np.pi**2. / 2.0) / (1.0 + np.cosh(np.pi * z))
    print('cosh: ', 1 + np.cosh(np.pi * z))
    print('Hsp_analytic: ', Hsp_analytic)
    return Hsp_analytic


def h_bc(x, line, p, temp=1e4, radius=1e11, L=1., tau0=1e7, xi=0.0, **kwargs):

    # Quantities derived from line constants
    vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
    delta = line.nu0 * vth / c.c.cgs.value
    a = line.gamma / (4.0 * np.pi * delta)
    phix = voigtx_fast(a, x)
    sigma = p.beta * x**3. / a
    sigma0 = line.strength / (np.sqrt(np.pi) * delta)
    numden = tau0 / (sigma0 * radius)
    kx = numden * line.strength / delta

    # Frequency and wavenumber variables
    n = len(x)
    dsigma = 2.0 * (p.beta * np.max(x)**3. / a) / (n - 1)
    ds = 2.0 * np.pi / (n * dsigma)
    smax = ds * (n - 1) / 2.
    s = -smax + ds * np.arange(n)
    print('s: ', s)
    # Set up matrix equation solution vector
    b = np.sqrt(3.0) * hsp_analytic(x, line, p, **kwargs)
    print('b: ', b)

    # Bessel functions, derivative of bessel functions, ratio between them
    z = np.abs(kx * radius * s)
    i0 = spherical_in(0, z, derivative=False)
    di0 = spherical_in(0, z, derivative=True)
    rat = di0 / i0

    # Set matrix elements
    M = np.zeros((n, n), dtype=np.cdouble)
    for i in range(n):
        for j in range(n):
            M[i, j] = (ds / 2.0 / np.pi) * np.exp(1j * s[j] * sigma[i]) * \
                (1.0 + np.abs(s[j]) * rat[j] / np.sqrt(3.0) / phix[i])
    print('M:', M)
    # Scale each row
    for i in range(n):
        maxval = np.amax(np.absolute(M[i, :]))
        M[i, :] = M[i, :] / maxval
        b[i] = b[i] / maxval
    print('Scaled M: ', M)
    # Solve the matrix equation for the Fourier amplitudes
    amp = linalg.solve(M, b)
    print('amp: ', amp)

    # Equations for J and H homogeneous
#    Jh=np.zeros(n,dtype=np.cdouble)
    Hh = np.zeros(n, dtype=np.cdouble)
    for i in range(n):
        for j in range(n):
            #            Jh[i]=Jh[i] + (ds/2.0/np.pi)*np.exp(1j*s[j]*sigma[i])*amp[j]
            Hh[i] = Hh[i] + (ds / 2.0 / np.pi) * np.exp(1j * s[j] * sigma[i]) * \
                amp[j] * (-np.abs(s[j]) / (3.0 * phix[i])) * rat[j]

    print('Hh reals: ', np.real(Hh))
    return np.abs(np.real(Hh))


if __name__ == '__main__':

    # Set up line and get data from bin files
    lya = Line(1215.6701, 0.4164, 6.265e8)
    p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e4)
    mu, x, time = read_bin('../data/1m_tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/')

    norm = 4.0 * np.pi * p.R**2. * p.delta * 4.0 * np.pi / p.energy
    print('norm: ', norm)

    # Set up matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Histogram data from Monte Carlo x
    xbins, n, err = monte_carlo(x)
#    xbins2, n2, err2 = monte_carlo(x, bins=256)
    ax.errorbar(xbins, n, yerr=err, fmt='.', c='m', ms=3, label='Monte Carlo')
#    ax.errorbar(xbins2, n2, yerr=err2, fmt='.', c='c', ms=3, label='Monte Carlo 2', alpha=0.25)

    # H_0 from analytic solution
    ax.plot(
        xbins,
        hsp_analytic(
            xbins,
            lya,
            p) * norm,
        'g-',
        linewidth=1,
        alpha=0.5,
        label=r'$H_0$')

    # H_bc from Fourier solution
    ax.plot(
        xbins,
        h_bc(
            xbins,
            lya, 
            p) * norm,
        'c-',
        linewidth=1,
        alpha=0.5,
        label=r'$H_{bc}$')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$P(x)$')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # plt.savefig('x_pdf_log.pdf')
