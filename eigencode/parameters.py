from constants import fundconst, lymanalpha
import numpy as np

fc = fundconst()
la = lymanalpha()


class Parameters:
    '''
    Stores physical parameters of the problem for a particular solve.

    Parameters
    ----------
    temp : float
        Temperature of the gas in K.
    tau0 : float
        Line center optical depth of the sphere.
    radius : float
        Radius of the sphere in cm.
    energy : float
        Impulse energy in erg (set to 1 for normalization)
    xsource : float
        Frequency of source photons, expressed as the difference between the
        source frequency and the line center in Doppler widths.
    alpha_abs : float
        Absorption coefficient in cm^{-1}.
    prob_dest : float
        Probability of destruction by collisional de-excitation.
    nsigma : float
        Number of frequency points to calculate solutions. Frequency resolution
        of the solution.
    nmax : float
        Maximum number of spatial eigenmodes to compute.
    mmax : float
        Maximum number of frequency eigenmodes to compute.

    Returns
    ----------
    `Parameters` object
        An object storing physical parameters as attributes.
    '''

    def __init__(self, temp, tau0, radius, energy, xsource,
                 alpha_abs, prob_dest, nsigma, nmax, mmax):
        self.temp = temp
        self.tau0 = tau0
        self.radius = radius
        self.energy = energy
        self.xsource = xsource
        self.alpha_abs = alpha_abs
        self.prob_dest = prob_dest

        self.vth = np.sqrt(2.0 * fc.kboltz * temp / fc.amu)                    # thermal velocity in cm s^{-1}
        self.Delta = la.nu0 * self.vth / fc.clight                             # doppler width in Hz
        self.a = la.Gamma / (4.0 * np.pi * self.Delta)                         # damping parameter
        self.sigma0 = la.osc_strength * fc.line_strength / \
            (np.sqrt(np.pi) * self.Delta)                                      # line center cross section in cm^2
        self.numden = tau0 / (self.sigma0 * radius)                            # H(1s) number density in cm^{-3}
        self.k = self.numden * fc.line_strength * la.osc_strength              # in nu units. tau0=k*radius/(sqrt(pi)*delta)
        self.xmax = np.rint(4.0 * (self.a * tau0)**0.333)                      # wing 1.e4 times smaller than center
        self.c1 = np.sqrt(2.0 / 3.0) * np.pi / (3.0 * self.a)                  # sigma(x) = c1*x^3, c1 = 0.855/a
        self.c2 = self.c1**(2.0 / 3.0) * self.a / (np.pi * self.Delta)         # phi(sigma) = c2 / sigma**(2.0/3.0), c2=0.287*a**(1.0/3.0)/Delta
        self.sigmas = self.c1 * xsource**3
        self.nsigma = nsigma
        self.nmax = nmax
        self.mmax = mmax
        self.sigma_offset = 1e3
        self.sigma_master = make_sigma_grids(self, xuniform=True)
        self.sigma = np.array(sorted(np.concatenate(list(self.sigma_master.values()))))


if __name__ == '__main__':
    pass
