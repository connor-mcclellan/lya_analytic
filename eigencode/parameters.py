from constants import fundconst,lymanalpha
import warnings
import numpy as np
import pdb

fc=fundconst()
la=lymanalpha()

class Parameters:
  def __init__(self,temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax):#,s):
    self.temp=temp                                              # temperature in K
    self.tau0=tau0                                              # line center optical depth of sphere
    self.radius=radius                                          # radius of sphere in cm
    self.energy=energy						# impulse energy in erg
    self.xsource = xsource
    self.alpha_abs=alpha_abs					# absorption coefficient in cm^{-1}
    self.prob_dest=prob_dest					# probability of destruction by collisional de-excitation
    self.vth = np.sqrt( 2.0 * fc.kboltz * temp / fc.amu )       # thermal velocity in cm s^{-1}
    self.Delta = la.nu0*self.vth/fc.clight                      # doppler width in Hz
    self.a = la.Gamma/(4.0*np.pi*self.Delta)                    # damping parameter
    self.sigma0 = la.osc_strength * fc.line_strength/(np.sqrt(np.pi)*self.Delta)  # line center cross section in cm^2
    self.numden = tau0/(self.sigma0*radius)                     # H(1s) number density in cm^{-3}
    self.k=self.numden*fc.line_strength*la.osc_strength         # in nu units. tau0=k*radius/(sqrt(pi)*delta)
    self.xmax=np.rint(4.0*(self.a*tau0)**0.333)                 # wing 1.e4 times smaller than center
    self.c1=np.sqrt(2.0/3.0)*np.pi/(3.0*self.a)               	# sigma(x) = c1*x^3, c1 = 0.855/a
    self.c2 = self.c1**(2.0/3.0)*self.a/(np.pi*self.Delta)      # phi(sigma) = c2 / sigma**(2.0/3.0), c2=0.287*a**(1.0/3.0)/Delta
    self.sigmas=self.c1*xsource**3
    self.nsigma=nsigma
    self.nmax=nmax
    self.sigma_bounds = get_sigma_bounds(self.nmax, self)
    self.sigma_offset = self.sigma_bounds[1]/self.nsigma # Should be around ~7e3 for integrator to be deterministic near source
    self.sigma_master = make_sigma_grids(self.nmax, self)


def get_sigma_bounds(n, p):
    gam_0 = n**2 * fc.clight / (p.a * p.tau0)**(1/3) / p.radius
    sigma_tp = p.tau0 * (0.05 / gam_0)**(3/2.) # Location of first resonance hardcoded in
    sigma_efold = p.tau0 / np.sqrt(np.pi) / n

    sigma_left = -(sigma_tp + 40*sigma_efold) # TODO: Parametrize 23?
    sigma_right = (sigma_tp + 40*sigma_efold)

    return sigma_left, sigma_right


def make_sigma_grids(n, p): ## Make master sigma grid uniform in x

    sigma_left, sigma_right = get_sigma_bounds(n, p)

    # Determine the integration ranges
    delimiters = sorted(np.array([sigma_left, p.sigmas, 0., sigma_right]))

    # Build an array that's evenly spaced between leftmost and rightmost sigma
    # The -1e-3 is to ensure the last data point, which is equal to sigma_right,
    # does not end up in a bin all by itself
    even_array = np.linspace(sigma_left, sigma_right-1e-3, p.nsigma)

    # Bin this evenly-spaced array into the integration ranges we found earlier
    # These are the indices that specify which bin the data points fall into:
    inds = np.digitize(even_array, delimiters)

    # Count how many points have fallen into each integration range
    nleft, nmiddle, nright = [len(inds[inds==i]) for i in range(1, 4)]

    # For a nonzero source, make sure that the middle grid has enough points
    if nmiddle < 4 and p.sigmas != 0.:
        warnings.warn('Middle grid is critically undersampled. Adding 4 points.')
        nmiddle += 4
        nright -= 2
        nleft -= 2

    ### Create grids
    # Leftgrid goes from sigma_left to whichever is smaller: source, or 0
    # Its values will be ordered from small to large --- increasingly
    leftgrid = np.linspace(sigma_left, min(p.sigmas, 0), nleft)
    # Middle grid goes from 0 to source
    # Its values are either ordered increasingly or decreasingly, depending  
    # on whether source>0 or source<0, respectively
    middlegrid = np.linspace(0, p.sigmas, nmiddle)
    # Right grid goes from sigma_right to whichever is larger: source, or 0
    # Its values are always ordered decreasingly
    rightgrid = np.linspace(sigma_right, max(0, p.sigmas), nright)

    # Set an offset applied about source
    # This helps resolve the dJ discontinuity better. If it is not large enough,
    # the integrator will not be deterministic near the source
    if np.abs(leftgrid[-1]-p.sigma_offset) > np.abs(leftgrid[-2]-leftgrid[-1]):
        # If offset is larger than bin spacing, split the distance to the end
        p.sigma_offset = np.abs(leftgrid[-2]-leftgrid[-1])/2.
    if len(middlegrid) != 0 and p.sigma_offset > np.diff(middlegrid)[0]:
        p.sigma_offset = np.diff(middlegrid)[0]/2

    if p.sigmas < 0.:
      middlegrid[-1] += p.sigma_offset
      leftgrid[-1] -= p.sigma_offset
    elif p.sigmas > 0.:
      middlegrid[-1] -= p.sigma_offset
      rightgrid[-1] += p.sigma_offset
    else:
      leftgrid[-1] -= p.sigma_offset
      rightgrid[-1] += p.sigma_offset
    return {'left':leftgrid, 'middle':middlegrid, 'right':rightgrid}


if __name__ == '__main__':
    energy=1.e0
    temp=1.e4
    tau0=1.e7
    radius=1.e11
    alpha_abs=0.0
    prob_dest=0.0
    xsource=2.0
    nmax=6
    nsigma=512
    nomega=10
    s = np.arange(0.2,-15.0,-0.01)
    p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
    pdb.set_trace()
