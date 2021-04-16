from constants import fundconst,lymanalpha
import warnings
import numpy as np
import pdb

fc=fundconst()
la=lymanalpha()

class Parameters:
  def __init__(self,temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax,mmax):
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
    self.mmax=mmax
    self.sigma_bounds = get_sigma_bounds(self.nmax, self.mmax, self)
    self.sigma_offset = 1e3#self.sigma_bounds[1]/self.nsigma/1e2
    self.sigma_master = make_sigma_grids(self.nmax, 1, self, xuniform=True)


def gamma(n, m, p): 
     return 2**(-1/3) * np.pi**(13/6)*n**(4/3)*(m-7/8)**(2/3)*fc.clight/p.radius/(p.a * p.tau0)**(1/3)


def get_sigma_bounds(n, m, p):
    gam_0 = fc.clight / (p.a * p.tau0)**(1/3) / p.radius
    gam_max = gamma(n, m, p)
    sigma_tp = p.tau0 * (gam_max / gam_0)**(3/2.)
    sigma_efold = p.tau0 / np.sqrt(np.pi) / n

    sigma_left = -(sigma_tp + 40*sigma_efold) # TODO: Parametrize?
    sigma_right = (sigma_tp + 40*sigma_efold)
#    pdb.set_trace()
    return sigma_left, sigma_right


def make_sigma_grids(n, m, p, xuniform=False): ## Make master sigma grid uniform in x

    left, right = get_sigma_bounds(n, m, p)
    source = p.sigmas
    offset = p.sigma_offset

    if xuniform:
        left = np.cbrt(left/p.c1)
        right = np.cbrt(right/p.c1)
        source = p.xsource
        offset = np.cbrt(p.sigma_offset/p.c1)

    # Determine the integration ranges
    delimiters = sorted(np.array([left, source, 0., right]))

    # Build an array that's evenly spaced between leftmost and rightmost sigma
    # The -1e-3 is to ensure the last data point, which is equal to sigma_right,
    # does not end up in a bin all by itself
    even_array = np.linspace(left, right-1e-6, p.nsigma)

    # Bin this evenly-spaced array into the integration ranges we found earlier
    # These are the indices that specify which bin the data points fall into:
    inds = np.digitize(even_array, delimiters)

    # Count how many points have fallen into each integration range
    nleft, nmiddle, nright = [len(inds[inds==i]) for i in range(1, 4)]

    # For a nonzero source, make sure that the middle grid has enough points
    if nmiddle < 4 and source != 0.:
        warnings.warn('Middle grid is critically undersampled. Adding 4 points.')
        nmiddle += 4
        nright -= 2
        nleft -= 2

    ### Create grids
    # Leftgrid goes from sigma_left to whichever is smaller: source, or 0
    # Its values will be ordered from small to large --- increasingly
    leftgrid = np.linspace(left, min(source, 0), nleft)
    # Middle grid goes from 0 to source
    # Its values are either ordered increasingly or decreasingly, depending  
    # on whether source>0 or source<0, respectively
    middlegrid = np.linspace(0, source, nmiddle)
    # Right grid goes from sigma_right to whichever is larger: source, or 0
    # Its values are always ordered decreasingly
    rightgrid = np.linspace(right, max(0, source), nright)

    # Set an offset applied about source
    # This helps resolve the dJ discontinuity better. If it is not large enough,
    # the integrator will not be deterministic near the source
    if np.abs(leftgrid[-1]-offset) > np.abs(leftgrid[-2]-leftgrid[-1]):
        # If offset is larger than bin spacing, split the distance to the end
        offset = np.abs(leftgrid[-2]-leftgrid[-1])/2.
    if len(middlegrid) != 0 and offset > np.diff(middlegrid)[0]:
        offset = np.diff(middlegrid)[0]/2

    if source < 0.:
      middlegrid[-1] += offset
      leftgrid[-1] -= offset
    elif source > 0.:
      middlegrid[-1] -= offset
      rightgrid[-1] += offset
    else:
      leftgrid[-1] -= offset
      rightgrid[-1] += offset

    if xuniform:
        leftgrid = p.c1 * leftgrid**3.
        rightgrid = p.c1 * rightgrid**3.
        middlegrid = p.c1 * middlegrid**3.

    return {'left':leftgrid, 'middle':middlegrid, 'right':rightgrid}


if __name__ == '__main__':
    energy=1.e0
    temp=1.e4
    tau0=1.e7
    radius=1.e11
    alpha_abs=0.0
    prob_dest=0.0
    xsource=0.0
    nmax=6
    nsigma=512
    nomega=10
    s = np.arange(0.2,-15.0,-0.01)
    p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
    pdb.set_trace()
