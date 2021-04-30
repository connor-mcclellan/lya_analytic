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
    self.sigma_offset = 1e3
    self.sigma_master = make_sigma_grids(self, xuniform=True)
    self.sigma = np.array(sorted(np.concatenate(list(self.sigma_master.values()))))

def get_sigma_bounds(n, s, p):

    kappan=n*np.pi/p.radius
    wavenum = kappan*p.Delta/p.k
    phi_crit = wavenum**2 * fc.clight*p.k / ( 3.0*np.abs(s)*p.Delta**2 )
    x_tp = np.sqrt( p.a/(np.pi*p.Delta*phi_crit) )
    sigma_tp = p.c1*x_tp**3
    sigma_efold = p.k/(kappan*p.Delta)

    sigma_left = -(sigma_tp + 23*sigma_efold) # TODO: Parametrize?
    sigma_right = (sigma_tp + 23*sigma_efold)
    source = p.sigmas
    offset = p.sigma_offset
    pdb.set_trace()

    return ((sigma_left, min(source-offset, 0)), 
            (min(source-offset, 0), max(source+offset, 0)),
            (max(source+offset, 0), sigma_right))

def make_sigma_grids(p, xuniform=True): ## Make master sigma grid uniform in x
    
    width = p.c1 * 50 * p.a * p.tau0
    print("SIGMA GRID WIDTH: {:.1f}   ({:.0f} * tau0)    (x={:.1f})".format(width, width/p.tau0, np.cbrt(width/p.c1)))
    source = p.sigmas
    offset = p.sigma_offset
    left, middle, right = ((-width, min(source-offset, 0)), 
                           (min(source-offset, 0), max(source+offset, 0)),
                           (max(source+offset, 0), width))

    if xuniform:
        left = np.cbrt(np.array(left)/p.c1)
        right = np.cbrt(np.array(right)/p.c1)
        middle = np.cbrt(np.array(middle)/p.c1)
        source = p.xsource

    # Determine the integration ranges
    delimiters = sorted(np.array([left[0], source, 0., right[1]]))

    # Build an array that's evenly spaced between leftmost and rightmost sigma
    # The -1e-6 is to ensure the last data point, which is equal to sigma_right,
    # does not end up in a bin all by itself
    even_array = np.linspace(left[0], right[1]-1e-6, p.nsigma)

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
    leftgrid = np.linspace(*left, nleft)
    middlegrid = np.linspace(*middle, nmiddle)
    rightgrid = np.linspace(*right, nright)

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

