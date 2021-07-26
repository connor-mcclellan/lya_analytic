import numpy as np
import pdb
import pickle
from constants import fundconst
fc = fundconst()

def construct_sol(directory, nmax, mmax, nmin=1):
    '''
    Loads numpy arrays from an output directory and constructs them in memory.

    Parameters
    ----------
    directory : `pathlib.Path` object
          
    '''

    data = np.load(directory/"n{:03d}_m{:03d}.npy".format(nmin, 1), allow_pickle=True).item()
    Jsoln = np.zeros((nmax-nmin+1, mmax, len(data['J'])))
    ssoln = np.zeros((nmax-nmin+1, mmax))
    intJsoln = np.zeros((nmax-nmin+1, mmax))

    for n in range(nmin, nmax+1):
        for m in range(1, mmax+1):
            data = np.load(directory/"n{:03d}_m{:03d}.npy".format(n, m), allow_pickle=True).item()
            Jsoln[n-nmin, m-1, :] = data['J']
            ssoln[n-nmin, m-1] = data['s']
            intJsoln[n-nmin, m-1] = data['Jint']

    p = pickle.load(open(directory/'parameters.p', 'rb'))
    p.nmax = nmax
    p.mmax = mmax

    return Jsoln, ssoln, intJsoln, p

def midpoint_diff(t):
  midpoints = 0.5*(t[1:]+t[:-1])
  dt = np.diff(midpoints)
  dt = np.insert(dt, 0, midpoints[1] - midpoints[0])
  dt = np.append(dt, midpoints[-1] - midpoints[-2])
  return dt

def get_Pnm(ssoln, intJsoln, p):
    n = np.arange(1, p.nmax+1)
    Pnmsoln = (np.sqrt(1.5) * 16.0*np.pi**2 * p.radius * p.Delta**2 
              / (3.0 * p.k * p.energy) * (-1.0)**(n) / ssoln.T
              * intJsoln.T).T
    return Pnmsoln

def gamma(n, m, p): 
    return 2**(-1/3.) * np.pi**(13/6.)*n**(4/3.)*(m-7/8.)**(2/3.)*fc.clight/p.radius/(p.a * p.tau0)**(1/3.)

def dgamma(n, m, p):
    return (2/3.) * 2**(-1./3) * np.pi**(13./6)*n**(4./3)*(m-7./8)**(-1./3)*fc.clight/p.radius/(p.a * p.tau0)**(1/3.)

def waittime(Jsoln, ssoln, intJsoln, t, p):
    Pnmsoln = get_Pnm(ssoln, intJsoln, p)
    P = np.sum(np.sum(
         - np.expand_dims(Pnmsoln, 2) * np.expand_dims(ssoln, 2)
         * np.exp(np.expand_dims(ssoln, 2) * t),
        axis=0), axis=0)
    return P

def line_profile(sigma, p):					# units of Hz^{-1}
    line_profile = (2 * p.a / 27 / np.pi)**(1./3) / (np.abs(sigma) + 0.01)**(2./3) / p.Delta
    return line_profile

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

    return ((sigma_left, min(source, 0-offset)), 
            (min(source, 0+offset), max(source, 0-offset)),
            (max(source, 0+offset), sigma_right))

def scinot(num):
    ''' 
    Formats numbers in scientific notation for LaTeX.
    '''
    numstrs = '{:.1E}'.format(num).split('E+')
    return r'${} \times 10^{{{}}}$'.format(numstrs[0], int(numstrs[1]))

def make_sigma_grids(p, xuniform=True): ## Make master sigma grid uniform in x
    
    width = p.c1 * 50 * p.a * p.tau0
    print("SIGMA GRID WIDTH: {:.1f}   ({:.0f} * tau0)    (x={:.1f})".format(width, width/p.tau0, np.cbrt(width/p.c1)))
    source = p.sigmas
    offset = p.sigma_offset
    left, middle, right = ((-width, min(source, 0-offset)), 
                           (min(source, 0+offset), max(source, 0-offset)),
                           (max(source, 0+offset), width))

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
