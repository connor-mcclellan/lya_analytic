
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from constants import fundconst,lymanalpha
#from rk import rk
import warnings
import pdb

# max number of solutions at each n
nsolnmax=20                                          # maximum number of solutions for each n.

# integration accuracy
relative_tol=1.e-10 # 1.49012e-8
absolute_tol=1.e-10 # 1.49012e-8

fc=fundconst()
la=lymanalpha()

class parameters:
  def __init__(self,temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax):
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
    self.sigmas=self.c1*xsource**3
    self.c2=self.c1**(2.0/3.0)*self.a/(np.pi*self.Delta)        # phi(sigma) = c2 / sigma**(2.0/3.0), c2=0.287*a**(1.0/3.0)/Delta
    self.nsigma=nsigma
    self.nmax=nmax

def line_profile(sigma,p):					# units of Hz^{-1}
  x=(np.abs(sigma)/p.c1)**(1.0/3.0)
  #line_profile = ( np.exp(-x**2)/np.sqrt(np.pi) + p.a/np.pi/(0.01+x**2) ) / p.Delta		# doppler and natural

  # TODO: What is going on with this line profile? 
  #line_profile = p.a/np.pi/x**2/p.Delta								# just natural. don't use sigma=0!

  line_profile = p.a / np.pi / (0.01 + x**2) / p.Delta
  return line_profile

'''
# Differential equation for 'infegrate'
def func(sigma, y, args):
  n, s, p = args
  J = y[0]
  dJ = y[1]
  phi=line_profile(sigma,p)
  kappan=n*np.pi/p.radius
  wavenum = kappan * p.Delta / p.k
  term1 = wavenum**2 
  term2 = 3.0*s*p.Delta**2*phi/(fc.clight*p.k)
  dydsigma=np.zeros(2)
  dydsigma[0]=dJ
  dydsigma[1]= (term1+term2) * J
  return dydsigma
'''

# Differential equation for odeint
def func(y,sigma,n,s,p):
  J = y[0]
  dJ = y[1]
  phi=line_profile(sigma,p)
  kappan=n*np.pi/p.radius
  wavenum = kappan * p.Delta / p.k
  term1 = wavenum**2 
  term2 = 3.0*s*p.Delta**2*phi/(fc.clight*p.k)
  dydsigma=np.zeros(2)
  dydsigma[0]=dJ
  dydsigma[1]= (term1+term2) * J
  return dydsigma


def integrate(sigma, y_start, n, s, p):
  sol = odeint(func, y_start, sigma, rtol=relative_tol, atol=absolute_tol, args=(n,s,p))
  #sol = rk(func, [sigma[0], sigma[-1]], y_start, t_eval=sigma, args=(n, s, p))
  return sol

def one_s_value(n,s,p):
  # solve for response given n and s

  sigma_left = -80.0*p.tau0
  sigma_right = 80.0*p.tau0

  # check if sigma endpoints ok
  phi=line_profile(sigma_left,p)
  kappan=n*np.pi/p.radius
  wavenum = kappan * p.Delta / p.k
  term1 = wavenum**2
  term2 = 3.0*s*p.Delta**2*phi/(fc.clight*p.k)
  if np.abs(term2/term1) > 1.0:
    warnings.warn("term2/term1 at the edge={}".format(term2/term1))
    quit()

  # Determine how many points belong in each integration range
  delimiters = sorted(np.array([sigma_left, p.sigmas, 0., sigma_right]))

  # One extra sigma because a single duplicated datapoint will be removed later
  naive_array = np.linspace(sigma_left, sigma_right-1e-3, p.nsigma+1)
  inds = np.digitize(naive_array, delimiters)

  nleft, nmiddle, nright = [len(inds[inds==i]) for i in range(1, 4)]
  if nmiddle < 3 and p.sigmas != 0.:
      warnings.warn('Middle grid is critically undersampled. Replacing with minimum of 3 points.')
      nmiddle = 3
      nright -= 2  # Make room for the new points
      nleft -= 1

  # Create grids
  leftgrid = np.linspace(sigma_left, min(p.sigmas, 0), nleft)
  middlegrid = np.linspace(0, p.sigmas, nmiddle)
  rightgrid = np.linspace(sigma_right, max(0, p.sigmas), nright)

  kappan=n*np.pi/p.radius
  wavenum = kappan*p.Delta/p.k # used for initial conditions

  # rightward integration
  J=1.0
  dJ=wavenum*J
  y_start=np.array( (J,dJ) )
  sol = integrate(leftgrid,y_start,n,s,p)
  Jleft=sol[:,0]
  dJleft=sol[:,1]
  A=Jleft[-1]
  B=dJleft[-1]

  # leftward integration
  J=1.0
  dJ=-wavenum*J
  y_start=np.array( (J,dJ) )
  sol = integrate(rightgrid,y_start,n,s,p)
  Jright=sol[:,0]
  dJright=sol[:,1]
  C=Jright[-1]
  D=dJright[-1]

  # middle integration
  # If source > 0, integrate rightward from 0 to source, matching at 0
  if p.sigmas > 0.:

      # Match initial conditions at 0 (leftward integration)
      J = Jleft[-1]
      dJ = dJleft[-1]
      y_start = np.array((J, dJ))

      # Find solution in middle region
      sol = integrate(middlegrid, y_start, n, s, p)
      Jmiddle=sol[:, 0]
      dJmiddle=sol[:, 1]

      # Set coefficients of matrix equation at 0
      A = Jmiddle[-1]
      B = dJmiddle[-1]

  # If source < 0, integrate leftward from 0 to source, matching at 0
  elif p.sigmas < 0.:

      # Match initial conditions at 0 (rightward integration)
      J = Jright[-1]
      dJ = dJright[-1]
      y_start = np.array((J, dJ))

      # Find solution in middle region
      sol = integrate(middlegrid, y_start, n, s, p)
      Jmiddle=sol[:, 0]
      dJmiddle=sol[:, 1]

      # Set coefficients of matrix equation at 0
      C = Jmiddle[-1]
      D = dJmiddle[-1]

  # If source = 0, do nothing
  else:
      Jmiddle = np.nan
      dJmiddle = np.nan

  # solution of the matrix equation
  scale_right = - 1.0/(D-B*(C/A)) * np.sqrt(6.0)/8.0 * n**2 * p.energy/(p.k*p.radius**3)
  scale_left = C/A * scale_right
  Jleft = Jleft * scale_left
  dJleft = dJleft * scale_left
  Jright = Jright * scale_right
  dJright = dJright * scale_right
  Jmiddle = Jmiddle * scale_left if p.sigmas > 0. else Jmiddle * scale_right
  dJmiddle = dJmiddle * scale_left if p.sigmas > 0. else Jmiddle * scale_right

  # offset to sigma source points
  offset = 1e-3


  if p.sigmas < 0.:

      # reorder middle grid for consistency
      middlegrid = middlegrid[::-1]
      Jmiddle = Jmiddle[::-1]
      dJmiddle = dJmiddle[::-1]

      # Offset must be applied to the last point of Jleft and first of Jmiddle
      middlegrid[0] += offset
      leftgrid[-1] -= offset

      # Remove duplicated point at 0
      Jright = Jright[:-1]
      dJright = dJright[:-1]
      rightgrid = rightgrid[:-1]

  elif p.sigmas > 0.:
      # Offset must be applied to last point of Jmiddle and last of Jright
      middlegrid[-1] -= offset
      rightgrid[-1] += offset

      # Remove duplicated point at 0
      Jleft = Jleft[:-1]
      dJleft = dJleft[:-1]
      leftgrid = leftgrid[:-1]
      
  else:
      # Remove duplicated point at 0
      Jright = Jright[:-1]
      dJright = dJright[:-1]
      rightgrid = rightgrid[:-1]


  # combine left, middle, and right in one array
  try:
      sigma=np.concatenate((leftgrid, middlegrid, rightgrid[::-1]))
      J = np.concatenate((Jleft, Jmiddle, Jright[::-1]))
      dJ = np.concatenate((dJleft, dJmiddle, dJright[::-1]))
  except:
      sigma=np.concatenate((leftgrid, rightgrid[::-1]))
      J = np.concatenate((Jleft, Jright[::-1]))
      dJ = np.concatenate((dJleft, dJright[::-1]))
  return sigma,J,dJ


def solve(s1,s2,s3,n,p):
  # iterate to find the eigenfrequency sres and eigenvector Jres(sigma)
  # three frequencies s1, s2, s3
  err=1.e20 # initialize error to be something huge
  i=0
  refine_pts = []
  refine_res = []
  refine_log = []
  while err>1.e-6:
    print(i, err, s1, s2, s3)
    print('before: s1={} s2={} s3={}'.format(s1, s2, s3))
    i=i+1
    sigma,J1,dJ1=one_s_value(n,s1,p)
    sigma,J2,dJ2=one_s_value(n,s2,p)
    sigma,J3,dJ3=one_s_value(n,s3,p)
    f1 = np.sum(np.abs(J1)) # sum of absolute values of ENTIRE spectrum
    f2 = np.sum(np.abs(J2)) # this is the size of the response!
    f3 = np.sum(np.abs(J3))
    err=np.abs((s3-s1)/s2) # error is fractional difference between eigenfrequencies

    if i in [13, 14, 15]:
      pdb.set_trace()

    # s is the imaginary part of frequency omega
    # J of sigma is the spectrum at all frequency points sigma
    #print i,err,s1,s2,s3,f1,f2,f3
    sl=0.5*(s1+s2) # s between s1 and s2
    sigma,Jl,dJl=one_s_value(n,sl,p)
    fl = np.sum(np.abs(Jl))
    sr=0.5*(s2+s3) # s between s2 and s3
    sigma,Jr,dJr=one_s_value(n,sr,p)
    fr = np.sum(np.abs(Jr))
    refine_pts.append([s1, sl, s2, sr, s3])
    refine_res.append([f1, fl, f2, fr, f3])
    refine_log.append([i-1, err])
    annotation = [
        'CASE 0 : fl>f1, fl>f2          Setting s3=s2, s2=sl...',
        'CASE 1 : f2>fl, f2>fr          Setting s1=sl, s3=sr...',
        'CASE 2 : fr>f2, fr>f3          Setting s1=s2, s2=sr...',
        'CASE -1: No condition satisfied'
    ]
# three sets of three points --- one of those sets will have a maximal response in the center
# find that maximum response
    iflag = -1
    if fl>f1 and fl>f2:
      s3=s2
      s2=sl
      iflag = 0
    elif f2>fl and f2>fr:
      s1=sl
      s3=sr
      iflag = 1
    elif fr>f2 and fr>f3:
      s1=s2
      s2=sr
      iflag = 2
    # exit and say something has gone bad
    elif f3 == max([f1, fl, f2, fr, f3]) or f1 == max([f1, fl, f2, fr, f3]):
      warnings.warn("peak is outside of refinement window")
#      plot_refinement(refine_pts, refine_res, refine_log, notes=annotation[iflag])
      pdb.set_trace()
      quit()

    print('after: f1={:.2E} fl={:.2E} f2={:.2E} fr={:.2E} f3={:.2E}'.format(f1, fl, f2, fr, f3))
    print('after: s1={} s2={} s3={}'.format(s1, s2, s3))
    print('iflag={}'.format(iflag))
#    plot_refinement(refine_pts, refine_res, refine_log, notes=annotation[iflag])

    if i==100:
      warnings.warn("too many iterations in solve")
      pdb.set_trace()
      quit()

  pdb.set_trace()
  print()
  # choose middle point to be eigenfrequency
  sres=s2
  # Response looks very close to the eigenvector when frequency is close to the eigenfrequency
  Jres = (J3-J1)*(s3-sres)*(s1-sres)/(s1-s3)
  # Slighly better estimate than using just J2 --- derivation in notes
  return sigma,sres,Jres


def plot_refinement(refine_pts, refine_res, refine_log, notes=None):
    #for j in range(len(refine_pts)):
      j = len(refine_pts)-1
      for i in range(len(refine_pts)):
          if j != i:
              plt.plot(refine_pts[i], refine_res[i], marker='|', label="i={}  err={:.1E}".format(*refine_log[i]), alpha=0.2)
      plt.plot(refine_pts[j], refine_res[j], marker='|', label="j={}  err={:.1E}".format(*refine_log[j]), alpha=1)
      for k, txt in enumerate(['s1', 'sl', 's2', 'sr', 's3']):
          plt.annotate(txt, (refine_pts[j][k], refine_res[j][k]))
          plt.annotate('{}={}'.format(txt, refine_pts[j][k]), (0.8, 0.5-0.05*k), xycoords='axes fraction')

      if notes is not None:
          plt.annotate(notes, (0.05, 1.05), xycoords='axes fraction')

      plt.legend(prop={'size': 6})
      plt.ylim((min(np.ndarray.flatten(np.array(refine_res))), 10*max(np.ndarray.flatten(np.array(refine_res)))))
      plt.xlim((min(refine_pts[j])-0.1*(max(refine_pts[j])-min(refine_pts[j])), max(refine_pts[j])+0.1*(max(refine_pts[j])-min(refine_pts[j]))))
      plt.yscale('log')
      plt.show()



def sweep(s,p):
  # loop over n and s=-i\omega. when you find a maximum in the size of the response, call the solve function
  # tabulate s(n,m) and J(n,m,sigma).

  Jsoln=np.zeros((p.nmax,nsolnmax,p.nsigma))
  ssoln=np.zeros((p.nmax,nsolnmax))
  for n in range(1,p.nmax):
    print ("n=",n)
    nsoln=-1
    norm=np.zeros(s.size)
    for i in range(s.size):
      sigma,J,dJ=one_s_value(n,s[i],p)
      norm[i]=np.sum(np.abs(J))
      print("nsoln,n,s,response=",nsoln,n,s[i],norm[i])
      if i>1 and norm[i-2]<norm[i-1] and norm[i]<norm[i-1]:
        nsoln=nsoln+1
        if nsoln>nsolnmax-1:
          break
        sigma,sres,Jres = solve(s[i-2],s[i-1],s[i],n,p)
        ssoln[n,nsoln]=sres
        Jsoln[n,nsoln,:]=Jres
  return sigma,ssoln,Jsoln


def get_Pnm(ssoln,sigma,Jsoln,p):
  Pnmsoln=np.zeros((p.nmax,nsolnmax))
  dsigma=sigma[1]-sigma[0]
  for n in range(1,p.nmax):
    for i in range(nsolnmax):
      if ssoln[n,i]==0.0:
        continue
      Pnmsoln[n,i] = np.sqrt(1.5) * p.Delta**2 * (16.0*np.pi**2*p.radius/3.0/p.k/p.energy) \
                     * (-1.0)**(n+1) * np.sum(Jsoln[n,i,:])*dsigma
  return Pnmsoln

def main():

  # choices
  energy=1.e0
  temp=1.e4
  tau0=1.e7
  radius=1.e11
  alpha_abs=0.0
  prob_dest=0.0
  xsource=2.0
  nmax=6+1
  nsigma=512
  nomega=10
  p = parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
  tdiff = (p.radius/fc.clight)*(p.a*p.tau0)**0.333

  s = np.arange(0.02,-15.0,-0.01)
  sigma,ssoln,Jsoln=sweep(s,p)

  output_data = np.array([energy,temp,tau0,radius,alpha_abs,prob_dest,xsource,nmax,nsigma,nomega,tdiff,sigma,ssoln,Jsoln])
  np.save('./eigenmode_data.npy',output_data,allow_pickle=True, fix_imports=True)
  

if __name__ == "__main__":
  main()
