import numpy as np
import matplotlib.pyplot as plt
from constants import fundconst,lymanalpha
from scipy.integrate import solve_ivp, odeint
from parameters import Parameters, make_sigma_grids
#from rk import rk
import warnings
import pdb

# TODO:
# [X] Add in master sigma grid
# [X] Rewrite sol to use interpolants on master sigma grid
# [ ] Insert zero at line center for Jsoln?
# [ ] Zero is duplicated for off-center sources - fix it!
# [ ] rtol and atol convergence


# max number of solutions at each n
nsolnmax=20
fc=fundconst()
la=lymanalpha()

def line_profile(sigma,p):					# units of Hz^{-1}
  x=(np.abs(sigma)/p.c1)**(1.0/3.0)
  #line_profile = ( np.exp(-x**2)/np.sqrt(np.pi) + p.a/np.pi/(0.01+x**2) ) / p.Delta		# doppler and natural
  line_profile = p.a / np.pi / (x**2) / p.Delta
  return line_profile

# Differential equation for solve_ivp
def func(sigma,y,n,s,p):
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
  '''
  Returns interpolants which can be used to evaluate the function at any sigma
  '''
  sol = solve_ivp(func, [sigma[0], sigma[-1]], y_start, args=(n,s,p), rtol=1e-10, atol=1e-10, dense_output=True)
  return sol.y.T, sol.sol


def one_s_value(n,s,p, debug=False, trace=False):
  '''Solve for response given n and s'''

  kappan=n*np.pi/p.radius
  wavenum = kappan * p.Delta / p.k

  # Construct grids for this value of n
  leftgrid, middlegrid, rightgrid = make_sigma_grids(n, p).values()

  # rightward integration
  J=1.0
  dJ=wavenum*J
  y_start=np.array( (J,dJ) )
  sol, interp_left = integrate(leftgrid,y_start,n,s,p)
  Jleft=sol[:,0]
  dJleft=sol[:,1]
  A=Jleft[-1]    # Set matrix coefficient equal to Jleft's rightmost value
  B=dJleft[-1]

  # leftward integration
  J=1.0
  dJ=-wavenum*J
  y_start=np.array( (J,dJ) )
  sol, interp_right = integrate(rightgrid,y_start,n,s,p)
  Jright=sol[:,0]
  dJright=sol[:,1]
  C=Jright[-1]   # Set matrix coefficient equal to Jright's leftmost value
  D=dJright[-1]

  # middle integration
  # If source > 0, integrate rightward from 0 to source, matching at 0
  # Middlegrid is ordered increasingly, starting at 0 and going to source
  if p.sigmas > 0.:

      # Match initial conditions at 0 (leftward integration)
      J = Jleft[-1]
      dJ = dJleft[-1]
      y_start = np.array((J, dJ))

      # Find solution in middle region
      sol, interp_middle = integrate(middlegrid, y_start, n, s, p)
      Jmiddle=sol[:,0]
      dJmiddle=sol[:,1]

      # Set coefficients of matrix equation at the source
      A = Jmiddle[-1]    # Overwrite previous matrix coefficients
      B = dJmiddle[-1]

  # If source < 0, integrate leftward from 0 to source, matching at 0
  # Middlegrid is ordered decreasingly, starting at zero and going to source
  elif p.sigmas < 0.:

      # Match initial conditions at 0 (rightward integration)
      J = Jright[-1]
      dJ = dJright[-1]
      y_start = np.array((J, dJ))

      # Find solution in middle region
      sol, interp_middle = integrate(middlegrid, y_start, n, s, p)
      Jmiddle=sol[:,0]
      dJmiddle=sol[:,1]

      # Set coefficients of matrix equation at the source
      C = Jmiddle[-1]   # Overwrite previous matrix coefficients
      D = dJmiddle[-1]

  # If source = 0, do nothing
  else:
      Jmiddle = np.nan
      dJmiddle = np.nan

  # solution of the matrix equation
  scale_right = - 1.0/(D-B*(C/A)) * np.sqrt(6.0)/8.0 * n**2 * p.energy/(p.k*p.radius**3)
  scale_left = C/A * scale_right
  scale_middle = scale_left if p.sigmas > 0. else scale_right

  # Scale solutions by each factor to match discontinuity
  Jleft, dJleft = interp_left(p.sigma_master['left']) * scale_left
  Jright, dJright = interp_right(p.sigma_master['right']) * scale_right
  if Jmiddle is not np.nan:
      Jmiddle, dJmiddle = interp_middle(p.sigma_master['middle']) * scale_middle

  if p.sigmas < 0.:
      # reorder middle grid to be increasing, starting from source & going to 0
      Jmiddle = Jmiddle[::-1]
      dJmiddle = dJmiddle[::-1]

  # combine left, middle, and right in one array
  try:
      J = np.concatenate((Jleft, Jmiddle, Jright[::-1]))
      dJ = np.concatenate((dJleft, dJmiddle, dJright[::-1]))
  except:
      J = np.concatenate((Jleft, Jright[::-1]))
      dJ = np.concatenate((dJleft, dJright[::-1]))

  return J,dJ


def solve(s1,s2,s3,n,p):
  # iterate to find the eigenfrequency sres and eigenvector Jres(sigma)
  # three frequencies s1, s2, s3
  err=1.e20 # initialize error to be something huge
  i=0

  refine_pts = []
  refine_res = []
  refine_log = []

  while err>1.e-6:
    J1,dJ1=one_s_value(n,s1,p)
    J2,dJ2=one_s_value(n,s2,p)
    J3,dJ3=one_s_value(n,s3,p)

    f1 = np.sum(np.abs(J1)) # sum of absolute values of ENTIRE spectrum
    f2 = np.sum(np.abs(J2)) # this is the size of the response!
    f3 = np.sum(np.abs(J3))
    err=np.abs((s3-s1)/s2) # error is fractional difference between eigenfrequencies

    sl=0.5*(s1+s2) # s between s1 and s2
    Jl,dJl=one_s_value(n,sl,p)
    fl = np.sum(np.abs(Jl))
    sr=0.5*(s2+s3) # s between s2 and s3
    Jr,dJr=one_s_value(n,sr,p)
    fr = np.sum(np.abs(Jr))

    #####
    refine_pts.append([s1, sl, s2, sr, s3])
    refine_res.append([f1, fl, f2, fr, f3])
    refine_log.append([i, err])
    #####

# three sets of three points --- one of those sets will have a maximal response in the center
# find that maximum response
    if fl>f1 and fl>f2:
      s3=s2
      s2=sl
    elif f2>fl and f2>fr:
      s1=sl
      s3=sr
    elif fr>f2 and fr>f3:
      s1=s2
      s2=sr

    if i==100:
      warnings.warn("too many iterations in solve")
      plot_refinement(refine_pts, refine_res, refine_log)
      pdb.set_trace()

    i=i+1

  # choose middle point to be eigenfrequency
  sres=s2
  # Response looks very close to the eigenvector when frequency is close to the eigenfrequency
  Jres = (J3-J1)*(s3-sres)*(s1-sres)/(s1-s3)
  # Slighly better estimate than using just J2 --- derivation in notes
  return sres,Jres


def plot_refinement(refine_pts, refine_res, refine_log):
    #for j in range(len(refine_pts)):
      j = len(refine_pts)-1
      for i in range(len(refine_pts)):
          if j != i:
              plt.plot(refine_pts[i], refine_res[i], marker='|', label="i={}  err={:.1E}".format(*refine_log[i]), alpha=0.2)
      plt.plot(refine_pts[j], refine_res[j], marker='|', label="j={}  err={:.1E}".format(*refine_log[j]), alpha=1)
      for k, txt in enumerate(['s1', 'sl', 's2', 'sr', 's3']):
          plt.annotate(txt, (refine_pts[j][k], refine_res[j][k]))
          plt.annotate('{}={}'.format(txt, refine_pts[j][k]), (0.8, 0.5-0.05*k), xycoords='axes fraction')

      plt.legend(prop={'size': 6})
      plt.ylim((min(np.ndarray.flatten(np.array(refine_res))), 10*max(np.ndarray.flatten(np.array(refine_res)))))
      plt.yscale('log')
      plt.show()


def sweep(p):
  # loop over n and s=-i\omega. when you find a maximum in the size of the response, call the solve function
  # tabulate s(n,m) and J(n,m,sigma).

  Jsoln=np.zeros((p.nmax,nsolnmax,p.nsigma))
  ssoln=np.zeros((p.nmax,nsolnmax))
  for n in range(1,p.nmax+1):
    print ("n=",n)
    nsoln=0

    # TODO: s_start and s_incr: make parameters in the future
    s = 0.2
    s_increment = -0.01

    norm=[]
    while nsoln < nsolnmax:
      J,dJ=one_s_value(n,s,p)
      norm.append(np.sum(np.abs(J)))
      print("nsoln,n,s,response=",nsoln,n,s,norm[-1])
      if len(norm)>2 and norm[-3]<norm[-2] and norm[-1]<norm[-2]:
        nsoln=nsoln+1
        sres,Jres = solve(s-2*s_increment,s-s_increment,s,n,p)
        ssoln[n-1,nsoln-1]=sres
        Jsoln[n-1,nsoln-1,:]=Jres
      s += s_increment
  return ssoln,Jsoln


def main():
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
#  s = np.arange(0.2,-15.0,-0.01)
  p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)
  tdiff = (p.radius/fc.clight)*(p.a*p.tau0)**0.333
  ssoln,Jsoln=sweep(p)
  sigma = np.array(sorted(np.concatenate(list(p.sigma_master.values()))))
  output_data = np.array([energy,temp,tau0,radius,alpha_abs,prob_dest,xsource,nmax,nsigma,nomega,tdiff,sigma,ssoln,Jsoln])
  np.save('./data/eigenmode_data_xinit{:.0f}_tau{:.0e}_n{}_m{}_smalloffset.npy'.format(xsource, tau0, p.nmax, nsolnmax).replace('+0',''),output_data, allow_pickle=True, fix_imports=True)

if __name__ == "__main__":
  main()
