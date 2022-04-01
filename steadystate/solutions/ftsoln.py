
import numpy as np
from scipy.special import wofz
from scipy import integrate as integrate
from scipy.special import spherical_in
from scipy import interpolate
from scipy import linalg
from solutions.util import voigtx, voigtx_full, midpoint_diff
import matplotlib.pyplot as plt
import pdb
from astropy.utils.console import ProgressBar

# fundamental constants

cgrav=6.67429e-8
hplanck=6.62606896e-27
clight=2.997924589e10
kboltz=1.3806504e-16
charge=4.80320427e-10
abohr=0.52917720859e-8
melectron=9.10938215e-28
mproton=1.672621637e-24
amu=1.660538782e-24

# Lyman alpha stuff

lambda0 = 1215.6701 * 1.e-8
osc_strength = 0.4164
Gamma= 6.265e8
nu0 = clight / lambda0
line_strength = np.pi*charge**2/(melectron*clight) * osc_strength

def init():
  global temp,tau0,radius
  global vth,delta,a,sigma0,numden,kx
  global xmax,beta

  vth = np.sqrt( 2.0 * kboltz * temp / amu )
  delta = nu0*vth/clight
  a = Gamma/(4.0*np.pi*delta)
  sigma0 = line_strength/(np.sqrt(np.pi)*delta)	# line center cross section
  numden = tau0 / (sigma0*radius)
  kx=numden*line_strength/delta			# in x units. tau0 = kx*radius/sqrt(pi).
  xmax=np.rint(4.0*(a*tau0)**0.333)		# wing 1.e4 times smaller than center
  print('XMAX: ', xmax)
  beta=np.sqrt(2.0/3.0)*np.pi/3.0		# sigma(x) = beta*x^3/a, beta = 0.855

##########################################################################################################

def gaussianx(x):
  line_profile = np.exp( -x**2 ) / np.sqrt(np.pi)
  return line_profile

def lorentzianx(x):
  global a
  line_profile = a/np.pi/(x**2 + a**2)
  return line_profile

def test_line_profile():
  global a
  x=np.arange(-100.0,100.0,0.1)
  plt.plot(x,voigtx(x),label='voigt')
  plt.plot(x,voigtx_fast(a, x),label='approx voigt')
  plt.plot(x,np.abs(voigtx(x)-voigtx_fast(a, x)),label='diff')
  #plt.plot(x,gaussianx(x))
  #plt.plot(x,lorentzianx(x))
  plt.legend()
  plt.yscale('log')
  plt.ylim((1.e-8,1.1))
  plt.xlabel('x')
  plt.ylabel('$\phi(x)$')
  plt.show()

def sigma_func(x):
  global a
  phix = voigtx(a, x)
  integrand = np.sqrt(2.0/3.0)/phix
  return integrand

def get_sigma(x):				# this uses numerical integration and is slow.
  result = integrate.quad(sigma_func,0.0,x)
  return result[0]

def test_sigma():
  global a
  x=np.arange(0.0,10.0,0.1)
  y=np.zeros(x.size)
  for i in range(x.size):
    y[i]=get_sigma(x[i])
  plt.plot(x,y,label='exact')
  plt.plot(x,np.sqrt(2.0/3.0)*np.pi*x**3/(3.0*a),label='wing')
  plt.plot(x,np.abs(y-np.sqrt(2.0/3.0)*np.pi*x**3/(3.0*a)),label='difference')
  plt.legend(loc='best')
  plt.yscale('log')
  plt.xlabel('x')
  plt.ylabel('$\sigma$')
  plt.show()

##########################################################################################################

# create x and sigma arrays which are uniform in x

def get_arrays_uniform_in_x():
  global n,xmax
  global x,sigma,dx

  dx=2.0*xmax/(n-1.0)
  x=np.zeros(n)
  for i in range(n):
    x[i]=-xmax+dx*i
  sigma=np.zeros(x.size)
  for i in range(x.size):
    sigma[i]=get_sigma(x[i])		# uses numerical integration and is slow
  dsigma = midpoint_diff(sigma)

# slow version with numerical integration for sigma(x) and then interpolation
def get_arrays_uniform_in_sigma_slow():
  global n,sigmamax
  global x,sigma,dsigma
  print("Getting arrays uniform in sigma...")
  get_arrays_uniform_in_x()	# use these for interpolation
  oldx=x
  oldsigma=sigma

  sigmamax=get_sigma(xmax)
  print("SIGMAMAX = ", sigmamax)
  dsigma=2.0*sigmamax/(n-1)
  sigma=np.zeros(n)
  x=np.zeros(n)
  for i in range(n):
    sigma[i]=-sigmamax+dsigma*i

  f = interpolate.interp1d(oldsigma,oldx)
  x[0]=oldx[0]
  x[n-1]=oldx[n-1]
  for i in range(1,n-1):
    x[i] = f(sigma[i])

  print("Done!")

# use analytic sigma(x) = beta*x**3/a
def get_arrays_uniform_in_sigma_fast():
  global n,sigmamax,a,beta,xmax
  global x,sigma,dsigma

  sigmamax=beta*xmax**3/a
  sigma = np.linspace(-sigmamax, sigmamax, n)
  dsigma = np.diff(sigma)[0]
  x = np.cbrt(sigma*a/beta)


def get_s():                       # assumes sigma array is evenly spaced, from get_sigma_x_arrays
  global n,dsigma,sigma
  global s,ds
  ds=2.0*np.pi/(n*dsigma)
  smax=ds*(n-1.0)/2.0
  s=np.zeros(n)
  for j in range(n):
    s[j]=-smax + ds*j


def get_s_nonuniform():
  global n, dsigma, sigma
  global s, ds
  ds = 2.0*np.pi/(n*midpoint_diff(sigma))
  smax = np.pi*(n-1)**2/(2*n*np.abs(sigma[0]))
  s = np.concatenate((np.array([-smax]), -smax + np.cumsum(ds[1:])))
  dsigma = midpoint_diff(sigma)
##########################################################################################################


def particular_solution(r):
  global kx,sigma,sigmai,L,delta,phix_full
  global Jp,Hp
  r4sq = (kx*r)**2 + (sigma-sigmai)**2
  Jprefac = np.sqrt(6.0)/(16.0*np.pi**3) * kx**2*L/delta
  Jp = Jprefac / r4sq
  Hp = 1.0/(3.0*kx*phix_full)*Jprefac*(2.0*r*kx**2) / r4sq**2

def test_particular_solution():
  global x,Jp,Hp
  plt.plot(x,Jp,label='Jp')
  plt.plot(x,Hp,label='Hp')
  plt.legend(loc='best')
  plt.yscale('log')
  plt.xlabel('$x$')
  plt.ylabel('$J_p,H_p$')
  plt.show()

##########################################################################################################


def f_surf(z):
  f = (np.pi**2/2.0)/(1.0+np.cosh(np.pi*z)) - 2.0/(1.0+z**2)**2
  return f

def surface_solution_numerical():  # inward extension of J=-J_p at r=R.
  global n,Jp,kx,L,delta,radius,s,sigmai,phix_full
  global Js,Jscheck,Hs

  Js=-Jp
  amps = np.zeros(n,dtype=np.cdouble)
  Jprefac = np.sqrt(6.0)/(16.0*np.pi**3) * kx**2*L/delta
  amps = - np.pi * Jprefac / (kx*radius) * np.exp( -kx*radius*np.abs(s) - 1j*s*sigmai )
  Jscheck=np.zeros(n,dtype=np.cdouble)
  Hs=np.zeros(n,dtype=np.cdouble)
  print('Solving surface solution...\n')
  pb = ProgressBar(n*n)
  for i in range(n):
    for j in range(n):
      z = np.abs(kx*radius*s[j])
      i0=spherical_in(0,z,derivative=False)
      di0=spherical_in(0,z,derivative=True)
      rat = di0/i0
      Hs[i]=Hs[i]+(ds/2.0/np.pi)*np.exp(1j*s[j]*sigma[i])*amps[j]*(-np.abs(s[j])*rat/3.0/phix_full[i])
      Jscheck[i]=Jscheck[i]+(ds/2.0/np.pi)*np.exp(1j*s[j]*sigma[i])*amps[j]
      pb.update()

def surface_solution_analytic():
  global n,Jp,kx,L,delta,radius,sigmai,phix_full
  global Hs_analytic,Hsp_analytic,Jsp_analytic 

  Jprefac = np.sqrt(6.0)/(16.0*np.pi**3) * kx**2*L/delta
  Hs_analytic = np.zeros(n,dtype=np.double)
  Hs_analytic_prefac = Jprefac/(3.0*phix_full*(kx*radius)**3)
  z = (sigma-sigmai)/(kx*radius)
  Js_analytic = -Jp
  Hs_analytic = Hs_analytic_prefac * f_surf( z )
  Jsp_analytic=np.zeros(n)
  Hsp_analytic = Jprefac/(3.0*phix_full) / (kx*radius)**3 * (np.pi**2/2.0)/(1.0+np.cosh(np.pi*z))
  norm = 4.0*np.pi*radius**2.*delta*4.0*np.pi/L


def test_surface_particular_solution():
  global x,Jp,Hp,Js,Jscheck,Hs,Hs_analytic,Jsp_analytic
  #plt.plot(x,np.abs(Jp),label='Jp')
  #plt.plot(x,np.abs(Hp),linewidth=2,label='Hp')
  #plt.plot(x,np.abs(Hs),label='numerical Hs')
  #plt.plot(x,np.abs(Hs_analytic),label='analytic Hs')
  #plt.plot(x,np.abs(Hs),linewidth=2,label='Hs')
  #plt.plot(x,np.abs(Hs_analytic),'o',linewidth=2,label='analytic Hs')
  plt.plot(x,np.abs(Hs+Hp),'b-',linewidth=2,label='Hs+Hp numerical')
  plt.plot(x,np.abs(Hsp_analytic),'g--',linewidth=2,label='Hs+Hp analytic')
  plt.yscale('log')
  #plt.ylim((1.e-45,1.e-35))
  #plt.plot(x,Js,label='Js')
  #plt.plot(x,Jscheck,'o',label='Jscheck')
  #plt.plot(x,Hp,label='Hp')
  #plt.plot(x,Hs,label='Hs')
  plt.legend(loc='best')
  plt.xlabel('$x$')
  plt.ylabel('$H$')
  plt.show()


##########################################################################################################

def get_homo_soln_slow():
  global n,Hsp,kx,radius,s,ds,sigma,phix,phix_full

  b=np.zeros(n,dtype=np.cdouble)
  b=np.sqrt(3.0)*Hsp
  M = np.zeros((n,n),dtype=np.cdouble)
  for i in range(n):
    for j in range(n):
      z = np.abs(kx*radius*s[j])
      i0=spherical_in(0,z,derivative=False)
      di0=spherical_in(0,z,derivative=True)
      rat = di0/i0
      M[i,j]=(ds/2.0/np.pi)*np.exp(1j*s[j]*sigma[i])*(1.0+np.abs(s[j])*rat/np.sqrt(3.0)/phix[i])
  for i in range(n):                            # scale each row
    maxval=np.amax(np.absolute(M[i,:]))
    M[i,:]=M[i,:]/maxval
    b[i]=b[i]/maxval
  amp = linalg.solve(M,b,debug=True) # fourier coefficients
  check=np.dot(M,amp)

  Jh=np.zeros(Jp.size,dtype=np.cdouble)
  Hh=np.zeros(Hp.size,dtype=np.cdouble)
  for i in range(n):
    for j in range(n):
      z = np.abs(kx*radius*s[j])
      i0=spherical_in(0,z,derivative=False)
      di0=spherical_in(0,z,derivative=True)
      rat = di0/i0
      Jh[i]=Jh[i] + (ds/2.0/np.pi)*np.exp(1j*s[j]*sigma[i])*amp[j]
      Hh[i]=Hh[i] + (ds/2.0/np.pi)*np.exp(1j*s[j]*sigma[i])*amp[j]*(-np.abs(s[j])/(3.0*phix_full[i]))*rat
  return Jh, Hh

def get_homo_soln_fast():
  global n,Hsp,kx,radius,s,ds,sigma,phix,phix_full

  b=np.zeros(n,dtype=np.cdouble)
  b=np.sqrt(3.0)*Hsp
  z = np.abs(kx*radius*s)
  i0=spherical_in(0,z,derivative=False)
  di0=spherical_in(0,z,derivative=True)
  rat = di0/i0
  print("NUMBER OF POINTS, SIGMAMAX = ", n, np.max(sigma))
  # Construct matrix using array operations
  M=(ds/2.0/np.pi)*np.exp(1j*s[:]*sigma[:, None])*(1.0+np.abs(s[:])*rat/np.sqrt(3.0)/phix[:, None])

  # Divide each row by the max value in that row
  maxvals = np.amax(np.absolute(M), axis=1)
  M = M/maxvals[:, None]
  b = b/maxvals
  
  print('Solving matrix equation...\n')
  amp = linalg.solve(M,b,debug=True) # fourier coefficients
  check=np.dot(M,amp)

  Jh = np.sum((ds/2.0/np.pi)*np.exp(1j*s[:]*sigma[:, None])*amp[:], axis=1)
  Hh = np.sum((ds/2.0/np.pi)*np.exp(1j*s[:]*sigma[:, None])*amp[:]*(-np.abs(s[:])/(3.0*phix_full[:, None]))*rat[:], axis=1)
  return Jh, Hh

def test_full_solution_linear(filename):
  global x,sigma,tau0,Hsp,Hh,Jh,a
  x0=2*(a*tau0)**0.333
  plt.figure()
  plt.plot(x,Hp,'-.',linewidth=2,label='Hp')
  plt.plot(x,Hsp,':',linewidth=2,label=r'$H_0$')
  plt.plot(x,np.real(Hh),'-.',linewidth=2,label=r'$H_{\rm bc}$')
  plt.plot(x,Hsp+np.real(Hh),'-',linewidth=2,label=r'$H_0+H_{\rm bc}$')
  plt.plot(x,np.real(Jh)/np.sqrt(3.0),'--',linewidth=2,label=r'$J_{\rm bc}/\sqrt{3}$')
  plt.xlim((-x0,x0))
  plt.legend(loc='best')
  #plt.legend(loc='upper right')
  #plt.yscale('log')
  plt.xlabel('$x$',fontsize=20)
  plt.ylabel('$J,H$',fontsize=20)
  #plt.show()
  plt.savefig(filename)
  plt.close()

def test_full_solution_log(filename):
  global x,sigma,tau0,Hsp,Hh,Jh,a
  x0=2*(a*tau0)**0.333
  plt.figure()
  ymax=np.amax(np.abs(Hsp+np.real(Hh)))
  ymin=ymax*1.e-5
  plt.plot(x,np.abs(Hp),'-.',linewidth=2,label='Hp')
  plt.plot(x,np.abs(Hsp),':',linewidth=2,label=r'$H_{\rm 0}$')
  plt.plot(x,np.abs(np.real(Hh)),'-.',linewidth=2,label=r'$H_{\rm bc}$')
  plt.plot(x,np.abs(Hsp+np.real(Hh)),'-',linewidth=2,label=r'$H_{\rm 0}+H_{\rm bc}$')
  plt.plot(x,np.abs(np.real(Jh)/np.sqrt(3.0)),'--',linewidth=2,label=r'$J_{\rm bc}/\sqrt{3}$')
  plt.xlim((-x0,x0))
  plt.ylim((ymin,ymax))
  plt.legend(loc='best')
  #plt.legend(loc='upper right')
  plt.yscale('log')
  plt.xlabel('$x$',fontsize=20)
  plt.ylabel('$J,H$',fontsize=20)
  #plt.show()
  plt.savefig(filename)
  plt.close()

##########################################################################################################

def course_to_fine():
  global xmax

  nfine = 4097
  xfine=np.zeros(nfine)
  dxfine=2.0*xmax*(1.0-1.e-6)/(nfine-1.0)
  for i in range(nfine):
    xfine[i]=-xmax*(1.0-1.e-6)+dxfine*i
  f = interpolate.interp1d(x,Jp)
  Jpfine = f(xfine)
  plt.plot(x,Jp,'-')
  plt.plot(xfine,Jpfine)
  plt.show()

##########################################################################################################


def ftsoln_wrapper(tau0_in,xi_in,temp_in,radius_in,L_in):
  global n
  global tau0,sigmai,temp,radius,L
  global vth,delta,a,sigma0,numden,kx,beta
  global x,sigma,dsigma
  global phix, phix_full
  global s,ds
  global Jp,Hp
  global Js,Jscheck,Hs
  global Hsp_analytic,Hsp

  n=2**12 + 1 # 1025 # 2049 # 513 # 257 l# 2049 # 1025   # number of points used in solution

  tau0=tau0_in
  #sigmai=sigmai_in
  temp=temp_in
  radius=radius_in
  L=L_in

  init()
  sigmai = beta*xi_in**3/a # get_sigma(xi_in)
  norm = 4.0*np.pi*radius_in**2.*delta*4.0*np.pi/L_in
  get_arrays_uniform_in_sigma_slow()
#  get_arrays_uniform_in_x()
  #get_arrays_uniform_in_sigma_fast()
  phix = voigtx(a, x)
  phix_full = voigtx_full(a, x)
  get_s()
#  get_s_nonuniform()
  particular_solution(radius)
  #surface_solution_numerical()
  surface_solution_analytic()
  #H_surf_plus_particular_soln()
  Hsp=Hsp_analytic	# fast and more accurate
  Jsp=np.zeros(n)
  
  Jh, Hh = get_homo_soln_fast()

  #filename = "JH_" + str(tau0) + "_tau0_" + str(sigmai) + "_sigmai_linear.pdf"
  #test_full_solution_linear(filename)
  #filename = "JH_" + str(tau0) + "_tau0_" + str(sigmai) + "_sigmai_log.pdf"
  #test_full_solution_log(filename)

  return x,sigma,Jp,Hp,Jsp,Hsp,np.real(Jh),np.real(Hh)


##########################################################################################################

def main():

  temp = 1.e4
  radius = 1.e11
  L = 1.0

  tau0=1.e7
  xi=0.0
  ftsoln_wrapper(tau0,xi,temp,radius,L)

  #for i in range(4,11):
  #  tau0=10.0**i
  #  for j in range(0,4):
  #    sigmai=tau0*j/2.0
  #    ftsoln_wrapper(tau0,xi,temp,radius,L)
  
if __name__ == "__main__":
  main()
