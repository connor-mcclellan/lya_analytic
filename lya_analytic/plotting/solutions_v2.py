from util import read_bin, voigtx_fast, beta, Line
import astropy.constants as c
import ftsoln
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np
import math
from scipy.interpolate import interp1d
from astropy.utils.console import ProgressBar

def get_input_info(filename):
  f = open(filename, 'r')
  g=f.readlines()
  f.close()
  for i in g:
    for j in i.split():
      k = j.find("tau0=") 
      if k!=-1:
        tau0=j[k+5:]
      k = j.find("temp=")
      if k!=-1:
        temp=j[k+5:]
      k = j.find("x_init=")
      if k!=-1:
        x_init=j[k+7:]
      k = j.find("prob_abs=")
      if k!=-1:
        prob_abs=j[k+9:]
      k = j.find("rmax=")
      if k!=-1:
        rmax=j[k+5:]
  return np.float(tau0),np.float(temp),np.float(x_init),np.float(prob_abs),np.float(rmax)

def bin_x(x,n,mytitle,tau0,xinit,temp,radius,L,delta,a):
  count=np.zeros(n)
  x0=2.0*(a*tau0)**0.333
  # n bins, n+1 bin edges
  xmax=np.amax(x)
  xmin=np.amin(x)
  dx=(xmax-xmin)/n
  xe=np.zeros(n+1)
  for i in range(n+1):
   xe[i]=xmin + i*dx
  xc=np.zeros(n)
  for i in range(n):
   xc[i]=0.5*(xe[i]+xe[i+1])

  for xval in x:
    if xval<=xmin:
      j=0
    elif xval>=xmax:
      j=n-1
    else:
      j=np.rint(math.floor( (xval-xmin)/dx))
      j=int(j) # j=j.astype(int)
    if (xe[j+1]-xval)*(xval-xe[j]) < 0.0:
      print(j,(xe[j+1]-xval)*(xval-xe[j]))
    count[j]=count[j]+1.0
  err=np.sqrt(np.abs(1.0*count))
  count=count/x.size/dx
  err=err/x.size/dx

  x_ft,sigma_ft,Jp_ft,Hp_ft,Jsp_ft,Hsp_ft,Jh_ft,Hh_ft = ftsoln.ftsoln_wrapper(tau0,xinit,temp,radius,L)
  norm = 4.0*np.pi*radius**2*delta*4.0*np.pi/L
  print('norm: ', norm)
  print("check norm of data=",np.sum(count)*dx)

  ymax1=np.amax( (Hp_ft)*norm )
  ymax2=np.amax( (Hsp_ft)*norm )
  ymax3=np.amax( (Hh_ft)*norm )
  ymax4=np.amax( count )
  ymax=max([ymax1,ymax2,ymax3,ymax4]) * 1.1
  ymin=np.amin( Hh_ft*norm ) * 1.1

  # CM: Get values of Hsp_ft at x positions of monte carlo data points
  hsp_ft_func = interp1d(x_ft, Hsp_ft*norm)

  plt.figure()
  plt.plot(x_ft,(Hp_ft-Hsp_ft)*norm,label=r'$H_{\rm d} - H_0$')
  plt.plot(x_ft,Hh_ft*norm,label=r'$H_{\rm bc}$')
  plt.errorbar(xc,count-hsp_ft_func(xc),yerr=err,fmt='.',label="Monte Carlo - $H_0$")
  plt.title(mytitle)
  plt.legend(loc='best')
  plt.xlabel(r'$x$',fontsize=15)
  plt.ylabel(r'$P(x)$',fontsize=15)
  plt.savefig("x_pdf_subtracted.pdf",format='pdf')
  plt.close()

  plt.figure()
  plt.plot(x_ft,Hp_ft*norm,label=r'$H_{\rm d}$')
  plt.plot(x_ft,Hsp_ft*norm,label=r'$H_0$')
  plt.plot(x_ft,Hh_ft*norm,label=r'$H_{\rm bc}$')
  plt.plot(x_ft,(Hsp_ft+Hh_ft)*norm,label=r'$H_0+H_{\rm bc}$')
  plt.errorbar(xc,count,yerr=err,fmt='.',label="Monte Carlo")
  plt.xlim((-x0,x0))
  plt.ylim((ymin,ymax))
  plt.title(mytitle)
  plt.legend(loc='best')
  plt.xlabel(r'$x$',fontsize=15)
  plt.ylabel(r'$P(x)$',fontsize=15)
  #plt.show()
  plt.savefig("x_pdf.pdf",format='pdf')
  plt.close()

  ymax1=np.amax( (Hp_ft)*norm )
  ymax2=np.amax( (Hsp_ft)*norm )
  ymax3=np.amax( (Hh_ft)*norm )
  ymax4=np.amax( count )
  ymax=max([ymax1,ymax2,ymax3,ymax4]) * 1.1
  ymin=ymax*1.e-3

  plt.figure()
  plt.plot(x_ft,np.abs(Hp_ft*norm),label=r'$H_{\rm d}$')
  plt.plot(x_ft,np.abs(Hsp_ft*norm),label=r'$H_0$')
  plt.plot(x_ft,np.abs(Hh_ft*norm),label=r'$H_{\rm bc}$')
  plt.plot(x_ft,np.abs((Hsp_ft+Hh_ft)*norm),label=r'$H_0+H_{\rm bc}$')
  plt.errorbar(xc,count,yerr=err,fmt='.',label="Monte Carlo")
  plt.xlim((-x0,x0))
  plt.ylim((ymin,ymax))
  plt.yscale('log')
  plt.title(mytitle)
  plt.legend(loc='best')
  plt.xlabel(r'$x$',fontsize=15)
  plt.ylabel(r'$P(x)$',fontsize=15)
  #plt.show()
  plt.savefig("x_pdf_log.pdf",format='pdf')
  plt.close()

def bin_time(t,n,mytitle):

  count=np.zeros(n)

  tmax=np.amax(t)
  tmin=np.amin(t)
  dt=(tmax-tmin)/n
  te=np.zeros(n+1)
  for i in range(n+1):
   te[i]=tmin + i*dt
  tc=np.zeros(n)
  for i in range(n):
   tc[i]=0.5*(te[i]+te[i+1])

  xbar = np.average( np.log10(t) )
  xsqbar = np.average( (np.log10(t))**2 )
  xsigma = np.sqrt( xsqbar-xbar**2 )
  theory = 1.0/(np.sqrt(2.0*np.pi)*xsigma) * np.exp(-(np.log10(tc)-xbar)**2/(2.0*xsigma**2))

  norm = 0.0
  for i in range(n):
   dlog10t=np.log10(te[i+1]/te[i])
   norm = norm + dlog10t*theory[i]

  print('Binning time...')
  pb = ProgressBar(len(t))
  for tval in t:
    if tval<=tmin:
      j=0
    elif tval>=tmax:
      j=n-1
    else:
      j=math.floor( (tval-tmin)/dt)
      j=int(j) # j=j.astype(int)
    if (te[j+1]-tval)*(tval-te[j]) < 0.0:
      print(j,(te[j+1]-tval)*(tval-te[j]))
#      quit()
      continue
    count[j]=count[j]+1.0
    pb.update()
  count=count/t.size/dt

  plt.figure()
  plt.title(mytitle)
  plt.plot(tc,2.3*tc*count,".",label="data")
  plt.plot(tc,theory,label="theory")
  plt.yscale('log')
  plt.xscale('log')
  plt.legend(loc='best')
  plt.xlabel(r'$t$',fontsize=15)
  plt.ylabel(r'$2.3tP(t)$',fontsize=15)
  plt.savefig("time_pdf.pdf",format='pdf')
  plt.close()


if __name__ == '__main__':
  data_dir = '/home/connor/Documents/999x/9999/lya_analytic/phils_code/tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'

  lya = Line(1215.6701, 0.4164, 6.265e8)
  L = 1.0
  tau0,temp,xinit,prob_abs,radius = get_input_info(data_dir+'input')
  vth = np.sqrt(2.0*c.k_B.cgs.value*temp/c.m_p.cgs.value)
  delta = lya.nu0 * vth / c.c.cgs.value
  a=lya.gamma/(4.0*np.pi*delta)

  mytitle = r'$\tau_0=$' + str(tau0) + r', $x_{\rm init}=$' + str(xinit) \
            + r', $ T=$' + str(temp) + r', $p_{\rm abs}=$' + str(prob_abs)

  mu, x, time = read_bin(data_dir)
  bin_x(x,64,mytitle,tau0,xinit,temp,radius,L,delta,a)
  bin_time(time,64,mytitle)

