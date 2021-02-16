
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from constants import fundconst,lymanalpha
from efunctions import parameters
import pdb

# max number of solutions at each n
nsolnmax=30                                          # maximum number of solutions for each n.

fc=fundconst()
la=lymanalpha()

def get_Pnm(ssoln,sigma,Jsoln,p,sigma_bounds=None):

  if sigma_bounds is not None:
    sig_min, sig_max = sigma_bounds
    sigma_inds = np.arange(len(sigma))[np.logical_and(sigma <= sig_max, sigma >= sig_min)]
    sigma_eval = sigma[sigma_inds]
    dsigma=np.diff(sigma_eval)
  else:
    sigma_inds = np.arange(len(sigma))
    dsigma=np.diff(sigma)

  Pnmsoln=np.zeros((p.nmax,nsolnmax))
  for k in sigma_inds[1:]:
    for n in range(1,p.nmax+1):
      for i in range(nsolnmax):
  #      Pnmsoln[n-1,i] = np.sqrt(1.5) * p.Delta**2 * (16.0*np.pi**2*p.radius/3.0/p.k/p.energy) \
  #                     * (-1.0)**(n+1) * np.sum(Jsoln[n-1,i,:])*dsigma
        Pnmsoln[n-1,i] = np.sqrt(1.5) * p.Delta**2 * (16.0*np.pi**2*p.radius/3.0/p.k/p.energy) \
                       * (-1.0)**(n+1) * np.sum(Jsoln[n-1,i,k])*dsigma[k-1]

  filename = "damping_times.data"
  fname=open(filename,'w')
  fname.write('%5s\t%5s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % ('n','m','s(Hz)','t(s)','Pnm','-Pnm/snm','cumul prob') )
  totalprob=0.0
  for n in range(1,p.nmax+1):
    for j in range(nsolnmax):
      if ssoln[n-1,j]==0.0:
        continue
      totalprob=totalprob - Pnmsoln[n-1,j]/ssoln[n-1,j]
      fname.write('%5d\t%5d\t%10.3e\t%10.3e\t%10.3e\t%10.3e\t%10.3e\n' % (n,j,ssoln[n-1,j],-1.0/ssoln[n-1,j],Pnmsoln[n-1,j],-Pnmsoln[n-1,j]/ssoln[n-1,j],totalprob) )
      #print("n,m,snm,- Pnm/ssm,cumulative_prob=",n,j,ssoln[n,j],- Pnmsoln[n-1,j]/ssoln[n,j],totalprob)
  fname.close()

  m=np.arange(0,nsolnmax)

  plt.figure()
  for n in range(1,p.nmax+1):
    plt.plot(m,-1.0/ssoln[n-1,:],label=str(n))
  plt.xlabel('mode number')
  plt.ylabel('decay time(s)')
  plt.legend(loc='best')
  plt.savefig('t_vs_m.pdf')
  plt.close()

  plt.figure()
  for n in range(1,p.nmax+1):
    plt.plot(m,Pnmsoln[n-1,:],label=str(n))
  plt.xlabel('mode number')
  plt.ylabel(r'$P_{nm}(s^{-1})$')
  plt.legend(loc='best')
  plt.savefig('Pnm_vs_m.pdf')
  plt.close()

  plt.figure()
  for n in range(1,p.nmax+1):
    plt.plot(m,-Pnmsoln[n-1,:]/ssoln[n-1,:],label=str(n))
  plt.xlabel('mode number')
  plt.ylabel(r'$-P_{nm}/s_{nm}$')
  plt.legend(loc='best')
  plt.savefig('Pnm_over_ssm_vs_m.pdf')
  plt.close()

  return Pnmsoln

def wait_time_vs_time(ssoln,Pnmsoln,times,p,intervals):

  tlc = p.radius/fc.clight

  plt.figure()

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2):
      for m in range(0,1):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,0)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2):
      for m in range(0,2):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,1)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2): 
      for m in range(0,3): 
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,2)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2):
      for m in range(0,4):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,4)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2):
      for m in range(0,5):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,5)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2):
      for m in range(0,6):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,6)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,2):
      for m in range(0,20):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(1,20)')

  plt.legend(loc='best')
  plt.yscale('log')
  plt.xlabel(r'$ct/R$',fontsize=15)
  plt.ylabel('$(R/c)\, P(t)$',fontsize=15)
  plt.title('n=1 and increasing m')
  #plt.show()
  plt.savefig('waittime_vs_time_n=1.pdf')
  plt.close
    

  plt.figure()

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,3):
      for m in range(0,1):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(2,0)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,3):
      for m in range(0,2):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(2,1)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,3):
      for m in range(0,3):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(2,2)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,3):
      for m in range(0,4):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(2,3)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,3):
      for m in range(0,5):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(2,4)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,3):
      for m in range(0,20):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(2,20)')

  plt.legend(loc='best')
  plt.yscale('log')
  plt.xlabel(r'$ct/R$',fontsize=15)
  plt.ylabel('$(R/c)\, P(t)$',fontsize=15)
  plt.title('all n=1-2 with increasing m')
  #plt.show()
  plt.savefig('waittime_vs_time_n=2.pdf')
  plt.close()

  plt.figure()

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,4):
      for m in range(0,1):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(3,0)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,4):
      for m in range(0,2):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(3,1)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,4):
      for m in range(0,3):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(3,2)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,4):
      for m in range(0,4):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(3,3)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,4):
      for m in range(0,5):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(3,4)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,4):
      for m in range(0,20):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(3,20)')

  plt.legend(loc='best')
  plt.yscale('log')
  plt.xlabel(r'$ct/R$',fontsize=15)
  plt.ylabel('$(R/c)\, P(t)$',fontsize=15)
  plt.title('all n=1-3 with increasing m')
  #plt.show()
  plt.savefig('waittime_vs_time_n=3.pdf')
  plt.close()

  plt.figure()
  '''
  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,p.nmax+1):
      for m in range(0,1):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(12,0)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,p.nmax+1):
      for m in range(0,2):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(12,1)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,p.nmax+1):
      for m in range(0,3):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(12,2)')

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,p.nmax+1):
      for m in range(0,4):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(12,3)')


  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,5+1):
      for m in range(0,30):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,'--',label='(5,30)', alpha=0.5)

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,6+1):
      for m in range(0,30):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,'--',label='(6,30)', alpha=0.5)
  '''

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,6+1):
      for m in range(0,30):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P,label='(5,30)', alpha=0.5)

  P = 0.0*times
  for i in range(times.size):
    t=times[i]
    for n in range(1,7+1):
      for m in range(0,30):
        P[i]=P[i] + Pnmsoln[n-1,m]*np.exp(ssoln[n-1,m]*t)
  plt.plot(times/tlc,tlc*P, '--', label='(6,30)', alpha=0.5)

  plt.legend(loc='best')
  plt.yscale('log')
  plt.xlabel(r'$ct/R$',fontsize=15)
  plt.ylabel('$(R/c)\, P(t)$',fontsize=15)
  plt.title('all n=1-6 with increasing m')
  plt.show()
  plt.savefig('waittime_vs_time_n=6.pdf')
  plt.close()

  return P


def dEdnudt(t,sigma,ssoln,Jsoln,p):
  # unnormalized wait time distribution for each separate frequency
  phi = line_profile(sigma,p)
  prefactor = 16.0*np.pi**2*p.radius/(3.0*p.k*phi*p.energy)
  wait_time = np.zeros((t.size,sigma.size))
  for i in range(t.size):
    for n in range(1,p.nmax+1):
      for j in range(nsolnmax):
        if ssoln[n,j]==0.0:
          continue
        wait_time[i,:] = wait_time[i,:] + prefactor * (-1.0)**(n+1) * Jsoln[n,j,:] * np.exp(ssoln[n,j]*t[i])
  return wait_time



def main():

  array = np.load('./eigenmode_data_xinit0.0_tau1e8_nmax12_nsolnmax30.npy',\
                  allow_pickle=True, fix_imports=True, )
  energy = array[0]
  temp = array[1]
  tau0 = array[2]
  radius = array[3]
  alpha_abs = array[4]
  prob_dest = array[5]
  xsource = array[6]
  nmax = array[7]
  nsigma = array[8]
  nomega = array[9]
  tdiff = array[10]
  sigma = array[11]
  ssoln = array[12]
  Jsoln = array[13]
  p = parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)

  Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)
  times = p.radius/fc.clight * np.arange(0.1,140.0,0.1)
  wait_time_dist = wait_time_vs_time(ssoln,Pnmsoln,times,p)

  print('Optical Depth =', tau0)
  print('Peak at ct/R =', fc.clight/p.radius * times[np.argmax(wait_time_dist)])
  print('t_c =', 1/(-ssoln[0][0]))

if __name__ == "__main__":
  main()
