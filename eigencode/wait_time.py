
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from constants import fundconst,lymanalpha
from efunctions import parameters, line_profile
from scipy.interpolate import interp1d
import pdb

# max number of solutions at each n
nsolnmax=20                                          # maximum number of solutions for each n.

fc=fundconst()
la=lymanalpha()

def get_Pnm(ssoln,sigma,Jsoln,p):
  dsigma=np.diff(sigma, prepend=sigma[1]-sigma[0])
  Pnmsoln=np.zeros((p.nmax,nsolnmax,len(dsigma)))
  for k in range(len(dsigma)):
    for n in range(1,p.nmax+1):
      for i in range(nsolnmax):
        Pnmsoln[n-1,i,k] = np.sqrt(1.5) * p.Delta**2 * (16.0*np.pi**2*p.radius/3.0/p.k/p.energy) \
                           * (-1.0)**(n+1) * Jsoln[n-1,i,k] * dsigma[k]

  filename = "damping_times.data"
  fname=open(filename,'w')
  fname.write('%5s\t%5s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % ('n','m','s(Hz)','t(s)','Pnm','-Pnm/snm','cumul prob') )
  totalprob=0.0
  # Normalize probabilities here?
  for n in range(1,p.nmax+1):
    for j in range(nsolnmax):
      if ssoln[n-1,j]==0.0:
        continue
      totalprob=totalprob - np.sum(Pnmsoln[n-1,j,:])/ssoln[n-1,j]
      fname.write('%5d\t%5d\t%10.3e\t%10.3e\t%10.3e\t%10.3e\t%10.3e\n' % (n,j,ssoln[n-1,j],-1.0/ssoln[n-1,j],np.sum(Pnmsoln[n-1,j,:]),-np.sum(Pnmsoln[n-1,j,:])/ssoln[n-1,j],totalprob) )
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
    plt.plot(m,np.sum(Pnmsoln[n-1,:,:], axis=1),label=str(n))
  plt.xlabel('mode number')
  plt.ylabel(r'$P_{nm}(s^{-1})$')
  plt.legend(loc='best')
  plt.savefig('Pnm_vs_m.pdf')
  plt.close()

  plt.figure()
  for n in range(1,p.nmax+1):
    plt.plot(m,-np.sum(Pnmsoln[n-1,:,:], axis=1)/ssoln[n-1,:],label=str(n))
  plt.xlabel('mode number')
  plt.ylabel(r'$-P_{nm}/s_{nm}$')
  plt.legend(loc='best')
  plt.savefig('Pnm_over_ssm_vs_m.pdf')
  plt.close()

  return Pnmsoln


def wait_time_line(ax, ssoln, Pnmsoln, times, p, nmax=6, mmax=20,alpha=0.5,norm=None):
    tlc = p.radius/fc.clight
    P = np.zeros(np.shape(times))
    for i, t in enumerate(times):
        for n in range(1, nmax+1):
            for m in range(0, mmax):
                P[i] += np.sum(Pnmsoln[n-1,m,:]) * np.exp(ssoln[n-1,m] * t)
                # TODO: Normalize positive part of the spectrum to 1
    if norm is not None:
        idx = np.argmin(np.abs(times/tlc - norm[0]))
        P = norm[1] * P/(P[idx]*tlc)
    line = ax.plot(times/tlc,tlc*P,label='({},{})'.format(nmax, mmax), alpha=alpha)
    return line


def wait_time_vs_time(ssoln,Pnmsoln,times,p):

  m_upper_lims = [1, 2, 3, 4, 5, 6, 20]
  n_upper_lims = [1, 2, 3, p.nmax]
  titles = ['n=1 and increasing m', 'all n=1-2 with increasing m', 
            'all n=1-3 with increasing m', 'all n=1-6 with increasing m']

  for k, nmax in enumerate(n_upper_lims):
      plt.figure()
      for mmax in m_upper_lims:
          wait_time_line(plt, ssoln, Pnmsoln, times, p, nmax=nmax, mmax=mmax, alpha=alpha)
      plt.legend(loc='best')
      plt.yscale('log')
      plt.xlabel(r'$ct/R$',fontsize=15)
      plt.ylabel('$(R/c)\, P(t)$',fontsize=15)
      plt.title(titles[k])
      plt.savefig('waittime_vs_time_n={}.pdf'.format(nmax))
      plt.close()


def mc_wait_time(mc_dir, bounds, p):

        freq_min, freq_max = bounds # in x units

        plt.figure()
        mu, x, time = np.load(mc_dir + 'mu_x_time.npy')
        nbins=64
        n_x, bins_x, _ = plt.hist(x, bins=nbins, density=True)
        bincenters_x = 0.5 * (bins_x[1:] + bins_x[:-1])

        mask = np.logical_and(np.abs(x)>freq_min, np.abs(x)<freq_max)
        t = time[mask]
        try:
            n, bins, _ = plt.hist(t, bins=np.logspace(np.log10(min(t)), np.log10(max(t)), nbins), density=True)
            bincenters = 0.5*(bins[1:] + bins[:-1])
            plt.close()
            return bincenters, n
        except:
            pass


def wait_time_freq_dependence(ssoln,sigma,Jsoln,Pnmsoln,times,p,bounds,):
    '''
    Produce a wait time distribution with all spatial and frequency eigenmodes,
    split into ranges of frequency. The ranges are constructed between the 
    values of the list argument 'bounds'.
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Fluence: integral of Pnm with respect to time
    spec = np.zeros(np.shape(Pnmsoln)[-1])

    for n in range(1, 6+1):
        for m in range(nsolnmax):
            spec += (-1)**n * Pnmsoln[n-1, m, :]/ssoln[n-1, m]

    sigma_to_x = np.cbrt(sigma/p.c1)

    # Make an array of uniformly spaced x-values (min, max, npoints)
    xuniform = np.linspace(np.min(sigma_to_x), np.max(sigma_to_x), len(sigma_to_x))

    # Find sigma at each x-value
    sigma_xuniform = (p.c1) * xuniform**3.

    # Calculate line profile at all the x points needed
    phi = line_profile(sigma, p)
    phi_xuniform = line_profile(xuniform**3 * p.c1, p)

    # Interpolate solutions from original points
    spec_interp = interp1d(sigma_to_x, spec * phi)

    # Apply interpolation to uniformly distributed x values, divide by line
    # profile at those x positions
    spec_xuniform = spec_interp(xuniform) / phi_xuniform


    ax2.plot(xuniform, spec_xuniform, 'k-', lw=0.5)

    mc_dir = '/home/connor/Documents/lya_analytic/data/1m_tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'

    for i in range(len(bounds)-1):

        freq_min = bounds[i]
        freq_max = bounds[i+1]

        Pnm_masked = Pnmsoln[:, :, np.logical_and(np.abs(sigma) >= freq_min, np.abs(sigma) <= freq_max)]
#        line = wait_time_line(ax1, ssoln, Pnm_masked, times, p, nmax=p.nmax, mmax=nsolnmax, alpha=0.5)

        xbounds = np.cbrt(np.array([freq_min, freq_max])/p.c1)
        x, y, = mc_wait_time(mc_dir, xbounds, p)
        x_at_ymax, ymax = (x[np.argmax(y)+9], y[np.argmax(y)+9])
        line = wait_time_line(ax1, ssoln, Pnm_masked, times, p, nmax=6, mmax=20, alpha=0.5, norm=[x_at_ymax, ymax])
        ax2.fill_between(np.cbrt(np.linspace(freq_min, freq_max)/p.c1), 10*np.max(spec), facecolor=line[-1].get_color(), alpha=0.5)
        ax2.fill_between(-np.cbrt(np.linspace(freq_min, freq_max)/p.c1), 10*np.max(spec), facecolor=line[-1].get_color(), alpha=0.5)

        ax1.scatter(x, y, facecolor=line[-1].get_color(), s=1)

    xlim = np.max(np.cbrt(np.array(bounds)/p.c1))
    ax2.set_xlim(-xlim, xlim)
    ax2.set_ylim(np.min(spec), 10*np.max(spec))
    #ax2.set_yscale('log')
    ax2.set_title('Spectrum')
    ax2.set_ylabel("$|J(x)|$")
    ax2.set_xlabel('$x$')

    ax1.legend(loc='best')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$ct/R$',fontsize=15)
    ax1.set_ylabel('$(R/c)\, P(t)$',fontsize=15)
    ax1.set_title('Wait Time')
    plt.tight_layout()
    plt.show()
    plt.savefig('waittime_vs_time_freq.pdf')
    plt.close()   


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

  array = np.load('./eigenmode_data_xinit0.0_tau1e7.npy',\
                  allow_pickle=True, fix_imports=True, )
  energy = array[0]
  temp = array[1]
  tau0 = array[2]
  radius = array[3]
  alpha_abs = array[4]
  prob_dest = array[5]
  xsource = array[6]
  nmax = array[7]-1
  nsigma = array[8]
  nomega = array[9]
  tdiff = array[10]
  sigma = array[11]
  ssoln = array[12][1:]
  Jsoln = array[13][1:]
  p = parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)

  Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)
  times = p.radius/fc.clight * np.arange(0.1,140.0,0.1)
#  wait_time_dist = wait_time_vs_time(ssoln,Pnmsoln,times,p)

#  peak = 60
#  x_bounds = np.array([0, peak/2, peak, 3*peak/2, 160])
  x_bounds = np.array([0, 15, 30])
  sigma_bounds = p.c1 * x_bounds**3.

  wait_time_freq_dependence(ssoln, sigma, Jsoln, Pnmsoln, times, p, sigma_bounds)

#  print('Optical Depth =', tau0)
#  print('Peak at ct/R =', fc.clight/p.radius * times[np.argmax(wait_time_dist)])
#  print('t_c =', 1/(-ssoln[0][0]))

if __name__ == "__main__":
  main()
