import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from constants import fundconst,lymanalpha
from efunctions import line_profile
from parameters import Parameters
from mc_visual import fit_mc_exp, mc_wait_time
from scipy.interpolate import interp1d
import pdb
import matplotlib
matplotlib.rcParams['text.usetex'] = True

# max number of solutions at each n
nsolnmax=100
fc=fundconst()
la=lymanalpha()

def midpoint_diff(t):
  midpoints = 0.5*(t[1:]+t[:-1])
  dt = np.diff(midpoints)
  dt = np.insert(dt, 0, midpoints[1] - midpoints[0])
  dt = np.append(dt, midpoints[-1] - midpoints[-2])
  return dt

def get_Pnm(ssoln,sigma,Jsoln,p):
  dsigma = midpoint_diff(sigma)
  Pnmsoln=np.zeros((p.nmax,nsolnmax,len(dsigma)))
  for k in range(len(dsigma)):
    for n in range(1,p.nmax+1):
      for i in range(nsolnmax): ### EQ 110 --- don't make as a function of frequency. Coefficients of n and m already integrated over sigma
        Pnmsoln[n-1,i,k] = np.sqrt(1.5) * 16.0*np.pi**2 * p.radius * p.Delta\
                           / (3.0 * p.k * p.energy) * (-1.0)**(n) / ssoln[n-1, i]\
                           * Jsoln[n-1,i,k] * dsigma[k]

  filename = "./data/damping_times.data"
  fname=open(filename,'w')
  fname.write('%5s\t%5s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % ('n','m','s(Hz)','t(s)','Pnm','-Pnm/snm','cumul prob') )
  totalprob=0.0
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
  plt.savefig('./plots/t_vs_m.pdf')
  plt.close()

  plt.figure()
  for n in range(1,p.nmax+1):
    plt.plot(m,np.sum(Pnmsoln[n-1,:,:], axis=1),label=str(n))
  plt.xlabel('mode number')
  plt.ylabel(r'$P_{nm}(s^{-1})$')
  plt.legend(loc='best')
  plt.savefig('./plots/Pnm_vs_m.pdf')
  plt.close()

  plt.figure()
  for n in range(1,p.nmax+1):
    plt.plot(m,-np.sum(Pnmsoln[n-1,:,:], axis=1)/ssoln[n-1,:],label=str(n))
  plt.xlabel('mode number')
  plt.ylabel(r'$-P_{nm}/s_{nm}$')
  plt.legend(loc='best')
  plt.savefig('./plots/Pnm_over_ssm_vs_m.pdf')
  plt.close()

  return Pnmsoln


def wait_time_vs_time(ssoln,Pnmsoln,times,p):

  m_upper_lims = [1, 2, 3, 4, 5, 6, 20]
  n_upper_lims = [1, 2, 3, p.nmax]
  titles = ['n=1 and increasing m', 'all n=1-2 with increasing m', 
            'all n=1-3 with increasing m', 'all n=1-6 with increasing m']

  for k, nmax in enumerate(n_upper_lims):
      plt.figure()
      for mmax in m_upper_lims:
          wait_time_line(plt, ssoln, Pnmsoln, times, p, nmax=nmax, mmax=mmax, alpha=0.5)
      plt.legend(loc='best')
      plt.yscale('log')
      plt.xlabel(r'$ct/R$',fontsize=15)
      plt.ylabel('$(R/c)\, P(t)$',fontsize=15)
      plt.title(titles[k])
      plt.savefig('./plots/waittime_vs_time_n={}.pdf'.format(nmax))
      plt.close()


def wait_time_line(ax, sigma, ssoln, Jsoln, Pnmsoln, times, p, nmax=6, mmax=20,alpha=0.5,label=None):
    '''
    Produces an analytic line for a wait time distribution using a sum over
    eigenfunctions. The line is normalized by integrating over the distribution
    from the right while all values are positive.
    '''


    tlc = p.radius/fc.clight

    # TODO; Eq 113 is for a single frequency, not a range of frequencies

    P = np.zeros((len(times), len(sigma)))
#    denom = np.zeros((len(times), len(sigma)))

    for i, t in enumerate(times):
        for n in range(1, nmax+1):
            for m in range(0, mmax): 

                ### EQ 113
                #P[i] += - (-1)**n * Jsoln[n-1, m, :] * np.exp(ssoln[n-1, m] * t)
                #denom += (-1)**n / ssoln[n-1, m] * Jsoln[n-1, m, :]

                ### EQ 112
                P[i] += - np.sum(Pnmsoln[n-1, m, :]) * ssoln[n-1, m] * np.exp(ssoln[n-1, m] * t) * p.Delta

    if label is None:
        label = '({},{})'.format(nmax, mmax)
    line = ax.plot(times/tlc,P*tlc,label=label, alpha=alpha)
    
    return line


def eigenmode_visualization(sigma,Jsoln,p):
    '''
    Visualise the eigenmodes for a particular set of n and m.
    '''

    from scipy.interpolate import make_interp_spline, BSpline
    import matplotlib.pylab as pl
    colors = pl.cm.viridis_r(np.linspace(0,1,p.nmax*nsolnmax))

    fig, ax = plt.subplots()
    for n in reversed(range(1, p.nmax+1)):
        for m in reversed(range(nsolnmax)):
            q = nsolnmax*(n-1)+m
            fine_sigma = np.linspace(-1.2e8, 1.2e8, len(sigma)*100)
            Jinterp = make_interp_spline(sigma, Jsoln[n-1, m, :], k=3)

#            fine_sigma = np.concatenate([fine_sigma, np.flip(fine_sigma)[1:]])
#            Jinterp = np.concatenate([Jinterp, np.flip(Jinterp)[1:]])

            ax.plot(fine_sigma, Jinterp(fine_sigma), label='({},{})'.format(n, m), alpha=1-q/(nsolnmax*p.nmax)/2, lw=1.5, color=colors[q]) #0.75 - (n/p.nmax + (m+1)/nsolnmax)/2/2
    ax.axis('off')
    plt.xlim(-1.2e8, 1.2e8)
    plt.ylim(-3.65e-35, 6.49e-35)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    plt.savefig('eigenfunctions_xs={:04.1f}.png'.format(p.xsource), )
    plt.show()


def wait_time_freq_dependence(ssoln,Jsoln,Pnmsoln,times,p,bounds,):
    '''
    Produce a wait time distribution with all spatial and frequency eigenmodes,
    split into ranges of frequency. The ranges are constructed between the 
    values of the list argument 'bounds'.
    '''

    fig1, ax1 = plt.subplots(1, 1, figsize=(4.8,5.4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4.8,5.4))

    # Fluence: integral of Pnm with respect to time
    sigma = np.array(sorted(np.concatenate(list(p.sigma_master.values()))))
    spec = np.zeros(np.shape(sigma))
    phi = line_profile(sigma, p)
    for n in range(1, p.nmax+1):   ### EQ 111
        for m in range(nsolnmax):
            spec += (
                    16. * np.pi**2 * p.radius * p.Delta
                    / (3.0 * p.k * p.energy * phi) * (-1)**n
                    * Jsoln[n-1, m, :] / ssoln[n-1, m]
                    )

    sigma_to_x = np.cbrt(sigma/p.c1)

    # Make an array of uniformly spaced x-values symmetric about 0 (min, max, npoints)
#    xuniform_l = np.linspace(np.min(sigma_to_x), -0.1, int(len(sigma_to_x)/2))
#    xuniform_r = np.linspace(0.1, np.max(sigma_to_x), int(len(sigma_to_x)/2))
#    xuniform = np.concatenate([xuniform_l, xuniform_r])

    # Find sigma at each x-value
#    sigma_xuniform = (p.c1) * xuniform**3.

    # Calculate line profile at all the x points needed
#    phi_xuniform = line_profile(xuniform**3 * p.c1, p)

    # Interpolate solutions from original points
#    spec_interp = interp1d(sigma_to_x, np.log(spec * phi)) # TODO: Interpolate over log space instead

    # Apply interpolation to uniformly distributed x values, divide by line
    # profile at those x positions
#    spec_xuniform = np.exp(spec_interp(xuniform)) / phi_xuniform

    ax2.plot(sigma_to_x, spec, 'k-', lw=0.5)

    mc_dir = '/home/connor/Documents/lya_analytic/data/1m_tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'

    for i in range(len(bounds)-1):

        freq_min = bounds[i]
        freq_max = bounds[i+1]
        xbounds = np.around(np.cbrt(np.array([freq_min, freq_max])/p.c1))

        # Get probability in between frequency bounds
        mask = np.logical_and(np.abs(sigma) >= freq_min, np.abs(sigma) <= freq_max)
        Pnm_masked = Pnmsoln[:, :, mask]
        J_masked = Jsoln[:, :, mask]

        # Load Monte Carlo data, plot spectrum scatter points and normalize
        tdata, xdata, poly = mc_wait_time(mc_dir, bounds=xbounds)
        t, y_t = tdata
        x, y_x = xdata
        ax2.scatter(x, y_x, color='k', s=1)
        dt = midpoint_diff(t)
        y = y_t/np.sum(y_t*dt)

        # Plot analytic escape time distribution
        label = 'Analytic $n_{{max}}={}$ $m_{{max}}={}$'.format(p.nmax,nsolnmax)
        line = wait_time_line(ax1, sigma[mask], ssoln, J_masked, Pnm_masked, times, p, nmax=p.nmax, mmax=nsolnmax, alpha=0.5, label=label)

        # Shade spectrum inbetween frequency bounds
        ax2.fill_between(np.cbrt(np.linspace(freq_min, freq_max)/p.c1), 10*np.max(spec), facecolor=line[-1].get_color(), alpha=0.5, label="${} < |x| < {}$".format(*xbounds))
        ax2.fill_between(-np.cbrt(np.linspace(freq_min, freq_max)/p.c1), 10*np.max(spec), facecolor=line[-1].get_color(), alpha=0.5)

        # Plot wait time scatter points
        ax1.scatter(t, y, facecolor=line[-1].get_color(), s=1, marker='^', label='MC (${} < |x| < {}$)'.format(*xbounds))

        # Add exponential fit lines
#        exp_fit = np.exp(poly[1]) * np.exp(poly[0]*times)
#        ax1.plot(times, exp_fit, '--', c=line[-1].get_color(), alpha=0.75, lw=1, label='${:.1f} e^{{-t/{:.1f}}}$'.format(np.exp(poly[1]), -1/poly[0]))
        print("mc exp fit: ", np.exp(poly[1]), -1/poly[0])


    # Set limits and labels of spectrum plot
    xlim = np.max(np.cbrt(np.array(bounds)/p.c1))
    ax2.set_xlim(-xlim, xlim)
    ax2.set_ylim(np.min(spec[spec>0]), np.max(spec))
#    ax2.set_yscale('log')
    ax2.set_ylabel(r"$E_\nu$")
    ax2.set_xlabel('$x$')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0+0.01*box.height,
                     box.width, 0.99*box.height])

    # Set limits and labels of escape time plot
    ax1.set_ylim(np.min(y[y>0]), 2*np.max(y))
    ax1.set_xlim(-5, max(t)+5)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$ct/R$',fontsize=15)
    ax1.set_ylabel('$(R/c)\, P(t)$',fontsize=15)

    # Formatting & output
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
    plt.savefig('./plots/waittime_vs_time_freq.pdf')
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
  filenames = [
              './data/eigenmode_data_xinit0_tau1e7_n6_m100_xuniform_masteronly.npy',
              #'./data/eigenmode_data_xinit0_tau1e7_n6_m20_xuniform_masteronly.npy',
              #'./data/eigenmode_data_xinit0_tau1e7_n6_m40.npy',
              ]
  for filename in filenames:
      array = np.load(filename, allow_pickle=True, fix_imports=True, )
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
      p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)

      Pnmsoln = get_Pnm(ssoln,sigma,Jsoln,p)
      times = p.radius/fc.clight * np.arange(0.1,140.0,0.1)
#      wait_time_dist = wait_time_vs_time(ssoln,Pnmsoln,times,p) # Uncomment to produce Phil's plots

    #  peak = 60
    #  x_bounds = np.array([0, peak/2, peak, 3*peak/2, 160])
      x_bounds = np.array([0, 30])
      sigma_bounds = p.c1 * x_bounds**3.

      wait_time_freq_dependence(ssoln, Jsoln, Pnmsoln, times, p, sigma_bounds)

plt.show()
#  print('Optical Depth =', tau0)
#  print('Peak at ct/R =', fc.clight/p.radius * times[np.argmax(wait_time_dist)])
#  print('t_c =', 1/(-ssoln[0][0]))

if __name__ == "__main__":
    main()
