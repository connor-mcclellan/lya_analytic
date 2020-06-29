import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np
import scipy
from scipy.stats import lognorm
from prob_ct_tau import prob_ct_tau
from glob import glob


def plot_fit(tau):
    norm = 1./tau * (tau + 2./3.)**2. * 3./np.pi**2.
    print(tau)
    print(norm)

    # Load photon data, take care of float errors in r
    data = np.loadtxt('./test_voigt_at_linecenter.dat')
    data[:, 0] = np.round(data[:, 0], 5)

    # Set up figure, make initial histogram, normalize x and y
    fig, ax = plt.subplots(1, 1, dpi=180)
    n, bins, patches = ax.hist(data[:, 6], bins=50, color='k', histtype='step', density=True)
    bins = bins/norm
    n = n*norm

    # Calculate new bin positions, check normalization sum, clear old histogram
    bincenters = 0.5*(bins[1:]+bins[:-1])
    print(np.sum(n*np.diff(bins)))
    plt.cla()

    # Scatter plot of the new bin positions and normalized counts
    ax.scatter(bincenters, n, color='k', s=3)
    ax.set_xlabel('Distance')
    ax.set_ylabel('n (normalized)')
    ax.set_yscale('log')
    ax.set_title(r'Total Distance Traveled, $\tau = {}$, $n = 10^6$'.format(int(tau)))

    # Calculate probability density from Shane's series solution code
    prob = np.zeros(np.shape(bincenters))
    for i in range(len(bincenters)):
        prob[i] = prob_ct_tau(bincenters[i], tau)
    ax.plot(bincenters, prob, 'b--', label='Series Solution', alpha=0.5)

    # Fit a log normal distribution to the normalized data
    shape, loc, scale = lognorm.fit(data[:, 6]/norm, loc=1)
    pdf = lognorm.pdf(bincenters, shape, loc, scale)
    ax.plot(bincenters, pdf, 'r--', label='Log Normal', alpha=0.5)
    ax.text(0.05, 0.05, r'$\mu={:.4f}$, $\sigma={:.4f}$'.format(np.log(scale), shape), transform=ax.transAxes)
    # Save or show the plot
    plt.legend()
    plt.savefig('./fit_tau{}.pdf'.format(int(tau)))
    plt.close()
    return np.log(scale), shape

mu, sigma = plot_fit(10.)
print(mu, sigma)

######################
# Exponential fit
#bincenters = bincenters[n >= 3]
#n = n[n >= 3]

#coef, res, _, _, _ = np.polyfit(bincenters, np.log(n), 1, full=True)
#poly1d = np.poly1d(coef)
#ax.plot(bincenters, np.exp(poly1d(bincenters)), '--r', label='Exponential')
#print(coef, res)
