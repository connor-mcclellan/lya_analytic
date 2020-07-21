import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)

# Load photon data
data_dir = '/home/connor/Documents/999x/9999/lya_analytic/data/'\
           '1m_tau0_10000000.0_xinit_0.0_temp_10000.0_probabs_0.0/'
mu, x, time = np.load(data_dir + 'mu_x_time.npy')

# Define frequency chunks by distance from line center
freqs = np.array([0., 15., 20., 40.])
n_bins = 64

# Set up figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

# Create x histogram data
n_x, bins_x, _ = plt.hist(x, bins=n_bins, density=True)
bincenters_x = 0.5 * (bins_x[1:] + bins_x[:-1])
plt.cla()

# Empty lists that will contain data from each frequency range
bc = []
counts = []
nu_bounds = []
size = []

# Loop over frequency ranges
for i in range(len(freqs) - 1):

    # Define frequency minimum and frequency maximum
    nu_min = freqs[i]
    nu_max = freqs[i + 1]

    # Create a truth array for indices where x is in the frequency range
    mask = (np.abs(x) > nu_min) & (np.abs(x) < nu_max)
    t = time[mask]

    # Create time histogram data, normalized for this frequency batch
    n, bins, _ = plt.hist(
        t, bins=np.logspace(
            np.log10(
                min(t)), np.log10(
                max(t)), n_bins), density=True)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    plt.cla()

    # Package the data for this frequency range
    bc.append(bincenters)
    counts.append(n)
    nu_bounds.append([nu_min, nu_max])
    size.append(len(t))

# Loop over frequency segments and plot the data we packaged in the
# previous loop
for i in range(len(bc)):

    # Plot the line
    p = ax1.plot(
        bc[i],
        counts[i],
        '.',
        ms=3.,
        alpha=0.5,
        label="${} < |x| < {}$, $n={}$".format(
            nu_bounds[i][0],
            nu_bounds[i][1],
            size[i]))

    # Shade the frequency region, matching the color of the line
    xspace = np.linspace(freqs[i], freqs[i + 1], 100)
    neg_xspace = np.linspace(-freqs[i], -freqs[i + 1], 100)
    ax2.fill_between(xspace,
                     np.max(n_x),
                     facecolor=p[-1].get_color(),
                     alpha=0.5)
    ax2.fill_between(neg_xspace, np.max(
        n_x), facecolor=p[-1].get_color(), alpha=0.5)

# Left-hand plot aesthetics
ax1.legend()
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$P(t)$')
ax1.set_xlim((1, 150))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Escape Time Distribution')

# Right-hand plot aesthetics
ax2.plot(bincenters_x, n_x, 'k-', ms=3.)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$P(x)$')
ax2.set_title('Monte Carlo Spectrum')
# ax2.set_yscale('log')
plt.show()
