import numpy as np
import matplotlib.pyplot as plt


def fit_mc_exp(n, t, buff=0):
    peak_index = np.argmax(n)+buff
    falloff_n = n[n>0][peak_index:]
    falloff_t = t[n>0][peak_index:]

    poly = np.polyfit(falloff_t, np.log(falloff_n), 1)
    return poly


def mc_wait_time(mc_dir, bounds=None):

    if bounds is not None:
        freq_min, freq_max = bounds # in x units
    else:
        freq_min, freq_max = (-np.inf, np.inf)

    plt.figure()
    mu, x, time = np.load(mc_dir + 'mu_x_time.npy')
    nbins=64
    n_x, bins_x, _ = plt.hist(x, bins=nbins, density=True)
    bincenters_x = 0.5 * (bins_x[1:] + bins_x[:-1])

    mask = np.logical_and(np.abs(x)>freq_min, np.abs(x)<freq_max)
    t = time[mask]

    n, bins, _ = plt.hist(t, bins=np.logspace(np.log10(min(t)), np.log10(max(t)), nbins), density=True)
    bincenters = 0.5*(bins[1:] + bins[:-1])
    plt.close()
    poly = fit_mc_exp(n, bincenters)
    return (bincenters, n), (bincenters_x, n_x), poly 

if __name__=='__main__':
    pass



