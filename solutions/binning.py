import numpy as np
import matplotlib.pyplot as plt

def photon_hist(data):
    '''
    Create binned histogram data from CM Monte Carlo outputs. Returns 
    normalized distance traveled and probability distribution as a tuple.
    '''
    # Load photon data, take care of float errors in r
    data[:, 0] = np.round(data[:, 0], 5)

    # Set up figure, make initial histogram, normalize x and y
    n, bins, patches = plt.hist(
        data[:, 6], bins=50, color='k', histtype='step', density=True)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    plt.close()

    return bincenters, n

