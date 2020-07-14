import numpy as np
from scipy.stats import lognorm
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
from solutions.util import tanf
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})


def series_fit(t, tau):
    norm = 1. / tau * (tau + 2. / 3.)**2. * 3. / np.pi**2.

    y = np.exp(-t/norm)
    prob = np.zeros(np.shape(t))
    yn2 = np.zeros(np.shape(t))
    for i in range(len(prob)):
        n = 1
        sol = optimize.root_scalar(
            tanf, args=(tau), bracket=[
                0.51 * np.pi, 1.49 * np.pi])
        lamn = sol.root
        dlamn = lamn
        In = 0.5 * tau * (1 + (1.5 * tau - 1) / ((1.5 * tau - 1)**2 + lamn * lamn))
        yn2[i] = (y[i])**(lamn * lamn / (np.pi * np.pi)) * np.cos(lamn) * (1.5 *
              tau**2 / (1 - 1.5 * tau) / In) * (lamn * lamn / (np.pi * np.pi))
        while (abs(yn2[i]) > 1.e-17):
            prob[i] = prob[i] + yn2[i]
            n = n + 1
            bracket = [lamn + (1. - 0.1 / n) * dlamn,
                       lamn + (1. + 0.1 / n) * dlamn]
            sol = optimize.root_scalar(tanf, args=(tau), bracket=bracket)
            dlamn = sol.root - lamn
            lamn = sol.root
            In = 0.5 * tau * (1 + (1.5 * tau - 1) /
                              ((1.5 * tau - 1)**2 + lamn * lamn))
            yn2[i] = (y[i])**(lamn * lamn / (np.pi * np.pi)) * np.cos(lamn) * (1.5 *tau**2 / (1 - 1.5 * tau) / In) * (lamn * lamn / (np.pi * np.pi))
    return prob


def lognorm_fit(x, xdata=None):
    '''
    Fit a log normal distribution to the data. Uses actual input data and is 
    slower as a result.
    '''
    if xdata is None:
        xdata = x
    shape, loc, scale = lognorm.fit(xdata, loc=1)
    pdf = lognorm.pdf(x, shape, loc, scale)
    return pdf, np.log(scale), shape


if __name__ == '__main__':
    pass
