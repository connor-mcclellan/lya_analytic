import numpy as np
import matplotlib.pyplot as plt
from solutions import fits
from solutions.prob_ct_tau import prob_ct_tau

array = np.load('./outputs/bin_time.npy')
tc = array[0]
ydata = array[1]
theory = array[2]
tau0=1e7

norm = np.linspace(1./10., 1./2., 10)
prob = np.zeros((len(norm), len(tc)))
for j in range(len(norm)):
    for i in range(len(tc)):
        prob[j, i] = prob_ct_tau(tc[i], tau0)/norm[j]
    plt.plot(tc, 2.3*prob[j]*tc, '--', label='Norm={:.2f}'.format(norm[j]), alpha=0.25)
plt.plot(tc, ydata, '.', label='Data')
plt.ylim((1e-4, 1e1))
plt.yscale('log')
plt.legend()
plt.show()

