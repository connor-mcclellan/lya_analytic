import numpy as np
import matplotlib.pyplot as plt

a = np.load('./outputs/n8_sigma1e6_omega64/Jrt.npy')
sigma = np.load('./outputs/n8_sigma1e6_omega64/sigma_grid.npy')
r = np.linspace(1e-1, 1e11, int(1e1))
t = np.linspace(0, 1e3, int(1e1))

for i in range(len(t)): 
    plt.plot(sigma, a[-1, :, i], '-', alpha=0.5, label='t={}'.format(t[i]))

plt.title('r={}'.format(r[-1]))  
plt.xlabel('sigma')
plt.ylabel('J(r, sigma, t)')
plt.yscale('log')
plt.legend()
plt.show()
