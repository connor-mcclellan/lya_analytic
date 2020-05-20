import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
import numpy as np

data = np.loadtxt('error.dat', skiprows=1)
data_dp = np.loadtxt('error_dp.dat', skiprows=1)

plt.figure(figsize=(8, 6))
plt.plot(data[:,0], data[:,1], 'c-', label='single precision', alpha=0.5)
plt.plot(data_dp[:,0], data_dp[:,1], 'm-', label='double precision', alpha=0.5)
plt.xlabel('log n')
plt.ylabel('log err')
plt.yscale('log')
plt.xscale('log')
plt.title('Fractional Error in Midpoint Integration')
plt.legend()
plt.show()

