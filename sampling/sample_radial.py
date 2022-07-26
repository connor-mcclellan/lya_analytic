# Given some function f(r), how do we randomly sample points to reconstruct the function?

import numpy as np
import matplotlib.pyplot as plt
import pdb
alpha = 0.66667

# The function f(r):
def func(r):
    return r**(-alpha)

def funcint(r):
    return 1/(-alpha + 3)*r**(-alpha + 3.)

# A random sample of n points from a distribution in r
n = 100000
R1 = 1.
R2 = 10.
weights = []
rsamp = []
for i in range(n):
    zi = np.random.random()
    r = np.cbrt((R2**3. - R1**3.) * zi + R1**3.)
    rsamp.append(r)
    weight = func(r)/3.*(R2**3. - R1**3.) / (funcint(R2) - funcint(R1))
    weights.append(weight)

rsamp = np.array(rsamp)
weights = np.array(weights)

fig, ax = plt.subplots()

# Weighted distribution
ax.hist(rsamp, bins=50, histtype='step', weights=weights, label="Weighted by f(r)")

# Uniform distribution
ns, bins, _ = ax.hist(rsamp, bins=50, histtype='step', label='Uniform in $r^2 dr$')

bincenters = (bins[1:] + bins[:-1])/2.
binwidths = np.diff(bins)
ax.set_xlabel('r')
ax.set_ylabel('N')

r = np.linspace(0, 10, 200)

# Weighted analytic solution
ax.plot(r, n * binwidths[0] * func(r) * r**2. / (funcint(R2) - funcint(R1)), label=r'$f(r) r^2 dr$ distribution')

# Uniform analytic solution
ax.plot(r, n * binwidths[0] * 3 * r**2. / (R2**3. - R1**3.), label=r'$r^2 dr$ distribution')
plt.legend()
plt.show()
