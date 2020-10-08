import mpmath as m
import pdb
import matplotlib.pyplot as plt
import numpy as np

def _integrator(x, y):
    (y2, y1) = y

   # y2 = dy/dx
   # y1 = y

    return [y1, y2]


sol = m.odefun(_integrator, -1000, [1., 0.], verbose=True, tol=m.mpf('1e600')) # Tol is arbitrarily high --- otherwise it takes FOREVER to obtain solutions

x_eval = np.linspace(-1000, -200)
plt.plot(x_eval, [sol(x) for x in x_eval], marker='o')
sols = [sol(x)[1] for x in x_eval]

print('Solution at x=-200:', sol(-200))

plt.show()


