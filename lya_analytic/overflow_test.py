import mpmath as m
import pdb
import matplotlib.pyplot as plt
import numpy as np
from solutions.rk import rk

def _integrator(x, y):
    (y2, y1) = y

   # y2 = dy/dx
   # y1 = y

    return np.array([y1, y2])


#sol = m.odefun(_integrator, -1000, [1., 0.], verbose=True, tol=m.mpf('1e600')) # Tol is arbitrarily high --- otherwise it takes FOREVER to obtain solutions

#x_eval = np.linspace(-1000, -200)
#plt.plot(x_eval, [sol(x) for x in x_eval], marker='o')
#sols = [sol(x)[1] for x in x_eval]

#print('Solution at x=-200:', sol(-200))


# Custom RK convergence test
for tol in np.linspace(1e0, 1e-4, 4):
    sol = rk(_integrator, [-1000, 0], [1., 0.], t_eval=np.linspace(-1000, 0, 10000), dx_max=tol, dx_min=tol, verbose=True)
    logx = np.array([m.log10(val) for val in sol.x[1]])
    pdb.set_trace()
    plt.plot(sol.t, logx, label='tol={} with eval'.format(tol))

plt.ylabel('log x')
plt.xlabel('t')



plt.show()


