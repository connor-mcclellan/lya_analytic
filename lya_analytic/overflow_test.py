import mpmath as m
import pdb
import matplotlib.pyplot as plt
import numpy as np
from solutions.rk import rk
import pickle
import time
from scipy.integrate import solve_ivp

def _integrator(x, y, args=None):
    (y2, y1) = y

   # y2 = dy/dx
   # y1 = y

    return np.array([y1, y2])


def solution(t):
    return np.log10(np.e) * (t + 1000)

#sol = m.odefun(_integrator, -1000, [1., 0.], verbose=True, tol=m.mpf('1e600')) # Tol is arbitrarily high --- otherwise it takes FOREVER to obtain solutions

#x_eval = np.linspace(-1000, -200)
#plt.plot(x_eval, [sol(x) for x in x_eval], marker='o')
#sols = [sol(x)[1] for x in x_eval]

#print('Solution at x=-200:', sol(-200))


# Custom RK convergence test

#try:
#    ax = pickle.load(open('./plotting/rk_converge.p', 'rb'))
#except:
ax = plt.subplot(111)



for tol in np.logspace(0, -2, 3):
    start = time.time()
    sol = rk(_integrator, [-999, 0], [0., 1.], dx_max=tol, dx_min=tol, verbose=True)
    end = time.time()
    logx = np.array([np.abs(m.log10(val) - solution(sol.t[i]))/solution(sol.t[i]) for i, val in enumerate(sol.x[1])])
    pickle.dump(np.array([sol.t, logx]), open('./plotting/tol{}_{:.1f}s_fracerr.p'.format(tol, end-start), 'wb'))

ivp = solve_ivp(_integrator, [-999, 0], [0., 1.])
pickle.dump(np.array([ivp.t, np.abs(np.log10(ivp.y[1]) - solution(ivp.t))/solution(ivp.t)]), open('./plotting/scipy.integrate.solve_ivp_fracerr.p', 'wb'))
#ax.plot(ivp.t, np.log10(ivp.y[1]), marker='o', ms=2, alpha=0.5, label='scipy.integrate.solve_ivp')
#plt.legend()
#ax.plot(sol.t, logx, marker='o', ms=2, alpha=0.5, label='{:.1f}s, tol={}'.format(end-start, tol))
#pickle.dump(ax, open('./plotting/rk_converge_noeval.p', 'wb'))
