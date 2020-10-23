import numpy as np
import mpmath as m
from collections.abc import Iterable
import pdb
import time
import matplotlib.pyplot as plt

class Solution(object):

    def __init__(self, t, x):
        self.t = t
        self.x = x


def rk(f, bounds, ivs, t_eval=None, dt=1., dx_max=1e-3, dx_min=1e-6, args=None, verbose=False):

    ivs = np.array([m.mpf(iv) for iv in ivs])
    bounds = np.array([m.mpf(bound) for bound in bounds])

    dt = m.mpf(dt)          # initial step size.
    dx_max = m.mpf(dx_max)  # Maximum allowed change in x
    # Min change in x, below which step size will increase
    dx_min = m.mpf(dx_min)

    if bounds[0] > bounds[1]:
        bounds = np.flip(bounds)
        sign = m.mpf('-1')
    else:
        sign = m.mpf('1')

    t = bounds[0]
    x = ivs

    xout = [np.array(x)]
    tout = [t]

    if verbose:
        print('Integrating between {} and {}... '.format(*[str(b) for b in bounds]))
        start = time.time()

    while (t < bounds[1]):

        # Calculate double step
        k1 = sign*f(t, x, args)
        k2 = sign*f(t + dt, x + dt * k1, args)
        k3 = sign*f(t + dt, x + dt * k2, args)
        k4 = sign*f(t + 2 * dt, x + 2 * dt * k3, args)
        dble_step_x = x + dt / 3 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate two normal steps
        k2 = sign*f(t + dt / 2, x + dt * k1 / 2, args)
        k3 = sign*f(t + dt / 2, x + dt * k2 / 2, args)
        k4 = sign*f(t + dt, x + dt * k3, args)
        step_x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        k1 = sign*f(t, step_x, args)
        k2 = sign*f(t + dt / 2, step_x + dt * k1 / 2, args)
        k3 = sign*f(t + dt / 2, step_x + dt * k2 / 2, args)
        k4 = sign*f(t + dt, step_x + dt * k3, args)
        step_x_2 = step_x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

 #       print(dt, np.abs(step_x_2 - dble_step_x)/np.abs(step_x_2))

        if (np.abs(step_x_2 - dble_step_x) > np.abs(step_x_2)*dx_max).any():
            # Error is too large for one or more x; decrease step size.
            dt = dt / 2
            continue
        elif (np.abs(step_x_2 - dble_step_x) <= np.abs(step_x_2)*dx_min).all():
            # Larger error is acceptable for all x; increase step size.
            dt = dt * 2
            continue
        else:
            # Truncation error is within desired bounds. Take higher precision solution.
            new_x = step_x_2

#        print(dt)
#        print(t, dt, step_x_2, dble_step_x)
        x = new_x
        t = t + 2*dt

        xout.append(np.array(x))
        tout.append(t)

    if verbose:
        end = time.time()
        print('Done in {:.1f} s.'.format(end-start))

    tout = np.array(tout)
    xout = np.array(xout)

    if t_eval is not None:
        if verbose:
            start = time.time()
            print('Interpolating solution at desired points...')

        x_eval = []
        for coldata in xout.T:
            xinterp = interpolate(tout, coldata)
            x_eval.append(xinterp(t_eval))

        if verbose:
            end = time.time()
            print('Done in {:.1f} s.'.format(end-start))
    else:
        t_eval, x_eval = tout, xout.T
    return Solution(np.array(t_eval), np.array(x_eval))


def interpolate(t, x):
    # Create a function which will return the value of x at any t_eval
    def intp(t_eval):
        ind = np.searchsorted(t, t_eval, side='right')

        # Remove last index if endpoint is at max
        endpoint_flag = False
        if isinstance(t_eval, Iterable):
            if t_eval[-1] == t[-1]:
                ind = ind[:-1]
                t_eval = t_eval[:-1]
                endpoint_flag = True
        else:
            # t_eval is a single number, at the rightmost endpoint
            if t_eval == t[-1]:
                return x[-1]

        x_hi = x[ind]
        x_lo = x[ind-1]
        t_hi = t[ind]
        t_lo = t[ind-1]
        delta = (t_eval - t_lo)/(t_hi - t_lo)
        x_eval = x_hi * delta + x_lo * (1. - delta)

        # Add in right endpoint, if excluded earlier
        if endpoint_flag:
            x_eval = np.hstack([x_eval, x[-1]])

        return x_eval
    return intp


if __name__ == '__main__':

    from scipy.integrate import solve_ivp

    def _integrator(t, x, args=None):
        (x2, x1) = x

       # x2 = dx/dt
       # x1 = x

       # print(t, x1)
        return np.array([x1, x2])

    def solution(t):
        return np.exp(t+999.)
  
    ivp = solve_ivp(_integrator, [-999, 0], [1., 1.])
    sol = rk(_integrator, [-999, 0], [1., 1.], verbose=True)
    fracerr = True
    if fracerr:
        plt.plot(ivp.t, np.abs(ivp.y[1]-solution(ivp.t))/solution(ivp.t), marker='o', ms=2, alpha=0.25, label='scipy')
        plt.plot(sol.t, [np.abs(x-solution(float(sol.t[i])))/solution(float(sol.t[i])) for i, x in enumerate(sol.x[1])], marker='o', ms=2, alpha=0.25, label='rk')
    else:
        plt.plot(ivp.t, np.log10(ivp.y[1]), marker='o', ms=2, alpha=0.25, label='scipy')
        plt.plot(sol.t, [m.log10(x) for x in sol.x[1]], marker='o', ms=2, alpha=0.25, label='rk')
        plt.plot(sol.t, solution(sol.t), marker='o', ms=2, alpha=0.25, label='solution')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')
    pdb.set_trace()
