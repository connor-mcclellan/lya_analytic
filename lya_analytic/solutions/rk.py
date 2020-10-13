import numpy as np
import mpmath as m
from collections.abc import Iterable
import pdb
import time

class Solution(object):

    def __init__(self, t, x):
        self.t = t
        self.x = x


def rk(f, bounds, ivs, t_eval=None, dt=1e-1, dt_min=1e-3,
       dx_max=1e0, dx_min=1e0, x_tol=1e-3, verbose=False):

    ivs = np.array([m.mpf(iv) for iv in ivs])
    bounds = np.array([m.mpf(bound) for bound in bounds])

    dt = m.mpf(dt)          # initial step size.
    dt_min = m.mpf(dt_min)  # Min step size.
    dx_max = m.mpf(dx_max)  # Maximum allowed change in x
    # Min change in x, below which step size will increase
    dx_min = m.mpf(dx_min)
    x_tol = m.mpf(x_tol)    # Fixed step size for where the solution is small

    if bounds[0] > bounds[1]:
        bounds = np.flip(bounds)
        sign = m.mpf('-1')
    else:
        sign = m.mpf('1')

    t = bounds[0]
    x = ivs

    if verbose:
        print('Integrating between {} and {}... '.format(*[str(b) for b in bounds]))
        start = time.time()

    while (t < bounds[1]):

        # Calculate normal step
        k1 = sign*f(t, x)
        k2 = sign*f(t + dt / 2, x + dt * k1 / 2)
        k3 = sign*f(t + dt / 2, x + dt * k2 / 2)
        k4 = sign*f(t + dt, x + dt * k3)
        step_x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate half step
        k2 = sign*f(t + dt / 4, x + dt * k1 / 4)
        k3 = sign*f(t + dt / 4, x + dt * k2 / 4)
        k4 = sign*f(t + dt / 2, x + dt * k3 / 2)
        half_step_x = x + dt / 12 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate double step
        k2 = sign*f(t + dt, x + dt * k1)
        k3 = sign*f(t + dt, x + dt * k2)
        k4 = sign*f(t + 2 * dt, x + 2 * dt * k3)
        dble_step_x = x + dt / 3 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Use a fixed step size if any x smaller than the x tolerance.
        if (abs(step_x) < x_tol).any():
            if (dt != dt_min):
                dt = dt_min
            new_x = step_x
        else:
            if (abs(step_x - half_step_x) / abs(step_x) > dx_max).any():
                # Error is too large for one or more x; decrease step size.
                dt = dt / 2
                new_x = half_step_x # Reject? Check again?
            elif (abs(step_x - dble_step_x) / abs(step_x) < dx_min).all():
                # Larger error is acceptable for all x; increase step size.
                dt = dt * 2
                new_x = dble_step_x
            else:
                new_x = step_x 

        x = new_x
        t = t + dt

        try:
            xout.append(np.array(x))
            tout.append(t)
        except BaseException:
            xout = [np.array(x)]
            tout = [t]

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
        x_eval = (x_hi - x_lo)/(t_hi - t_lo)*(t_eval-t[0]) + x[0]

        # Add in right endpoint, if excluded earlier
        if endpoint_flag:
            x_eval = np.hstack([x_eval, x[-1]])

        return x_eval
    return intp


if __name__ == '__main__':

    def _integrator(t, x):
        (x2, x1) = x

       # x2 = dx/dt
       # x1 = x

       # print(t, x1)
        return np.array([x1, x2])

    sol = rk(_integrator, [-1000, 0], [1., 0.], t_eval=np.linspace(-1000, 0, 100), verbose=True)
    pdb.set_trace()
