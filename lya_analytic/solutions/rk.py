import numpy as np
import mpmath as m
from scipy.interpolate import interp1d
import pdb


class Solution(object):

    def __init__(self, t, x):
        self.t = t
        self.x = x


def rk(f, bounds, ivs, t_eval=None, dt=1e-1, dt_min=1e-3,
       dx_max=1e0, dx_min=1e0, x_tol=1e-3):

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

    # TODO: Reduce solution array to t_eval, if t_eval is specified
    # TODO: Write multidimensional interpolator using mpmath
    if t_eval is not None:
        xinterp = interp1d(tout, xout)
        x_eval = xinterp(t_eval)
    else:
        t_eval, x_eval = tout, xout

    return Solution(np.array(t_eval), np.array(x_eval).T)


def interpolate(t, x):
    def intp(t_eval):
        for i in range(len(t_eval)):
            x_eval = np.zeros((len(t_eval), len(x)))
            t_index = (np.abs(t - t_eval[i])).argmin()
            if t[t_index] < t_eval[i]:
                t_lo = t[t_index]
                t_hi = t[t_index+1]
                x_lo = x[:, t_index]
                x_hi = x[:, t_index+1]
            else:
                t_hi = t[t_index]
                t_lo = t[t_index-1]
                x_hi = x[:, t_index]
                x_lo = x[:, t_index-1]
            x_eval[i] = (x_hi - x_lo)/(t_hi - t_lo)*t_eval[i]
        return x_eval
    return intp


if __name__ == '__main__':

    def _integrator(t, x):
        (x2, x1) = x

       # x2 = dx/dt
       # x1 = x

       # print(t, x1)
        return np.array([x1, x2])

    sol = rk(_integrator, [-1000, 0], [1., 0.])
    pdb.set_trace()
