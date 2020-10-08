import numpy as np
import mpmath as m
import pdb


class Solution(object):

    def __init__(self, t, x):
        self.t = t
        self.x = x


def rk(f, bounds, ivs, dt=1e-1, dt_min=1e-3,
       dx_max=1e0, dx_min=1e0, x_tol=1e-3):

    # TODO: What if bounds are reversed? Can this integrate right to left?
    # Perhaps changing the sign of dt will be enough

    ivs = np.array([m.mpf(iv) for iv in ivs])
    bounds = np.array([m.mpf(bound) for bound in bounds])

    dt = m.mpf(dt)          # initial step size.
    dt_min = m.mpf(dt_min)  # Min step size.
    dx_max = m.mpf(dx_max)  # Maximum allowed change in x
    # Min change in x, below which step size will increase
    dx_min = m.mpf(dx_min)
    x_tol = m.mpf(x_tol)    # Fixed step size for where the solution is small

    t = bounds[0]
    x = ivs

    while (t < bounds[1]):

        # Calculate partial steps.
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt * k1 / 2)
        k3 = f(t + dt / 2, x + dt * k2 / 2)
        k4 = f(t + dt, x + dt * k3)
        # Combine partial steps.
        step_x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate partial steps.
        k2 = f(t + dt / 4, x + dt * k1 / 4)
        k3 = f(t + dt / 4, x + dt * k2 / 4)
        k4 = f(t + dt / 2, x + dt * k3 / 2)
        # Combine partial steps.
        half_step_x = x + dt / 12 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate partial steps.
        k2 = f(t + dt, x + dt * k1)
        k3 = f(t + dt, x + dt * k2)
        k4 = f(t + 2 * dt, x + 2 * dt * k3)
        # Combine partial steps.
        dble_step_x = x + dt / 3 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Use a fixed step size if any x smaller than the x tolerance.
        if (abs(step_x) < x_tol).any():
            if (dt != dt_min):
                print("New step size", dt_min)
                dt = dt_min
            new_x = step_x
        else:
            if (abs(step_x - half_step_x) / abs(step_x) > dx_max).any():
                # Error is too large for one or more x; decrease step size.
                dt = dt / 2
                print("New step size", dt)
                new_x = half_step_x
            elif (abs(step_x - dble_step_x) / abs(step_x) < dx_min).all():
                # Larger error is acceptable for all x; increase step size.
                dt = dt * 2
                print("New step size", dt)
                new_x = dble_step_x
            else:
                new_x = step_x  # This step size is just right.

        x = new_x
        t = t + dt

        try:
            xout.append(np.array(x))
            tout.append(t)
        except BaseException:
            xout = [np.array(x)]
            tout = [t]

    return Solution(np.array(tout), np.array(xout).T)


if __name__ == '__main__':

    def _integrator(t, x):
        (x2, x1) = x

       # x2 = dx/dt
       # x1 = x

        print(t, x1)
        return np.array([x1, x2])

    sol = rk(_integrator, [-1000, 0], [1., 0.])
    pdb.set_trace()
