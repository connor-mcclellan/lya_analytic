import numpy as np
from solutions.util import Params, Line
from solutions.boundaryvalue import BoundaryValue
import pdb

lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1e6, energy=1., 
           sigma_source=0., n_points=1e5)

omegas = np.array([0])
ns = np.arange(1, 15, 1)

real = []
imag = []

for omega in omegas:

    real_omega = []
    imag_omega = []

    for n in ns:
        bv = BoundaryValue(n, omega, p)
        _, J_real, J_imag = bv.solve()

        real_omega.append(J_real)
        imag_omega.append(J_imag)

    real.append(real_omega)
    imag.append(imag_omega)

np.save('./outputs/J_real', np.array(real))
np.save('./outputs/J_imag', np.array(imag))
np.save('./outputs/omega', np.array(omegas))
np.save('./outputs/n', np.array(omegas))

