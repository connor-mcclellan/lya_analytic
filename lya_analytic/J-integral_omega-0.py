import numpy as np
from solutions.util import Params, Line
from solutions.boundaryvalue import BoundaryValue

lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1e6, energy=1., 
           sigma_source=0., n_points=1e5)

omegas = np.array([0])
ns = np.arange(1, 100, 10)

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

np.savetxt('./outputs/J_real.dat', np.array(real))
np.savetxt('./outputs/J_imag.dat', np.array(imag))
np.savetxt('./outputs/omega.dat', np.array(omegas))
np.savetxt('./outputs/n.dat', np.array(ns))
