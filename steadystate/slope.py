from solutions.util import Line, Params, scinot, voigtx, midpoint_diff
import solutions.ftsoln as ftsoln
import numpy as np
import matplotlib.pyplot as plt
import pdb


# Line parameters object
lya = Line(1215.6701, 0.4164, 6.265e8)
p = Params(line=lya, temp=1e4, tau0=1e7, energy=1., R=1e11, sigma_source=0.0, n_points=1e4)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for i, tau in enumerate([1e7]):
    tau0, xinit, temp, radius, L = (tau, 0.0, 1e4, 1e11, 1.)
    x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(tau0, xinit, temp, radius, L)
    norm = 4.0 * np.pi * radius**2 * p.delta * 4.0 * np.pi / L
    Hsp = norm*Hsp_ft
    deriv = midpoint_diff(Hsp)#dH0_dsigma(radius, sigma_ft, x_ft, p) * np.diff(sigma_ft)[0]

    ax.plot(x_ft, norm*Hh_ft, label='Hbc')
    ax.plot(x_ft, Hsp, label='H0')
    ax.plot(x_ft, deriv, label='dH0/dsigma')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.show()



