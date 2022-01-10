import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import pdb

# USER PARAMETERS
# =================
temp = 1e4 # Kelvin
R_s = 1e11 # cm
R_p = 1e10 # cm
d = 1e13 # cm
num_dens=1701290465.5139434
# =================

# Constants (CGS)
c = 29979245800.0
esu = 4.803204712570263e-10
m_e = 9.1093837015e-28
k_B = 1.380649e-16
m_p = 1.67262192369e-24

# Lya
lambda0 = 1215.6701
nu0 = c / (lambda0 * 1e-8)
osc_strength = 0.4164
gamma = 6.265e8

# Derived quantities
vth = np.sqrt(2.0 * k_B * temp / m_p)
delta = nu0 * vth / c
a = gamma / (4.0 * np.pi * delta)

def voigtx_full(a, x):
    # Full
    z = x + a*1j
    H = np.real(wofz(z))
    line_profile = H/np.sqrt(np.pi)
    return line_profile

def phi(a, x): 
    return voigtx_full(a, x) / np.sqrt(np.pi) / delta

def xsec(nu):
    return np.pi * esu**2 / m_e / c * osc_strength * phi(a, (nu-nu0)/delta)

def get_nu(x): 
    return delta * x + nu0

def tau(b, nu):
    return num_dens * (2*np.sqrt(R_p**2 - b**2)) * xsec(nu)

def spec(nu, I0=1., R_s=1e11, R_p=1e10):
    abs_piece = (1 - np.exp(-tau(0, nu))*(tau(0, nu) + 1))/2/(num_dens * xsec(nu))**2
    direct_piece = (R_s**2 - R_p**2)
    return (np.pi * I0 / d**2 * (abs_piece + direct_piece)) / (np.pi * I0 * R_s**2 / d**2)

nu_bins = get_nu(np.linspace(-100, 100, 300))
vel = (nu_bins - nu0)/nu0 * c

plt.plot(vel, spec(nu_bins), marker='o', alpha=0.7, lw=1, label=r'$R_s=10^{11}$ cm, $R_p=10^{10}$ cm')

plt.xlabel('velocity (cm/s)')
plt.ylabel(r'flux fraction ($F_{\rm transit}/F_0$)')

plt.legend(loc=3)
plt.show()
