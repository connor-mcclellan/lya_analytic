import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import pdb

# USER PARAMETERS
# =================
temp = 1e4 # Kelvin
#R_s = 1e11 # cm
#R_p = 1e10 # cm
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

def tau(nu, Rp=1e10):
    return 2*Rp*num_dens*xsec(nu)

def spec(nu, I0=1., Rs=1e11, Rp=1e10):
    abs_piece = Rp**2/tau(nu, Rp=Rp)**2 * (1 - np.exp(-tau(nu, Rp=Rp))*(tau(nu, Rp=Rp) + 1))
    direct_piece = 0.5*(Rs**2 - Rp**2)
    return (2 * np.pi * I0 / d**2 * (abs_piece + direct_piece)) / (np.pi * I0 * Rs**2 / d**2)

nu_bins = get_nu(np.linspace(-100, 100, 300))
vel = (nu_bins - nu0)/nu0 * c

plt.plot(vel, spec(nu_bins), marker='o', ms=1, alpha=0.7, lw=1, label=r'$R_s=10^{11}$ cm, $R_p=10^{10}$ cm')
plt.plot(vel, spec(nu_bins, Rp=1.5e10), marker='s', ms=1, alpha=0.7, lw=1, label=r'$R_s=10^{11}$ cm, $R_p=1.5\times 10^{10}$ cm')

plt.xlabel('velocity (cm/s)')
plt.ylabel(r'flux fraction ($F_{\rm transit}/F_0$)')

plt.legend(loc=3)
plt.show()
