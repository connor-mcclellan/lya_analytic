import numpy as np


class fundconst:
    def __init__(self):			# cgs units
        self.cgrav = 6.67429e-8		# gravitational constant
        self.hplanck = 6.62606896e-27		# h = plancks contstant
        self.clight = 2.997924589e10		# speed of light
        self.kboltz = 1.3806504e-16		# boltzmann's constant
        self.charge = 4.80320427e-10		# charge unit
        self.abohr = 0.52917720859e-8		# bohr radius
        self.melectron = 9.10938215e-28 	# electron mass
        self.mproton = 1.672621637e-24  # proton
        self.amu = 1.660538782e-24		# mass unit
        self.angstrom_in_cm = 1.e-8		# 1 angstrom is 10^{-8} cm
        # classical oscillator "line strength" in cgs
        self.line_strength = np.pi * self.charge**2 / \
            (self.melectron * self.clight)


class lymanalpha:
    def __init__(self):
        fc = fundconst()
        self.lambda0 = 1215.6701 * fc.angstrom_in_cm  # wavelenth in cm
        self.osc_strength = 0.4164				# oscillator strength
        self.Gamma = 6.265e8					# decay rate in s^{-1}
        self.nu0 = fc.clight / self.lambda0			# line center frequency in Hz


def main():

    T = 1.e4
    fc = fundconst()
    la = lymanalpha(T)

    # print
    # print "fundamental constants:"
    # print "cgrav=",fc.cgrav
    # print "hplanck=",fc.hplanck
    # print "clight=",fc.clight
    # print "kboltz=",fc.kboltz
    # print "charge=",fc.charge
    # print "abohr=",fc.abohr
    # print "melectron=",fc.melectron
    # print "mproton=",fc.mproton
    # print "amu=",fc.amu
    # print "light strength=",fc.line_strength
    # print
    # print "Lyman alpha stuff:"
    # print "lambda0=",la.lambda0
    # print "Gamma=",la.Gamma
    # print "nu0=",la.nu0
    # print


if __name__ == '__main__':
    main()
