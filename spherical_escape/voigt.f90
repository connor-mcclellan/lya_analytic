
! voigt function computed from fried-conte plasma dispersion function

! this voigt function is the standard notation. not normalized to 1!!!
! must divide by sqrt(pi) to get something normalized to 1 (for unit doppler width).

! as of feb 2013, zeta.for "plasma dispersion function" is fastest, while still
! accurate (to better than part in 10^5, checke against other codes, like
! ode_real). downloaded from https://w3.pppl.gov/~hammett/comp/src/zeta.for.

! this routine is 150x faster than numerical integration (with eps=1.d-6), but
! has a numerical accuracy that is fixed (but small, no worse than part in
! 10^5).

! H(a,x) = (a/pi) \int_{-\infty}^{\infty} dt exp(-t^2) / ( a^2 + (t-x)^2 )
! = Im( f(z) ) / sqrt(pi)
! where z=x+ia.

! x=(nu-nu0)/Delta_nu_dopp
! nu0=line center frequency
! Delta_nu_Dopp = nu0 * vthermal / clight
! a=Gamma/(4*pi*Delta_nu_Dopp) is the "damping parameter"
! Gamma is decay rate from upper to lower state
! H=voigt function, normed by \int dx H = sqrt(pi)

module m_voigt

contains

subroutine voigt(a,x,H)
use m_defs, only : dp, pi
implicit none
real (dp), intent (in) :: a,x
real (dp), intent (out) :: H
real :: a4,x4,H4,pi4
complex :: z,imag,f
interface
  function zeta(arg)
  COMPLEX ZETA, DZETA, ARG
  end function zeta
end interface

a4=real(a)
x4=real(x)
pi4=real(pi)
z = cmplx(x4,a4)
f = zeta(z)
H4 = aimag(f) / sqrt(pi4)
H=dble(H4)

end subroutine voigt

end module m_voigt
