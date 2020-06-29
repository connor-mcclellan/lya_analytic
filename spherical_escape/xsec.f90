! wrapper for cross section. using voigt now but can switch to doppler or natural if needed.

module m_xsec

contains

!------------------------------------------------------------------------------------

! used to compute optical depth
! line derived type in module linelist
! T=temp of the scatterer
! nu in Hz, in rest frame of fluid
! sigmatot in cm^2

subroutine sigmatot_wrapper(T,nu,sigmatot)
use m_defs, only: dp,kboltz
use m_linelist, only : line
implicit none
integer :: i
real (dp), intent (in) :: T,nu
real (dp), intent (out) :: sigmatot
real (dp) :: vth
integer :: isc,iimp

vth = sqrt( 2.d0 * kboltz * T / line%mass )
call xsec_voigt(nu,vth,sigmatot)

end subroutine sigmatot_wrapper

!------------------------------------------------------------------------------------

! cross sections with Lorenztian damping

subroutine xsec_lorentzian(nu,sigmatot)
use m_defs, only : dp,pi,charge,melectron,clight
use m_linelist, only : line
implicit none
real (dp), intent (in) :: nu
real (dp), intent (out) :: sigmatot
real (dp) :: prefactor,L

prefactor = pi*charge**2/(melectron*clight) * line%f

! normalized Lorentzians
L = line%lorwidth/pi / ( (nu-line%nu)**2 + line%lorwidth**2 )

sigmatot = prefactor * L

end subroutine xsec_lorentzian

!------------------------------------------------------------------------------------

! cross section with just Doppler broadening

subroutine xsec_doppler(nu,vth,sigmatot)
use m_defs, only : dp,pi,charge,melectron,clight
use m_linelist, only : line
implicit none
real (dp), intent (in) :: nu,vth
real (dp), intent (out) :: sigmatot
real (dp) :: prefactor,doppwidth,x,lineprofile

prefactor = pi*charge**2/(melectron*clight) * line%f

! normalized Lorentzians
doppwidth = line%nu * vth / clight
x = (nu-line%nu)/doppwidth
lineprofile = exp(-x**2) / sqrt(pi) / doppwidth

sigmatot = prefactor * lineprofile

end subroutine xsec_doppler


!-------------------------------------------------------------------------------------------

! cross section with voigt profile
! nu in Hz, in rest frame of fluid
! vth = thermal velocity in cm/s of atom
! sigmatot in cm^2

subroutine xsec_voigt(nu,vth,sigmatot)
use m_defs, only : dp,pi,charge,melectron,clight
use m_linelist, only : line
use m_voigt
implicit none
real (dp), intent (in) :: nu,vth
real (dp), intent (out) :: sigmatot
real (dp) :: prefactor,doppwidth,x,H,lineprofile,a

prefactor = pi*charge**2*line%f/(melectron*clight) 		! line strength

doppwidth = line%nu * vth / clight
a = line%lorwidth / doppwidth
x = (nu-line%nu)/doppwidth
call voigt(a,x,H)
print *, H
lineprofile = H / sqrt(pi) / doppwidth
print *, prefactor, lineprofile
sigmatot = prefactor * lineprofile

if (sigmatot < 0.d0) then
  print *,"problem with sigmatot"
  print *,"nu=",nu
  print *,"vth=",vth
  print *,"prefactor=",prefactor
  print *,"line%f=",line%f
  print *,"voigt a=",a
  print *,"x=",x
  print *,"H=",H
  print *,"doppwidth=",doppwidth
  print *,"lineprofile=",lineprofile
  stop
endif

end subroutine xsec_voigt


end module m_xsec
