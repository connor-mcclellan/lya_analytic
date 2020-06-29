
module m_redistribute_unpol

contains

!------------------------------------------------------------------------------------

! line is the type variable containing line info. accessed through m_linelist module
! temp = atom temperature in K
! p = lab frame incoming photon parameters. replaced with outgoing lab frame values.

subroutine redistribute_wrapper(temp,p)
use m_defs
use m_util
implicit none
real (dp), intent (in) :: temp
type (photon_spherical), intent (inout) :: p
real (dp) :: newnu,newtheta,newphi

call redistribute_Doppler(temp,p%nu,p%theta,p%phi,newnu,newtheta,newphi)

! new frequency
p%nu=newnu

! new direction angles relative to cartesian axes
p%theta=newtheta
p%phi=newphi

! direction vector along cartesian axes
p%nc%x=sin(newtheta)*cos(newphi)
p%nc%y=sin(newtheta)*sin(newphi)
p%nc%z=cos(newtheta)

! direction vector along spherical axes at position theta,phi
call convert_cart_vector_to_sph_vector(p%nc%x,p%nc%y,p%nc%z,p%xs%theta,p%xs%phi,p%ns%r,p%ns%theta,p%ns%phi)

end subroutine redistribute_wrapper

!----------------------------------------------------------------------

! assumes the gas of atoms has a Maxwell-Boltzman distribution
! T is atom temperature in Kelvin
! incoming photon frequency in fluid (not atom) rest frame

subroutine redistribute_Doppler(T,nu_in,theta_in,phi_in,nu_out,theta_out,phi_out)
use m_defs
use m_linelist
use m_sample_uparallel
implicit none
real (dp), intent (in) :: T,nu_in,theta_in,phi_in
real (dp), intent (out) :: nu_out,theta_out,phi_out
real (dp) :: vth,nu0,doppwidth,a,x,r1,r2,vperp,upar,vpar,nu_atom,cosgamma,singamma

vth = sqrt( 2.d0 * kboltz * T / line%mass )    ! atom thermal velocity
nu0=line%nu
doppwidth = nu0*vth/clight
a=line%lorwidth/doppwidth
x = (nu_in-nu0)/doppwidth

! velocity component perpendicular to incident photon, but in
! plane of incident and outoing photons. draw from gaussian.
! this number can be positive or negative.
call random_number(r1)
call random_number(r2)
vperp = vth * sqrt(-log(r1)) * cos(2.d0*pi*r2) 

! parallel velocity component
call sample_uparallel(a,x,upar)
vpar = vth * upar

! atom frame scattering gives outgoing angle and stokes
call sample_dipole(theta_in,phi_in,theta_out,phi_out)

! transform frequency to lab frame. ignore aberration of photon direction.
! just doppler shift frequency.

cosgamma = sin(theta_in)*sin(theta_out)*cos(phi_in-phi_out) + cos(theta_in)*cos(theta_out)
singamma = sqrt( 1.d0 - cosgamma**2 )
nu_out = nu_in + nu0 * ( (cosgamma-1.d0) * vpar + singamma * vperp ) / clight

end subroutine redistribute_Doppler


! sample dipole distribution for outgoing photon direction
! true distribution is P(gamma,phi)dgamma dphi = (3/(16*pi))dgamma * dphi * sin(gamma) * (1 + cos^2(gamma))
! comparison distribution is P'(gamma,phi)d\gamma= sin(gamma) * dgamma * dphi / (4pi)
! so 2*cos(gamma)*(random number) < 1 + cos^2(gamma)

subroutine sample_dipole(theta_in,phi_in,theta_out,phi_out)
use m_defs
implicit none
real (dp), intent (in) :: theta_in,phi_in
real (dp), intent (out) :: theta_out,phi_out
real (dp) :: f,x,r,cosgam
integer :: it

it=0
do
  it=it+1

  ! sample theta_out from uniform in mu
  call random_number( r )
  x=2.d0*r-1.d0
  if (x<-1.d0) x=-1.d0
  if (x>1.d0) x=1.d0
  theta_out = acos( x )

  ! sample phi_out from uniform
  call random_number( r )
  phi_out = 2.d0 * pi * r

  ! compute distribution
  cosgam = sin(theta_in)*sin(theta_out)*cos(phi_in-phi_out) + cos(theta_in)*cos(theta_out)
  f = 1.d0 + cosgam**2

  ! test if ok to accept
  call random_number( r )
  if (2.d0 <= f) then
    print *,"max not big enough"
    stop
  endif
  if ( r*2.d0 <= f) exit

end do

end subroutine sample_dipole

end module m_redistribute_unpol
