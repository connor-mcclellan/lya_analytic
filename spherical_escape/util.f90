! utility routines for working with vectors, and making sure an angle variable phi is in the range [0,2pi]

module m_util

contains

subroutine convert_coords_cart_to_sph(x,y,z,r,th,phi)
use m_defs, only : dp,pi
implicit none
real (dp), intent (in) :: x,y,z
real (dp), intent (out) :: r,th,phi
real (dp) :: rsq

rsq = x**2+y**2+z**2

if (rsq <= 0.d0) then
  r=0.d0
  th=0.d0
  phi=0.d0
  return
endif

r = sqrt( rsq )
th = acos( z/r )                     ! in the range [0,pi]

if ( th == 0.d0 .or. th == pi) then
  phi = 0.d0
  return
endif

phi = atan2( y, x )
if (phi < 0.d0) phi = phi + 2.d0*pi     ! in the range [0,2pi]

end subroutine convert_coords_cart_to_sph




! here (vx,vy,vz) is a cartesian vector at position (r,th,phi).
! the spherical basis vectors depend on the positional th and phi.
! take dot products of spherical basis vector and cartesian vector

subroutine convert_cart_vector_to_sph_vector(vx,vy,vz,th,phi,vr,vth,vphi)
use m_defs, only: dp,pi
implicit none
real (dp), intent (in) :: vx,vy,vz,th,phi
real (dp), intent (out) :: vr,vth,vphi
real (dp) :: sth,cth,sphi,cphi

sth=sin(th)
cth=cos(th)
sphi=sin(phi)
cphi=cos(phi)

vr=sth*(cphi*vx + sphi*vy) + cth*vz
vth=cth*(cphi*vx + sphi*vy) - sth*vz
vphi=-sphi*vx+cphi*vy

end subroutine convert_cart_vector_to_sph_vector


subroutine wrap_phi(phi)
use m_defs, only : dp, pi
implicit none
real (dp) :: phi

phi=mod(phi,2.d0*pi)
if (phi < 0.d0) then
  phi=phi+2.d0*pi
else if (phi >= 2.d0*pi) then
  phi=phi-2.d0*pi
endif

end subroutine wrap_phi



subroutine basis_vectors(th,phi,er,eth,ephi)
use m_defs, only : dp
implicit none
real (dp), intent (in) :: th,phi
real (dp), intent (out) :: er(3),eth(3),ephi(3)
real (dp) :: sth,cth,sphi,cphi

sth=sin(th)
cth=cos(th)
sphi=sin(phi)
cphi=cos(phi)

er=(/sth*cphi,sth*sphi,cth/)
eth=(/cth*cphi,cth*sphi,-sth/)
ephi=(/-sphi,cphi,0.d0/)

end subroutine basis_vectors


end module m_util
