
! these routines sample "parallel velocity" (atom velocity parallel to the incoming photon)
! from the distribution f(u)=(a/pi/H(a,x)) * e^{-u^2}/ ( (x-u)^2 + a^2 ), where
! x is dimensionless frequency, a is damping parameter. See zme ref below.
! here H is the Voigt function.
! returned is parallel velocity u, in units of atom's thermal velocity (2kT/m_atom)^{1/2}.

! this file has some tries that didn't work out well. maybe come back to this. need a faster routine.


module m_sample_uparallel

contains

subroutine sample_uparallel(a,x,u)
use m_defs, only : dp,pi
implicit none
real (dp), intent (in) :: a,x
real (dp), intent (out) :: u
! use zme code below. nothing else working well at this point.
call zme(a,x,u)
end subroutine sample_uparallel


! 3 zones. not good.

subroutine thr(a,x_in,u)
use m_defs, only : dp,pi
use m_erfinv
implicit none
real (dp), intent(in) :: a,x_in
real (dp), intent(out) :: u
real (dp) :: x
real (dp) :: up,um,u3,sigma,m1,m2,m3,w1,w2,w3,p1,p2,p3,r1,r2,r3,thp,th3
integer :: i
logical :: done

x=abs(x_in)

if (x**2<4.d0 ) then
  print *,"x**2<4!!"
  stop
endif

up=x-1.d0 ! where second derivative of exponent changes sign
!up= x/2.d0 * ( 1.d0 + sqrt(1.d0-4.d0/x**2) )
um=x/2.d0 * ( 1.d0 - sqrt(1.d0-4.d0/x**2) )
u3=2.d0*x-up

if ( (um-x)**2 < 1.d0) then
  print *,"(um-x)**2 < 1"
  stop
endif

sigma=1.d0/sqrt(2.d0)/sqrt( 1.d0 - 1.d0/(um-x)**2 )

m1 = exp(-up**2)/( (up-x)**2 + a**2 ) * exp((up-um)**2/(2.d0*sigma**2))
m2 = exp(-up**2)
m3 = exp(-u3**2)/( (u3-x)**2 + a**2 ) * exp((u3-um)**2/(2.d0*sigma**2))

w1 = m1 * sigma * sqrt(pi/2.d0) * ( erf( (up-um)/sqrt(2.d0)/sigma ) + 1.d0 )
w2 = m2 / a * ( atan((u3-x)/a) - atan((up-x)/a) )
w3 = m3 * sigma * sqrt(pi/2.d0) * ( 1.d0 - erf( (u3-um)/sqrt(2.d0)/sigma ) )

p1=w1/(w1+w2+w3)
p2=w2/(w1+w2+w3)
p3=w3/(w1+w2+w3)

call random_number(r1)
if (r1<p1) then
  
  done=.false.
  do
    if (done) exit
    call random_number(r2)
    u=um + sqrt(2.d0)*sigma*erfinv( r2*( erf( (up-um)/sqrt(2.d0)/sigma ) + 1.d0 ) - 1.d0  )
    call random_number(r3)
    !print *,u,m1*exp(-(u-um)**2/2.d0/sigma**2),exp(-u**2)/( (u-x)**2 + a**2 )
    if ( r3*m1*exp(-(u-um)**2/2.d0/sigma**2) < exp(-u**2)/( (u-x)**2 + a**2 ) ) done=.true.
  end do

else if (r1>=p1 .and. r1<p2) then
  th3=atan((u3-x)/a)
  thp=atan((up-x)/a)
  done=.false.
  do
    if (done) exit
    call random_number(r2)
    u=x + a*tan( r2*(th3-thp) + thp )
    call random_number(r3)
    if (r3*m2 <= exp(-u**2) ) done=.true.
  end do

else if (r1>=p2) then

  done=.false.
  do
    if (done) exit
    call random_number(r2)
    u=um + sqrt(2.d0)*sigma*erfinv( r2*( 1.d0 - erf( (u3-um)/sqrt(2.d0)/sigma ) ) + erf( (u3-um)/sqrt(2.d0)/sigma )  )
    call random_number(r3)
    if (r3*m3*exp(-(u-um)**2/2.d0/sigma**2) < exp(-u**2)/( (u-x)**2 + a**2 ) ) done=.true.
  end do

endif

if (x_in<0.d0) u=-u

end subroutine thr




! rejection method to sample the parallel atom velocity.
! use zheng and miralda-escude (2002) appendix.
! choose parallel velocity given incoming lab frame frequency.
! "monte Carlo Simulation of Ly alpha Scattering and
! Application to Damped Ly alpha Systems"
! The Astrophysical Journal, Volume 578, Issue 1, pp. 33-42.

subroutine zme(a,x_in,u)
use m_defs, only : dp,pi
implicit none
real (dp), intent (in) :: a,x_in
real (dp), intent (out) :: u
real (dp) :: x,u0,th0,p,th,r1,r2,r3
integer :: nit

x=abs(x_in)                     ! switch sign at end
u0=0.96*x
th0=atan((u0-x)/a)
p=(th0+pi/2.0)/(th0+pi/2.0 + exp(-u0**2)*(pi/2.0-th0))

nit=0
do
  nit=nit+1

  call random_number(r1)
  call random_number(r2)
  call random_number(r3)
  if (r1<=p) then
    th=-pi/2.d0 + (th0+pi/2.d0)*r2
    u=x+a*tan(th)
    if (r3<exp(-u**2)) exit
  else
    th=th0 + (pi/2.0-th0)*r2
    u=x+a*tan(th)
    if (r3<exp(u0**2-u**2)) exit
  endif

end do

! now use symmetry (x,u) <-> (-x,-u) and switch sign
if (x_in<0.d0) then
 u=-u
endif

end subroutine zme

! sample lorentzian and compare to gaussian
! fast in line core but slow on wings

subroutine lor(a,x,u)
use m_defs, only : dp,pi
implicit none
real (dp), intent(in) :: a,x
real (dp), intent(out) :: u
real (dp) :: r
logical :: done

done=.false.
do
 if (done) exit
 call random_number(r)
 u = x + a * tan( pi*(r-0.5d0) )
 call random_number(r)
 if (r<exp(-u**2)) done=.true.
end do

end subroutine lor

! sample gaussian and compare to lorentzian
! always incredibly slow

subroutine gau(a,x,u)
use m_defs, only: dp,pi
implicit none
real (dp), intent(in) :: a,x
real (dp), intent(out) :: u
real (dp) :: r1,r2,r3
logical :: done

done=.false.
do
 if (done) exit
 call random_number(r1)
 call random_number(r2)
 u=sqrt( log(1.d0/(1.d0-r1)) ) * cos(2.d0*pi*r2)
 call random_number(r3)
 if (r3< a**2/( (u-x)**2 + a**2 ) ) done=.true.
end do

end subroutine gau




! approximate the distribution as lorentzian plus gaussian distributions
! see derivation in my notes.
! the u<2 case is only approximate, should use rejection with comparison function exp(-u**2).
! the u>2 case should be good on the wing, where rejection (e.g. sme code) may take 100's of tries

subroutine sample_upar_approx(a,x,u)
use m_defs
use m_voigt
real (dp), intent(in) :: a,x
real (dp), intent(out) :: u
real (dp) :: H,r,p_lor,p_gau,um,sigma,r1,r2
integer :: nu,i

if (abs(x)<2.d0) then
  print *,"can't call sample_upar_approx with |x|<2"
  stop
endif

call voigt(a,x,H)

p_lor = exp(-x**2) / H
p_gau = 1.d0-p_lor

call random_number(r)
if (r<p_lor) then
  call random_number(r)
  u = x + a * tan( pi*(r-0.5d0) )
else
  if (x>0.d0) then
    um=(x-sqrt(x**2-4.d0))/2.d0
  else if (x<0.d0) then
    um=(x+sqrt(x**2-4.d0))/2.d0
  endif
  sigma=1.d0/sqrt( 2.d0 + 2.d0/( (um-x)**2 + a**2 ) - 4.d0*(um-x)**2/( (um-x)**2 + a**2 )**2 )
  call random_number(r1)
  call random_number(r2)
  u=um + sigma*sqrt(-2.d0*log(r1))*cos(2.d0*pi*r2)
endif

end subroutine sample_upar_approx

end module m_sample_uparallel
