! inverse error function. not using this in anything now.

module m_erfinv

contains


! Giles M. Approximating the erfinv function. GPU Computing Gems. 2011;2:109–16
!download from
!https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
! this is single precision version given in paper. paper does not give
! coeff for double precision version.

real (dp) function erfinv(x)
use m_defs, only : dp,pi
implicit none
real (dp), intent (in) :: x
real (dp) :: w,p

w=-log( (1.d0-x)*(1.d0+x) )

if (w<5.d0) then
  w = w - 2.5d0
  p =  2.81022636d-08
  p =  3.43273939d-7 + p*w
  p = -3.5233877d-6 + p*w
  p = -4.39150654d-6 + p*w
  p =  0.00021858087d0 + p*w
  p = -0.00125372503d0 + p*w
  p = -0.00417768164d0 + p*w
  p =  0.246640727d0 + p*w
  p =  1.50140941d0 + p*w
else
  w = sqrt(w) - 3.d0
  p =  -0.000200214257d0
  p = 0.000100950558d0 + p*w
  p = 0.00134934322d0 + p*w
  p = -0.00367342844d0 + p*w
  p = 0.00573950773d0 + p*w
  p = -0.0076224613d0 + p*w
  p = 0.00943887047d0 + p*w
  p = 1.00167406d0 + p*w
  p = 2.83297682d0 + p*w
endif
erfinv=p*x

end function erfinv



! this version found at 
! https://stackoverflow.com/questions/5971830/need-code-for-inverse-error-function

real (dp) function v2_erfinv(y)
use m_defs, only : dp,pi
implicit none
real (dp), intent (in) :: y
real (dp) :: x
real (dp), parameter, dimension (4) :: &
a=(/ 0.886226899d0, -1.645349621d0,  0.914624893d0, -0.140543331d0 /),&
b=(/ -2.118377725d0,  1.442710462d0, -0.329097515d0,  0.012229801d0 /),&
c=(/ -1.970840454d0, -1.624906493d0,  3.429567803d0,  1.641345311d0 /)
real (dp), parameter, dimension (2) :: d=(/ 3.543889200d0,  1.637067800d0 /)
real (dp) :: y0,z

y0 = 0.7

if (y<=-1.d0 .or. y>=1.d0) then
  print *, "out of range"
  stop
endif

if (y<-y0) then
  z = sqrt(-log((1.d0+y)/2.d0))
  x = -(((c(4)*z+c(3))*z+c(2))*z+c(1))/((d(2)*z+d(1))*z+1.d0)
else if (y>=-y0 .and. y<y0) then
  z = y**2
  x = y*(((a(4)*z+a(3))*z+a(2))*z+a(1))/((((b(4)*z+b(4))*z+b(2))*z+b(1))*z+1.d0)
else if (y>y0) then
  z = sqrt(-log((1.d0-y)/2.d0))
  x = (((c(4)*z+c(3))*z+c(2))*z+c(1))/((d(2)*z+d(1))*z+1.d0)
endif
x = x - (erf(x) - y) / (2.d0/sqrt(pi) * exp(-x**2))
x = x - (erf(x) - y) / (2.d0/sqrt(pi) * exp(-x**2))
v2_erfinv=x

end function v2_erfinv



end module m_erfinv
