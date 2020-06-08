module constants

  integer, parameter :: dp = kind(1.d0)
  real (dp), parameter :: pi = 3.1415926535
  real (dp), dimension(3, 3) :: identity = reshape((/1., 0., 0., 0., 1., 0., 0., 0., 1./), shape(identity))
  real (dp), dimension(6) :: reg
end module constants
