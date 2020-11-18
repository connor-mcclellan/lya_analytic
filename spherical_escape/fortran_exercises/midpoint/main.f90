program main
  use modules
  implicit none
  real (dp) :: a=0.0, b=10.0
  call ferror(30, a, b)  !compute fractional error for n=2, 4, 8, 16, 32 between bounds a and b
  stop                  !5 here means go up to 2^5 in n
end program main
