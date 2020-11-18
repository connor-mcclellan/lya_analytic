program main

  use constants
  use modules

  implicit none

  real (dp), dimension(2) :: y = (/1.0, 0.0/)
  real (dp) :: x = 0.0, dx = 1.0e-4
  real (dp) :: xmax

  omega0 = 1.0
  xmax = 10.0*2.*pi/omega0

  call odeint(x, xmax, y, dx)
  write(*, *) y
end program main
