program main

  use modules

  implicit none
  integer :: n = 10000
  integer :: i
  real (dp) :: b, theta, R=5.0

  open (1, file='coords.dat', status='replace')
  write(1, '(a4,2X,a28)') 'b', 'theta'

  do i=0, n-1
    call polar(R, b, theta)
    write(1, *) b, theta
  end do

  close(1)
end program main
