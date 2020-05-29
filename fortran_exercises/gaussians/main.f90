program main

  use modules

  implicit none
  integer :: n = 100000
  integer :: i
  real (dp) :: mu, sigma, offset, x

  mu = 0.0
  sigma = 1.0
  offset = 2.0

  open (1, file='gauss.dat', status='replace')
  write(1, *) 'x'

  do i=0, n-1
    call gaussian(mu, sigma, offset, x)
    write(1, *) x
  end do

  close(1)
end program main
