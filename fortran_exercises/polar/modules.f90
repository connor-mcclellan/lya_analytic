! Purpose: to evenly sample impact parameters around a central point in polar coordinates
! Uses Monte Carlo and outputs a file of coordinates to demonstrate uniformity
! xi and zeta are two independent random numbers between 0 and 1

module modules
  use constants
  contains

    subroutine polar(R, b, theta)
      implicit none
      real (dp), intent(in) :: R
      real (dp), intent(out) :: b, theta
      real (dp) :: xi, zeta

      call random_number(xi)
      b = R * sqrt(xi)

      call random_number(zeta)
      theta = zeta * 2.0 * pi

    end subroutine polar
end module modules
