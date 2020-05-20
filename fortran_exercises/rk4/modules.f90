module modules

use constants

contains

  subroutine derivs(x, y, dydx)
    implicit none
    real (dp), intent(in) :: x
    real (dp), intent(in), dimension(2) :: y
    real (dp), intent(out), dimension(2) :: dydx

    dydx(1) = y(2)
    dydx(2) = - omega0**2 * y(1)
  end subroutine derivs

  subroutine rk4step(x, y, h)
    implicit none
    real (dp), intent(inout) :: x, y(:)
    real (dp), intent(in) :: h
    real (dp), dimension(size(y)) :: k1, k2, k3, k4, dydx

    call derivs(x, y, dydx)
    k1 = h*dydx
    call derivs(x + 0.5*h, y + 0.5*k1, dydx)
    k2 = h*dydx
    call derivs(x + 0.5*h, y + 0.5*k2, dydx)
    k3 = h*dydx
    call derivs(x+h, y+k3, dydx)
    k4 = h*dydx
    y = y + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
    x = x + h
  end subroutine rk4step

  subroutine odeint(x, xmax, y, dx)
    implicit none
    real (dp), intent(inout) :: y(:), x
    real (dp), intent(in) :: xmax, dx
    real (dp), dimension(2) :: analytic !comment out eventually

    open (1, file='sol.dat', status='replace')

    do
      call rk4step(x, y, dx)
      analytic = (/cos(omega0*x), -omega0*sin(omega0*x)/) !comment out eventually
      write(1, *) x, y, analytic !comment out eventually
      if (x>xmax) exit
    end do
  end subroutine odeint
end module modules
