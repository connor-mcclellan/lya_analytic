module modules

implicit none
integer, parameter :: dp = kind(4)

contains

  function func(x) result (value)
    implicit none
    real (dp), intent(in) :: x
    real (dp) :: value
    value = exp(-x)
  end function func


  subroutine midpoint(n, a, b, integral)
    implicit none
    integer, intent(in) :: n
    real (dp), intent(in) :: a, b
    real (dp), intent(out) :: integral
    real (dp) :: x
    integer :: i

    integral=0.0

    do i=0,n-1
      x = (b-a)*(i+0.5)/n
      !write(*,*) x, func(x)                    !Uncomment to display steps and y values
      integral = integral + func(x)*(b-a)/n
    end do

  end subroutine midpoint


  subroutine ferror(npowers, a, b)
    implicit none
    integer, intent(in) :: npowers !Number of powers of two (values for n to take)
    real (dp), intent(in) :: a, b
    integer :: i, n
    real (dp) :: numerical, analytic, error
    
    open (1, file='error.dat', status='replace')
    write(1, '(a12,2X,a15)') 'n', 'err'

    do i=0, npowers
      n = 2.0**i
      call midpoint(n, a, b, numerical)
      analytic = -func(b) + func(a)
      error = abs(numerical - analytic)/analytic
      write(1,*) n, error
    end do 

    close(1)

  end subroutine ferror

end module
