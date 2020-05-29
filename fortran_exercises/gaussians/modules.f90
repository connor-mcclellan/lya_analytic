module modules

use constants

contains

  subroutine erfi(z, x)
    implicit none
    real (dp), intent(in) :: z
    real (dp), intent(out) :: x

    x = 0.5 * sqrt(pi) * (z + pi/12.*z**3.0 + 7.*pi**2.*z**5./480. + 127.*pi**3.*z**7./40320. + 4369.*pi**4.*z**9./5806080.)
  end subroutine erfi


  subroutine gaussian(mu, sigma, offset, x)
    implicit none
    real (dp), intent(in) :: mu, sigma, offset
    real (dp), intent(out) :: x
    real (dp) :: xi, coinflip, alpha, pm

    ! PDF: 1.0 / sigma / sqrt(2.0*pi) * exp(-0.5 * ((x-mu)/sigma)**2.0)
    
    call random_number(coinflip)
    if (coinflip .le. 0.5) then 
      pm = -offset
    else
      pm = offset
    endif

    call random_number(xi)
    call erfi(2.0*xi - 1.0, alpha)
    x = alpha * sigma * sqrt(2.0) + (mu + pm)

  end subroutine gaussian
end module modules
