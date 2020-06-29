! info for lyman alpha line

module m_linelist

use m_defs, only : dp
type singlet
  character (len=20) :: name
  real (dp) :: lambda,f,gamma,mass
  real (dp) :: nu,omega,lorwidth
end type singlet
type (singlet) :: line


contains


! atomic data. treats lines as singlets.

subroutine make_line_data
use m_defs, only : angstrom,pi,clight,amu
implicit none
integer :: i

! H lyman alpha data from morton 1991.
line%name = "HLya"
line%lambda= 1215.6701	! wavelength in angstroms.
line%f=  0.4164		! oscillator strength
line%gamma= 6.265d8	! equals einstein A = probability decay rate, in 1/sec.
line%mass = amu		! mass of scatterer
line%nu = clight / (line%lambda * angstrom)	! frequencies in cycles per second
line%omega=2.d0*pi*line%nu			! frequencies in radians per second
line%lorwidth = line%gamma / (4.d0*pi)	! width appearing in Lorenztian denominator

end subroutine make_line_data


end module m_linelist
