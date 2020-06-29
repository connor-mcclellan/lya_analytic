! define constants, derived variable types, operations on those variables, etc.

module m_defs
implicit none

integer, parameter :: dp = kind(1.d0)
integer, parameter :: longint = selected_int_kind(16)

real (dp), parameter :: pi=2.d0*asin(1.d0)
real (dp), parameter :: twopi=2.d0*pi

real (dp), parameter :: cgrav=6.67429d-8
real (dp), parameter :: hplanck=6.62606896d-27
real (dp), parameter :: clight=2.997924589d10
real (dp), parameter :: clightkms=2.997924589d5
real (dp), parameter :: kboltz=1.3806504d-16
real (dp), parameter :: charge=4.80320427d-10
real (dp), parameter :: abohr=0.52917720859d-8
real (dp), parameter :: melectron= 9.10938215d-28
real (dp), parameter :: mproton=1.672621637d-24
real (dp), parameter :: amu=1.660538782d-24
real (dp), parameter :: eV = 1.602176487d-12
real (dp), parameter :: angstrom = 1.d-8

real (dp), parameter :: pc = 3.0857e18
real (dp), parameter :: au = 1.49597870700d13
real (dp), parameter :: msun = 1.9884d33
real (dp), parameter :: rsun = 6.9551d10
real (dp), parameter :: mjup = 1.8987d30
real (dp), parameter :: rjup = 7.1492d9

real (dp), parameter :: day = 86400.d0
real (dp), parameter :: kms = 1.d5

logical :: dbg

real (dp), parameter :: small=1.d-6

type cartesian_vector
	real (dp) :: x,y,z
end type cartesian_vector

! can't say "use m_defs, only: +,*
! how can we do that? what if we only want some of variables in this module,
! rather than all?

interface operator (+)			!overload these operators to use with our derived types
  module procedure add_vectors
end interface
interface operator (-)
  module procedure subtract_vectors
end interface
interface operator (*)
  module procedure mult_scalar_vector
end interface
interface operator (.dot.)
  module procedure cartesian_dot_product
end interface

type spherical_vector
	real (dp) :: r,theta,phi
end type spherical_vector

type photon_spherical
        real (dp) :: nu				! frequency
	type (cartesian_vector) :: xc		! x,y,z cartesian position
        type (spherical_vector) :: xs   	! r,theta,phi spherical position
        type (cartesian_vector) :: nc   	! x,y,z cartesian direction unit vector
        type (spherical_vector) :: ns   	! projection of direction vector along
						! spherical basis vectors,
						! which depends on position
	real (dp) :: theta,phi			! direction angles relative to cartesian axes
	real (dp) :: stokes(4)  		! stokes vector
	type (cartesian_vector) :: e1,e2	! polarization directions
end type photon_spherical


contains

function add_vectors(a,b) result (c)
implicit none
type (cartesian_vector), intent (in) :: a,b
type (cartesian_vector) :: c
c%x=a%x+b%x
c%y=a%y+b%y
c%z=a%z+b%z
end function

function subtract_vectors(a,b) result (c)
implicit none
type (cartesian_vector), intent (in) :: a,b
type (cartesian_vector) :: c
c%x=a%x-b%x
c%y=a%y-b%y
c%z=a%z-b%z
end function

function mult_scalar_vector(a,b) result (c)
implicit none
real (dp), intent (in) :: a
type (cartesian_vector), intent (in) :: b
type (cartesian_vector) :: c
c%x=a*b%x
c%y=a*b%y
c%z=a*b%z
end function

function cartesian_dot_product(a,b) result (c)
implicit none
type (cartesian_vector), intent (in) :: a,b
real (dp) :: c
c = a%x*b%x + a%y*b%y + a%z*b%z
end function

end module m_defs
