subroutine beam(p)

  ! Determines how photons enter the simulation domain.
  use m_defs, only : dp, pi, photon_spherical, spherical_vector, cartesian_vector
  use m_util
  use m_linelist, only : line
  implicit none
  type (photon_spherical), intent(inout) :: p
  real (dp) :: xi, zeta, b
  integer :: i

!!! INCIDENT ON POSITIVE Z HEMISPHERE IN NEGATIVE Z DIRECTION
!  b=1.d0
! 
!  call random_number(xi)
!  call random_number(zeta)
! 
!  pos(1) = 1.d0 - 1e-3 ! Set r coordinate to be rmax
!  pos(2) = asin(b * sqrt(zeta))
!  if (pos(2) .lt. 0.d0) then
!    pos(2) = pos(2) + pi
!  endif
!  pos(3) = xi * 2.0 * pi  ! Distribute phi uniformly over a circle
! 
!  ! Unit vector pointing in negative z direction
!  dir = (/pi, 0.d0/)

!!! PHOTONS RELEASED AT CENTER
  call random_number(xi)
  call random_number(zeta)

  p%nu = line%nu

  p%theta = acos(2.d0 * zeta - 1.d0)
  p%phi = 2.d0 * pi * xi

  p%xs%r = 0.d0
  p%xs%theta = 0.d0
  p%xs%phi = 0.d0

  p%nc%x=sin(p%theta)*cos(p%phi)
  p%nc%y=sin(p%theta)*sin(p%phi)
  p%nc%z=cos(p%theta)
  
  call convert_cart_vector_to_sph_vector(p%nc%x,p%nc%y,p%nc%z,p%xs%theta,p%xs%phi,p%ns%r,p%ns%theta,p%ns%phi)

end subroutine beam

subroutine step(p, exitflag, nsteps, d_tot)

  ! This is a first rewrite of my original step subroutine, aiming to integrate Phil's photon_spherical
  ! class with as few modifications to my original code as possible. It's super inefficient and neds to
  ! be cleaned up.
  use constants
  use m_defs, only : dp, pi, clight, kboltz, photon_spherical, spherical_vector, cartesian_vector
  use m_voigt
  use m_util
  use m_redistribute_unpol
  use m_linelist, only : line
  implicit none
  type (photon_spherical), intent(inout) :: p
  real (dp) :: xi, zeta, omega, theta, phi
  real (dp) :: x, y, z, x1, y1, z1, x2, y2, z2, r2, theta2, phi2
  real (dp), intent(inout) ::  d_tot
  real (dp) :: d, s, dot, position_magsq, displacement_magsq, doppwidth, a, dlam, H, vth
  integer, intent(inout) :: nsteps
  logical, intent(inout) :: exitflag

  ! random numbers
  call random_number(xi)
  call random_number(zeta)
  call random_number(omega)

  ! thermal velocity of H atoms
  vth = sqrt(2.d0 * kboltz * T / line%mass)

  ! cross-section at this frequency
  doppwidth = line%nu * vth / clight
  a = line%lorwidth / doppwidth
  dlam = (p%nu-line%nu)/doppwidth
  call voigt(a,dlam,H)

  ! displacement vector
  d = -log(1.d0 - xi) / (tau * H)

  ! cartesian displacement vector
  x = d*sin(p%theta)*cos(p%phi)
  y = d*sin(p%theta)*sin(p%phi)
  z = d*cos(p%theta)
  
  ! cartesian position vector
  x1 = p%xs%r*sin(p%xs%theta)*cos(p%xs%phi)
  y1 = p%xs%r*sin(p%xs%theta)*sin(p%xs%phi)
  z1 = p%xs%r*cos(p%xs%theta)

  ! new cartesian position vector
  x2 = x + x1
  y2 = y + y1
  z2 = z + z1

  ! new position vector
  r2 = norm2((/x2, y2, z2/))
  theta2 = acos(z2 / r2)
  phi2 = atan2(y2, x2)
  if (phi2 .lt. 0.d0) then
    phi2 = phi2 + 2.d0*pi   
  endif
  
  nsteps = nsteps + 1
  d_tot = d_tot + d

  if (r2 .gt. 1.d0) then

    ! Solve for the right distance to travel to reach r = rmax
    dot = (x*x1 + y*y1 + z*z1)
    position_magsq = norm2((/x1, y1, z1/))**2.d0
    displacement_magsq = norm2((/x, y, z/))**2.d0
    
    s = (-dot + sqrt(dot**2.d0 - displacement_magsq*(position_magsq - 1.d0**2.d0)))/displacement_magsq

    ! travel that distance
    x2 = s*x + x1
    y2 = s*y + y1
    z2 = s*z + z1
  
    ! new position vector
    r2 = norm2((/x2, y2, z2/))
    theta2 = acos(z2 / r2)
    phi2 = atan2(y2, x2)
    if (phi2 .lt. 0.d0) then
      phi2 = phi2 + 2.d0*pi   
    endif

    ! Update photon attributes
    p%xs%r = r2
    p%xs%theta = theta2
    p%xs%phi = phi2
    d_tot = d_tot + d*(s - 1.d0)
    call convert_cart_vector_to_sph_vector(p%nc%x,p%nc%y,p%nc%z,p%xs%theta,p%xs%phi,p%ns%r,p%ns%theta,p%ns%phi)

    exitflag = .true.

  else
    ! Update position
    p%xs%r = r2
    p%xs%theta = theta2
    p%xs%phi = phi2

    ! Scatter into a new direction
    call redistribute_wrapper(T, p)
  endif

end subroutine step

program problem
  use constants
  use m_defs, only : dp, photon_spherical
  use m_linelist
  implicit none
  real (dp) :: d_tot
  type (photon_spherical) :: p
  logical :: exitflag = .false., verbose
  integer :: i, j, n=1000000, nsteps=0
  character(len=64) :: filenum

  T = 1.d4
  call make_line_data

  ! Set range of tau here
  do j=10, 10
    print *, 'j=', j
    tau = float(j)
    verbose = .true.

    write (filenum, *) int(tau)
    open(1, file='./outputs/exit_photons_tau'//trim(adjustl(filenum))//'.dat', status='replace')
    do i=1, n
      call beam(p)
      d_tot = 0.d0
      if (verbose) then
        print *, 'photon', i
      end if
      do while (exitflag .eqv. .false.)
        call step(p, exitflag, nsteps, d_tot)
      end do

      write(1, *) p%xs%r, p%xs%theta, p%xs%phi, p%theta, p%phi, nsteps, d_tot, p%nu
      exitflag = .false.
      nsteps = 0
    end do
  end do
end program

