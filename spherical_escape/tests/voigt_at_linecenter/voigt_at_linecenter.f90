subroutine beam(pos, dir)

  ! Determines how photons enter the simulation domain.
  use m_defs, only : dp, pi
  implicit none
  real (dp), dimension(3), intent(inout) :: pos
  real (dp), dimension(2), intent(inout) :: dir
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

  pos = (/0.d0,  0.d0, 0.d0/)
  dir = (/acos(2.d0 * zeta - 1.d0), 2.d0 * pi * xi/)

end subroutine beam

subroutine step(pos, dir, exitflag, nsteps, d_tot, nu)

  use constants
  use m_defs, only : dp, pi, clight, kboltz
  use m_voigt
  use m_linelist, only : line
  real (dp) :: xi, zeta, omega, theta, phi
  real (dp) :: x, y, z, x1, y1, z1, x2, y2, z2, r2, theta2, phi2
  real (dp), dimension(3), intent(inout) :: pos
  real (dp), dimension(2), intent(inout) :: dir
  real (dp), intent(inout) ::  d_tot, nu
  real (dp) :: d, s, dot, position_magsq, displacement_magsq, doppwidth, a, dlam, H
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
  dlam = (nu-line%nu)/doppwidth
  call voigt(a,dlam,H)

  ! displacement vector
  d = -log(1.d0 - xi) / (tau * H)
  theta = dir(1)
  phi = dir(2)

  ! cartesian displacement vector
  x = d*sin(theta)*cos(phi)
  y = d*sin(theta)*sin(phi)
  z = d*cos(theta)
  
  ! cartesian position vector
  x1 = pos(1)*sin(pos(2))*cos(pos(3))
  y1 = pos(1)*sin(pos(2))*sin(pos(3))
  z1 = pos(1)*cos(pos(2))

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

    dir = (/theta, phi/)
    pos = (/r2, theta2, phi2/)
    d_tot = d_tot + d*(s - 1.d0)

    exitflag = .true.

  else
    pos = (/r2, theta2, phi2/)
    ! Scatter into a new direction
    dir = (/acos(2.d0 * zeta - 1.d0), 2.d0 * pi * omega/)
  endif

end subroutine step

program problem
  use constants
  use m_defs, only : dp
  use m_linelist
  implicit none
  real (dp), dimension(3) :: pos
  real (dp), dimension(2) :: dir
  real (dp) :: d_tot
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
      call beam(pos, dir)
      d_tot = 0.d0
      if (verbose) then
        print *, 'photon', i
      end if
      do while (exitflag .eqv. .false.)
        call step(pos, dir, exitflag, nsteps, d_tot, line%nu)
      end do

      write(1, *) pos, dir, nsteps, d_tot
      exitflag = .false.
      nsteps = 0
    end do
  end do
end program

