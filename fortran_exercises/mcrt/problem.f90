subroutine transf(vec, unitm, iunitm, coordm, icoordm)
  ! Returns the unit vector transformation matrices at a specified position
  use modules
  implicit none
  real (dp), dimension(3, 3), intent(out) :: unitm, iunitm, coordm, icoordm
  real (dp), dimension(3), intent(in) :: vec
  real (dp), dimension(3, 3) :: b, d
  real (dp) :: x1, x2, x3

  x1 = vec(1)
  x2 = vec(2)
  x3 = vec(3)
  b = identity
  d = identity

! UNIT VECTOR TRANSFORMATION MATRIX
  unitm = reshape((/ sin(x2)*cos(x3), sin(x2)*sin(x3),   cos(x2), &
                      cos(x2)*cos(x3), cos(x2)*sin(x3),  -sin(x2), &
                             -sin(x3),         cos(x3),     0.0d0  &
           /), shape(coordm))

  coordm = reshape((/ sin(x2)*cos(x3), x1*cos(x2)*cos(x3), -x1*sin(x2)*sin(x3), &
                      sin(x2)*sin(x3), x1*cos(x2)*sin(x3),  x1*sin(x2)*cos(x3), &
                              cos(x3),         x1*sin(x3),               0.0d0  &
           /), shape(coordm))
! Currently set to: SPHERICAL POLAR

  iunitm = unitm
  icoordm = coordm
  call gaussj(iunitm, 3, 3, b, 3, 3)
  call gaussj(icoordm, 3, 3, d, 3, 3)

end subroutine transf

subroutine beam(pos, dir)
  ! Determines how photons enter the simulation domain.
  use constants  
  implicit none
  real (dp), dimension(3), intent(inout) :: pos, dir
  real (dp) :: xi, zeta, b
  integer :: i

  call random_number(xi)
  call random_number(zeta)

  ! IMPACT PARAMETER
  b = 2.0d0

  pos(1) = reg(2, 1) - 1e-3 ! Set r coordinate to be rmax
  pos(2) = asin(b * sqrt(zeta) / reg(2, 1)) ! Distribute theta uniformly over the solid angle covered by the impact parameter
  pos(3) = xi * 2.0 * pi  ! Distribute phi uniformly over a circle

  dir = reshape((/-cos(pos(2)), sin(pos(2)), 0.d0/), shape(dir))
end subroutine beam

subroutine step(pos, dir, region, exitflag, nsteps)
  ! Have a single photon step to the location of its next scattering
  use constants
  implicit none
  real (dp), dimension(3), intent(inout) :: pos, dir
  real (dp), dimension(6), intent(in) :: region
  real (dp), dimension(3, 3) :: tran, itran, coord, icoord
  real (dp) :: xi, zeta, omega, psi, chi
  real (dp) :: x, y, z
  real (dp) :: tau, s
  integer, intent(inout) :: nsteps
  logical, intent(out) :: exitflag

  chi = 1e-2

  ! Generate an optical depth through which to travel
  call random_number(xi)
  tau = -log(1.d0 - xi)

  ! Update photon's position in cartesian coords
  call transf(pos, tran, itran, coord, icoord)
  pos = matmul(transpose(icoord), pos)
  dir = matmul(transpose(itran), dir)
  pos = pos + chi * tau * dir
  pos = matmul(transpose(coord), pos)
  dir = matmul(transpose(tran), dir)
  nsteps = nsteps + 1

  ! Check if photon is still within domain
  if (((region(1) .le. pos(1)) .and. (pos(1) .le. region(2)))        &
      .and. ((region(3) .le. pos(2)) .and. (pos(2) .le. region(4)))  &
      .and. ((region(5) .le. pos(3)) .and. (pos(3) .le. region(6)))) then

    ! Generate a new direction in cartesian coordinates
    call random_number(zeta)
    call random_number(omega)
    call random_number(psi)

    x = 2.d0 * zeta - 1.d0
    y = 2.d0 * omega - 1.d0
    z = 2.d0 * psi - 1.d0

    dir = reshape((/x, y, z/), shape(dir))
    dir = dir / norm2(dir)
    call transf(pos, tran, itran, coord, icoord)
    dir = matmul(transpose(tran), dir) ! Should still be normalized, right?
    exitflag = .false.

  else
    ! Step back to when the photon was within the domain
    ! Solve for the right distance to travel to reach r = rmax
    ! Travel that distance, update pos and dir vectors
    pos = pos - chi * tau * dir
    s = (region(2) - pos(1))/dir(1)
    pos = pos + s * dir
    
    ! Set exit flag so that main loop knows to save the outputs to a file
    exitflag = .true.
  endif
end subroutine step

program problem
  use constants
  implicit none
  real (dp), dimension(3) :: pos, dir
  logical :: exitflag = .false.
  integer :: i, n=100, nsteps=0

  ! GAS REGION SPECIFICATION (simple bounds in code units)
  ! x1min, x1max, x2min, x2max, x3min, x3max
  reg = reshape((/0.d0, 25.d0, 0.d0, pi, 0.d0, 2.*pi/), shape(reg))

  open(1, file='exit_photons.dat', status='replace')
  do i=1, n
    call beam(pos, dir)  
    print *, 'photon', i
    print *, 'incidence', pos
    do while (exitflag .eqv. .false.)
      call step(pos, dir, reg, exitflag, nsteps)
      print *, pos, dir, nsteps 
    end do

    write(1, *) pos, dir, nsteps
    exitflag = .false.
    nsteps = 0
  end do
end program

!! Write photon incidence instances to a file
!  open (1, file='vecs.dat', status='replace')
!  do i=1, n
!    call beam(pos, dir)
!    call transf(pos, tran, itran)
!    xyz_pos = matmul(transpose(tran), pos)
! 
!    write(1, *) pos, dir
!  end do
!  close (1)
