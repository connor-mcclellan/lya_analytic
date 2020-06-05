subroutine transf(vec, matrix, imatrix)
  ! Returns the unit vector transformation matrices at a specified position
  use modules
  implicit none
  real (dp), dimension(3, 3), intent(out) :: matrix, imatrix
  real (dp), dimension(3), intent(in) :: vec
  real (dp), dimension(3, 3) :: b, d
  real (dp) :: x1, x2, x3

  x1 = vec(1)
  x2 = vec(2)
  x3 = vec(3)
  b = identity
  d = identity

! UNIT VECTOR TRANSFORMATION MATRIX
  matrix = reshape((/ sin(x2)*cos(x3), sin(x2)*sin(x3),   cos(x2), &
                      cos(x2)*cos(x3), cos(x2)*sin(x3),  -sin(x2), &
                              sin(x3),         cos(x3),     0.0d0  &
           /), shape(matrix))
! Currently set to: SPHERICAL POLAR

  imatrix = matrix
  call gaussj(imatrix, 3, 3, b, 3, 3)

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

  pos(1) = reg(2, 1) ! Set r coordinate to be rmax
  pos(2) = asin(b * sqrt(zeta) / reg(2, 1)) ! Distribute theta uniformly over the solid angle covered by the impact parameter
  pos(3) = xi * 2.0 * pi  ! Distribute phi uniformly over a circle

  dir = reshape((/-cos(pos(2)), sin(pos(2)), 0.d0/), shape(dir))
end subroutine beam

subroutine step(pos, dir, region, exitflag)
  ! Have a single photon step to the location of its next scattering
  use constants
  implicit none
  real (dp), dimension(3), intent(inout) :: pos, dir
  real (dp), dimension(6), intent(in) :: region
  real (dp), dimension(3, 3) :: tran, itran
  real (dp) :: xi, zeta, omega, psi
  real (dp) :: x, y, z
  real (dp) :: tau, s
  logical, intent(out) :: exitflag

  ! Generate an optical depth through which to travel
  call random_number(xi)
  tau = -log(1.d0 - xi)

  ! Update photon's position in spherical polars
  pos = pos + tau * dir

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
    call transf(pos, tran, itran)
    dir = matmul(transpose(tran), dir) ! Should still be normalized, right?
    exitflag = .false.

  else
    ! Step back to when the photon was within the domain
    ! Solve for the right distance to travel to reach r = rmax
    ! Travel that distance, update pos and dir vectors
    pos = pos - tau * dir
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
  integer :: i, n=10000

  ! GAS REGION SPECIFICATION (simple bounds in code units)
  ! x1min, x1max, x2min, x2max, x3min, x3max
  reg = reshape((/0.d0, 5.d0, 0.d0, pi, 0.d0, 2.*pi/), shape(reg))

  open(1, file='exit_photons.dat', status='replace')
  do i=1, n
    call beam(pos, dir)  

    do while (exitflag .eqv. .false.)
      call step(pos, dir, reg, exitflag)
    end do

    write(1, *) pos, dir
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
