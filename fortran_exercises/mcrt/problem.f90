subroutine transf(vec, matrix, imatrix)
  ! Returns the transformation matrices at a specified position in code 
  ! coordinates
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

! COORDINATE TRANSFORMATION MATRIX
  matrix = reshape((/ sin(x2)*cos(x3), x1*cos(x2)*cos(x3), -x1*sin(x2)*sin(x3), &
                      sin(x2)*sin(x3), x1*cos(x2)*sin(x3),  x1*sin(x2)*cos(x3), &
                      cos(x2),                -x1*sin(x2),               0.0d0  &             
           /), shape(matrix))
! Currently set to: SPHERICAL POLAR
! Code coordinate = matrix * Cartesian coordinate

  imatrix = matrix
  call gaussj(imatrix, 3, 3, b, 3, 3)

end subroutine transf

subroutine beam(pos, dir)
  ! Determines how photons enter the simulation domain.
  use constants  
  real (dp), dimension(3), intent(out) :: pos, dir
  real (dp), dimension(3, 3) :: tran, itran
  real (dp), dimension(3) :: dir_cartesian
  real (dp) :: xi, zeta, b
  integer :: i

  call random_number(xi)
  call random_number(zeta)

  ! IMPACT PARAMETER
  b = 2.0d0

  pos(1) = reg(2, 1) ! Set r coordinate to be rmax
  pos(2) = acos(b * sqrt(zeta) / reg(2, 1)) ! Distribute theta uniformly over the solid angle covered by the impact parameter
  pos(3) = xi * 2.0 * pi  ! Distribute phi uniformly over a circle

  dir_cartesian = reshape((/0.d0, 0.d0, -1.d0/), shape(dir_cartesian)) ! Photons incoming from positive z
  call transf(pos, tran, itran) ! Calculate the transformation matrix at pos

  do i=1, 3
    print *, (tran(j, i), j=1,3)
  end do

  dir = matmul(transpose(tran), dir_cartesian) ! Transform dir to spherical polars
  dir = dir / norm2(dir)
  
end subroutine beam

program problem
  use constants
  implicit none
  real (dp), dimension(3) :: pos, dir
  integer :: i

  ! GAS REGION SPECIFICATION (simple bounds in code units)
  ! x1min, x1max, x2min, x2max, x3min, x3max
  reg = reshape((/0.d0, 5.d0, 0.d0, pi, 0.d0, 2.*pi/), shape(reg))

  call beam(pos, dir)

  do i=1, 3
    print *, pos(i)
  end do

  do i=1, 3
    print *, dir(i)
  end do
end program
