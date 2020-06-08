subroutine transf(vec, unitm, iunitm)

  ! Returns the unit vector transformation matrices at a specified position
  use modules
  implicit none
  real (dp), dimension(3, 3), intent(out) :: unitm, iunitm
  real (dp), dimension(3), intent(in) :: vec
  real (dp), dimension(3, 3) :: d
  real (dp) :: x1, x2, x3

  x1 = vec(1)
  x2 = vec(2)
  x3 = vec(3)
  d = identity

! UNIT VECTOR TRANSFORMATION MATRIX
  unitm = reshape((/ sin(x2)*cos(x3), sin(x2)*sin(x3),   cos(x2), &
                     cos(x2)*cos(x3), cos(x2)*sin(x3),  -sin(x2), &
                            -sin(x3),         cos(x3),     0.0d0  &
           /), shape(unitm))

  iunitm = unitm
  call gaussj(iunitm, 3, 3, d, 3, 3)

end subroutine transf

subroutine beam(pos, dir)

  ! Determines how photons enter the simulation domain.
  use constants  
  implicit none
  real (dp), dimension(3), intent(inout) :: pos, dir
  real (dp) :: xi, zeta
  integer :: i

  call random_number(xi)
  call random_number(zeta)

  pos(1) = reg(2) - 1e-3 ! Set r coordinate to be rmax
  pos(2) = asin(b * sqrt(zeta) / reg(2))
  if (pos(2) .lt. 0.d0) then
    pos(2) = pos(2) + pi
  endif
  pos(3) = xi * 2.0 * pi  ! Distribute phi uniformly over a circle

  ! Unit vector pointing in negative z direction
  dir = reshape((/-cos(pos(2)), sin(pos(2)), 0.d0/), shape(dir))

end subroutine beam

subroutine step(pos, dir, region, exitflag, nsteps)

  ! Have a single photon step to the location of its next scattering
  use constants
  implicit none
  real (dp), dimension(3), intent(inout) :: pos, dir
  real (dp), dimension(6), intent(in) :: region
  real (dp), dimension(3, 3) :: tran, itran
  real (dp), dimension(3) :: cpos, cdir, dist_test
  real (dp) :: xi, zeta, omega, psi
  real (dp) :: x, y, z
  real (dp) :: tau, s, dot, magsq
  integer :: i, j
  integer, intent(inout) :: nsteps
  logical, intent(out) :: exitflag

  ! Generate an optical depth through which to travel
  call random_number(xi)
  tau = -log(1.d0 - xi)

  ! Change photon's position to cartesian coords
  cpos = ((/pos(1)*sin(pos(2))*cos(pos(3)), &
            pos(1)*sin(pos(2))*sin(pos(3)), &
                        pos(1)*cos(pos(2))/))

  ! Transform direction unit vector to cartesian coords
  call transf(pos, tran, itran)
  cdir = matmul(transpose(itran), dir)

  ! Check to see how far the photon is from the origin after this step
  dist_test = cpos + chi * tau * cdir
  nsteps = nsteps + 1

  if (norm2(dist_test) .le. region(2)) then
    ! Update the photon's position (cartesian)
    cpos = cpos + chi * tau * cdir

    ! Transform photon's position back to spherical polars
    pos(1) = norm2(cpos)
    pos(2) = acos(cpos(2) / pos(1))
    pos(3) = atan2(cpos(2), cpos(1))
    if (pos(3) .lt. 0.d0) then
      pos(3) = pos(3) + 2.d0*pi   
    endif

    ! Generate a new direction in cartesian coordinates
    call random_number(zeta)
    call random_number(omega)
    call random_number(psi)

    x = 2.d0 * zeta - 1.d0
    y = 2.d0 * omega - 1.d0
    z = 2.d0 * psi - 1.d0

    ! Construct direction unit ve ctor
    cdir = reshape((/x, y, z/), shape(cdir))
    cdir = cdir / norm2(cdir)

    ! Transform cartesian direction vector to spherical polars
    call transf(pos, tran, itran)
    dir = matmul(transpose(tran), cdir)

    ! Tell the loop that the photon has yet to exit the domain
    exitflag = .false.

  else
    ! Solve for the right distance to travel to reach r = rmax
    dot = dot_product(cpos, cdir)
    magsq = norm2(cpos)**2.d0
    s = (-dot + sqrt(dot**2.d0 - 4.d0*(magsq - region(2)**2.d0)))/2.d0

    ! Travel that distance, update pos and dir vectors in cartesian coordinates
    cpos = cpos + s * cdir

    ! Transform position vector to spherical polar coordinates
    pos(1) = norm2(cpos)
    pos(2) = acos(cpos(2) / pos(1))
    pos(3) = atan2(cpos(2), cpos(1))
    if (pos(3) .lt. 0.d0) then
      pos(3) = pos(3) + 2.d0*pi
    endif

    ! Output cartesian direction vector
    dir = cdir    
    
    ! Set exit flag so that main loop knows to save the outputs to a file
    exitflag = .true.

  endif

end subroutine step

program problem
  use constants
  implicit none
  real (dp), dimension(3) :: pos, dir
  logical :: exitflag = .false., verbose
  integer :: i, n=100000, nsteps=0

  ! GAS REGION SPECIFICATION (simple bounds in code units)
  ! x1min, x1max, x2min, x2max, x3min, x3max
  reg = reshape((/0.d0, 20.d0, 0.d0, pi, 0.d0, 2.*pi/), shape(reg))
  ! Right now, only x1max matters

  ! IMPACT PARAMETER
  b = 20.d0

  ! distance traveled = chi * tau
  chi = 1e-2

  verbose = .false.

  open(1, file='exit_photons.dat', status='replace')
  do i=1, n
    call beam(pos, dir)
    if (verbose) then
      print *, 'photon', i
      print *, 'incidence', pos
    end if
    do while (exitflag .eqv. .false.)
      call step(pos, dir, reg, exitflag, nsteps)
      if (verbose) then
        print *, pos, dir, nsteps 
      endif
    end do

    write(1, *) pos, dir, nsteps
    exitflag = .false.
    nsteps = 0
  end do
end program

