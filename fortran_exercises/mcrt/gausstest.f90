program gausstest
  use constants
  use modules
  real (dp), dimension(3, 3) :: a
  real (dp), dimension(3, 3) :: b, c
  integer :: i, j
  a = reshape((/cos(1.d0), 2.d0, 2, 4, 5, 6, 7, 8, 9/), shape(a))
  b = reshape((/1, 0, 0, 0, 1, 0, 0, 0, 1/), shape(b))

  c = b

  do i=1, 3
    print *, (a(j, i), j=1, 3)
  end do

  call gaussj(a, 3, 3, c, 3, 3)
  
  do i=1, 3
    print *, (a(j, i), j=1, 3)
  end do

  do i=1, 3
    print *, (b(j, i), j=1, 3)
  end do

end program
