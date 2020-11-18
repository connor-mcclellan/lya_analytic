program multest
  implicit none
  real, dimension(3) :: a
  real, dimension(3, 3) :: m
  integer :: i, j

  m = reshape((/1., 3., 3., 4., 6., 6., 3., 4., 6./), shape(m))
  a = reshape((/0., 0., 1./), shape(a))

  do i=1,3
    print *, (m(j, i), j=1,3)
  end do

  print *, a

  print *, matmul(transpose(m), a)
end program
  
