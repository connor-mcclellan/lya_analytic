! Problem: column of light incident on a sphere of uniform optical depth.
! Monochromatic photons come in along the z-axis with a random impact parameter

! Scatter photons around until they leave the sphere. What are their exit 
! angles, and what frequency do they leave with?

! a photon has a pos (position) and dir (direction) array. Coordinates are
! arbitrary until specified in the problem file

! reg: an array that specifies a region. Format is ((x1min, x1max), (x2min, x2max), (x3min, x3max))

module modules

  use constants

  contains

    subroutine gaussj(a,n,np,b,m,mp)
      implicit none
      integer m,mp,n,np,NMAX
      real (dp) :: a(np,np), b(np,mp)
      parameter (NMAX=50)
      integer i,icol,irow,j,k,l,ll,indxc(NMAX),indxr(NMAX),ipiv(NMAX)
      real (dp) big,dum,pivinv

      do j=1,n
        ipiv(j)=0
      enddo

      do i=1,n
        big=0.d0
        do j=1,n
          if(ipiv(j).ne.1)then
            do k=1,n
              if (ipiv(k).eq.0) then
                if (abs(a(j,k)).ge.big)then
                  big=abs(a(j,k))
                  irow=j
                  icol=k
                endif
              endif
            enddo
          endif
        enddo
        ipiv(icol)=ipiv(icol)+1
        if (irow.ne.icol) then
          do l=1,n
            dum=a(irow,l)
            a(irow,l)=a(icol,l)
            a(icol,l)=dum
          enddo
          do l=1,m
            dum=b(irow,l)
            b(irow,l)=b(icol,l)
            b(icol,l)=dum
          enddo
        endif
        indxr(i)=irow
        indxc(i)=icol
        pivinv=1.d0/a(icol,icol)
        a(icol,icol)=1.d0
        do l=1,n
          a(icol,l)=a(icol,l)*pivinv
        enddo
        do l=1,m
          b(icol,l)=b(icol,l)*pivinv
        enddo
        do ll=1,n
          if(ll.ne.icol)then
            dum=a(ll,icol)
            a(ll,icol)=0.d0
            do l=1,n
              a(ll,l)=a(ll,l)-a(icol,l)*dum
            enddo
            do l=1,m
              b(ll,l)=b(ll,l)-b(icol,l)*dum
            enddo
          endif
        enddo
      enddo
      do l=n,1,-1
        if(indxr(l).ne.indxc(l))then
          do k=1,n
            dum=a(k,indxr(l))
            a(k,indxr(l))=a(k,indxc(l))
            a(k,indxc(l))=dum
          enddo
        endif
      enddo
      return
    end subroutine gaussj

end module
