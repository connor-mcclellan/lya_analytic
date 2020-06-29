	complex function zeta(arg)
c
c DEC-10 version of NFMECC Libris abstract p22.
c Numerical calculation of the plasma dispersion function.
c     COMPLEX ZETA, DZETA, ARG
c     ZETA(ARG) gives plasma dispersion function
c     DZETA(ARG) gives its derivative.
c
c Authors- Bill Sharp modified for PDP-10 by C. F. F. Karney.
c (This code has evolved from something Gary Swanson wrote in the
c mid-60's!  Hence the archaic fortran branch commands.)
c Modified for the VAX by Greg Hammett (also fixed the asymptotic 
c formula choice of imaginary part and put in an improved power series
c solution.)  This routine has been checked against the tables in
c Fried-Conte.
c 
c     This routine computes the fried-conte plasma dispersion function,
c f(z), where
c       f(z) = 1 / sqrt(pi) * def. integral from minus infinity
c       infinity of exp( -t * t) / (t - z) dt.
c an equivalent formulation is
c       f(z) = sqrt(pi) * i * w(z)
c       where
c       w(z) = exp(-z * z) * erfc( -i * z) .
c here erfc (z) = complimentary error function given by
c       erfc (z) = 2/sqrt(pi) * def. integral from z to infinity of
c       exp(- t*t) dt.
c there are three different approximation methods used depending on the
c value of the argument of the function.  the methods are:
c       1. continued fraction method: abs(y).ge.1.0
c       2. asymtotic series method: abs(x).ge.4.0 and abs(y).lt.1.0
c       3. power series method: abs(x).lt.4.0 and abs(y).lt.1.0.
c the routine is accurate to 1.e-07.  a new version of the code
c accurate to 1.e-11 is being developed and is available upon request.
c     the routine also computes the first derivative of the pdf after
c the pdf has been computed.  to obtain the first derivative a call
c must be made to the function dzeta which is an entry point in the
c subprogram zeta.
c
	COMPLEX DZETA,A1,A2,A3,B1,B2,B3,C1,C2,C3,D1,D2,D3,
	1 ARG,AUX0,AUX1,TERM,Z,ZZ
	complex a,b
	common /zetacmn/ imethod
c Force Re and Im parts into conseq mem locs to simulate STRUCTURE.
	common /zfoo/C3R,C3I,D1R,D1I,D2R,D2I,D3R,D3I,TERMR,TERMI,X,Y
	equivalence (C3,C3R),(D1,D1R),(D2,D2R),(D3,D3R),
	1   (TERM,TERMR),(Z,X)
c     STRUCTURE (C3,C3R/C3I),(D1,D1R/D1I),(D2,D2R/D2I),(D3,D3R/D3I),
c    1(TERM,TERMR/TERMI),(Z,X/Y)
	DATA D1R/0.0/, D1I/1.77245385090551/   !   i sqrt(pi)
	DATA D2R/0.0/, D2I/3.54490770181103/   ! 2 i sqrt(pi)
	DATA D3R/0.0/, D3I/7.08981540362206/   ! 4 i sqrt(pi)
	DATA D4/0.33333333333333/, EPS/1.0E-07/

c
c choose method of evaluating the z-function:
c imethod  =0 fast two pole approximation
c 	   =1 precise (1.0e-7 error) multi-region calculation 
	data imethod /1/

	if(imethod .ne. 0)goto 100
! two pole approximation valid in upper half plane, extend results to
! lower half plane using a symmetry ID:
	b=(.5,.80558)
	a=(.50556,-.81462)
	aux0=arg
	if(aimag(aux0) .lt. 0.0)aux0=conjg(aux0)
	zeta=b/(a-aux0)-conjg(b)/(conjg(a)+aux0)
	if(aimag(aux0) .lt. 0.0)then
	    zeta=conjg(zeta)+d2*cexp(-arg**2)
	endif

! this increased the loading by a factor of 13.  So it is very wrong to
! use different approximations for the real and imaginary parts.
! real part of two pole approximation, plus exact imaginary part:
!	zeta=zeta-aimag(zeta)
!	zeta=zeta+(0,1.77245)*exp(-arg**2)
	return

c     CODE ANALYSIS
c     OPTIMIZE
C
100	I=0
1	Z=ARG  
	ZZ=Z*Z
	YMAG=ABS(Y) 
	IF(YMAG-1.0)10,2,2
C
C     CONTINUED FRACTION METHOD: ABS(Y).GE.1.0
C
! GWH:  I don't understand the continued fraction method, nor where these
! formulas came from.  Fried-Conte, Barberio-Coresetti, and Abramowitz and
! Stegun all have what appear to be different formulas.  But I have
! checked the results against the tables in Fried-Conte, and against the
! other two methods along the borders. 
 
2	Y0=Y 
	Y=YMAG
	AUX1=1.5-Z*Z 
	AUX2=0.0 
	DEL=1.5
	A1=0.0 
	A2=-1.0 
	B1=1.0 
	B2=AUX1 
	C1=A2/B2
C
3	AUX1=AUX1+2.0 
	AUX2=AUX2-DEL 
	DEL=DEL+2.0
	A3=AUX1*A2+AUX2*A1 
	B3=AUX1*B2+AUX2*B1
	C2=A3/B3 
	C3=C2-C1
	IF(ABS(C3R)+ABS(C3I).LT.EPS)GO TO 5
	A1=A2 
	A2=A3 
	B1=B2 
	B2=B3 
	C1=C2 
	GO TO 3
5	IF(Y0)6,7,7
6	Y=Y0 
	C2=CONJG(C2)-D3*Z*CEXP(-ZZ)
7	AUX0=-(0.5*C2+1.0)/Z 
	GO TO 30
C
C     ASYMPTOTIC SERIES METHOD: ABS(X).GE.4.0 AND ABS(Y).LT.1.0
C
10	XMAG=ABS(X) 
	IF(XMAG-4.0)20,12,12
12	TERM=1.0/Z 
	AUX0=-TERM 
	AUX1=0.5*TERM**2 
	P=1.0

! Use the proper Stokes lines, the Fried and Conte choices are wrong.
! A derivation of the proper choice of Stokes' lines for the asymptotic
! formulas can be found in Stix's Plasma Waves book, Miyamoto, or 
! Greg Hammett's notes. in practice this doesn't matter since
! cexp(-z**2) is so small: 
	IF(Y .lt. -xmag) AUX0=AUX0+D2*CEXP(-ZZ) 
	if(y .ge. -xmag .and. y .lt. xmag) AUX0=AUX0+D1*CEXP(-ZZ)

18	TERM=AUX1*TERM*P 
	AUX0=AUX0-TERM 
	P=P+2.0
	IF(ABS(TERMR)+ABS(TERMI).LT.EPS)goto 30
	goto 18
C
C     POWER SERIES METHOD: ABS(X).LT.4.0 AND ABS(Y).LT.1.0
C
! instead of the Fried and Conte power series, use the one suggested by
! Barberio-Corsetti in MATT-773 (1970) based on the other of the two 
! power series representations for the error function.  The Fried and
! Conte power series has numerical problems for x as large as 4.0. 
 
20	nterm=0
	aux0=1.0
	aux1=1.0

22	nterm=nterm+1
	aux1=aux1*zz/nterm
	term=aux1/(2*nterm+1)
	aux0=aux0+term

	IF(ABS(TERMR)+ABS(TERMI) .gt. eps)goto 22

26	AUX0=CEXP(-ZZ)*(d1-2.0*Z*AUX0)
30	ZETA=AUX0 
	IF(I)35,35,40
35	RETURN

! entry to determine d Zeta/ d Arg:
	ENTRY DZETA(ARG) 
	I=1 
	IF(CABS(ARG-Z))1,40,1
40	DZETA=-2.0*(1.0+ARG*AUX0) 
	RETURN 
	END
