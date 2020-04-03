      subroutine krnlrowupdate( m, n, A, ldA, cvec, svec )

      implicit none

*
*     PURPOSE
*     =======
*
*     Applies a decreasing sequence of m - 1 row rotations to a
*     rectangular m-by-n matrix A.
*
*     More specifically, the following update of A is performed:
*
*       A := G(2) * G(3) * ... * G(m) * A
*
*     where G(k) for k = 2, 3, ..., m affects only rows k - 1 and k in
*     the following way. Let x denote row k - 1 and y denote row
*     k. Furthermore, define c = cvec(k) and s = svec(k). Then the
*     effect of G(k) * A is equivalent to the update
*
*       [ x ] := [ c  s ] * [ x ]
*       [ y ]    [-s  c ]   [ y ].
*
*
*     ARGUMENTS
*     =========
*
*     m             (input) INTEGER
*
*                   The number of rows in A.
*
*     n             (input) INTEGER
*
*                   The number of columns in A.
*
*     A             (input/output) DOUBLE PRECISION array
*                   dimension ldA-by-n
*
*                   The matrix A.
*
*     ldA           (input) INTEGER
*
*                   The column stride of A.
*
*     cvec          (input) DOUBLE PRECISION array
*                   dimension m
*
*                   cvec(k) is the c-parameter of G(k).
*
*     svec          (input) DOUBLE PRECISION array
*                   dimension m
*
*                   svec(k) is the s-parameter of G(k).
*     

*
*     Constants.
*     
      integer slabsz
      parameter( slabsz = 16 )

*     
*     Scalar arguments.
*     
      integer m, n, ldA

*
*     Array arguments.
*     
      double precision A( ldA, * ), cvec( * ), svec( * )

*
*     Intrinsics.
*     
      intrinsic min

*
*     Local scalars.
*     
      integer i, j, jj, jend
      double precision c, s, x, y

      do jj = 1, n, slabsz
         jend = min( n, jj + slabsz - 1 )
         do i = m, 2, -1
            c = cvec( i )
            s = svec( i )
            do j = jj, jend
               x = A( i - 1, j )
               y = A( i    , j )
               A( i - 1, j ) =   c * x + s * y
               A( i    , j ) = - s * x + c * y
            end do
         end do
      end do

      end subroutine

