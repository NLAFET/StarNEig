      subroutine krnlcolumnupdate( m, n, A, ldA, cvec, svec )

      implicit none

*
*     PURPOSE
*     =======
*
*     Applies a decreasing sequence of n - 1 column rotations to a
*     rectangular m-by-n matrix A.
*
*     More specifically, the following update of A is performed:
*
*       A := A * G(k) * G(k-1) * ... * G(2)
*
*     where G(k) for k = 2, 3, ..., n affects only columns k - 1 and k
*     in the following way. Let x denote column k - 1 and y denote
*     column k. Furthermore, define c = cvec(k) and s = svec(k). Then
*     the effect of A * G(k) is equivalent to the update
*
*       [ x y ] := [ x y ] * [ c -s ]
*                            [ s  c ].
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
*                   dimension n
*     
*                   cvec(k) is the c-parameter of G(k).
*
*     svec          (input) DOUBLE PRECISION array
*                   dimension m
*
*                   svec(k) is the s-parameter of G(k).
*     

*     
*     Scalar arguments.
*     
      integer m, n, ldA

*
*     Array arguments.
*     
      double precision A( ldA, * ), cvec( * ), svec( * )

*
*     Local scalars.
*     
      integer i, j
      double precision c, s, x, y

      do j = n, 2, -1
         c = cvec( j )
         s = svec( j )
         do i = 1, m
            x = A( i, j - 1 )
            y = A( i, j     )
            A( i, j - 1 ) = c * x - s * y
            A( i, j     ) = s * x + c * y
         end do
      end do
      
      end subroutine

