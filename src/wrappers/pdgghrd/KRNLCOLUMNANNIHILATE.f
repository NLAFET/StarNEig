      subroutine krnlcolumnannihilate( n, A, ldA, cvec, svec )

      implicit none

*
*     PURPOSE
*     =======
*
*     Reduces an upper Hessenberg n-by-n matrix A to upper triangular
*     form by constructing and applying a decreasing sequence of column
*     n - 1 column rotations. 
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
*     The transformation G(k) is chosen such that it annihilates the
*     subdiagonal element A(k,k-1) using the diagonal element A(k,k) as
*     a pivot.
*
*
*     ARGUMENTS
*     =========
*
*     n             (input) INTEGER
*
*                   The number of rows and columns in A.
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
*     cvec          (output) DOUBLE PRECISION array
*                   dimension n
*     
*                   cvec(k) is the c-parameter of G(k).
*
*     svec          (output) DOUBLE PRECISION array
*                   dimension m
*
*                   svec(k) is the s-parameter of G(k).
*     

*     
*     Scalar arguments.
*     
      integer n, ldA

*
*     Array arguments.
*     
      double precision A( ldA, * ), cvec( * ), svec( * )

*
*     Externals.
*
      external dlartg

*
*     Local scalars.
*     
      integer i, j
      double precision c, s, x, y, tmp

      do j = n, 2, -1
         tmp = A( j, j )
         call dlartg( tmp, A( j, j - 1 ), c, s, A( j, j ) )
         cvec( j ) = c
         svec( j ) = s
         A( j, j - 1 ) = 0.0D+0
         do i = 1, j - 1
            x = A( i, j - 1 )
            y = A( i, j     )
            A( i, j - 1 ) = c * x - s * y
            A( i, j     ) = s * x + c * y
         end do
      end do

      end subroutine

