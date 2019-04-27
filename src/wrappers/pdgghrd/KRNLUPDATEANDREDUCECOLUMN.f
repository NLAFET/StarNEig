      subroutine krnlupdateandreducecolumn( m, n, v, cs, ldcs )

      implicit none

*
*     PURPOSE
*     =======
*
*     TODO Describe the purpose.
*
*
*     ARGUMENTS
*     =========
*
*     TODO Describe the arguments.
*

*
*     Scalar arguments.
*     
      integer m, n, ldcs

*
*     Array arguments.
*     
      double precision v( * ), cs( ldcs, * )

*
*     Externals.
*
      external dlartg

*
*     Local scalars.
*
      integer i, j
      double precision c, s, x, y

*
*     EXECUTABLE STATEMENTS
*     =====================
*

*
*     Apply the n - 1 previous sequences.
*
      do j = 1, n - 1
         y = v( m )
         do i = m, j + 1, -1
            c = cs( i, 2 * j - 1 )
            s = cs( i, 2 * j     )
            x = v( i - 1 )
            v( i ) = - s * x + c * y
            y      =   c * x + s * y
         end do
         v( j ) = y
      end do

*
*     Generate the n-th sequence.
*
      do i = m, n + 1, -1
         x = v( i - 1 )
         call dlartg( x, v( i ),
     $      cs( i, 2 * n - 1 ), cs( i, 2 * n ),
     $      v( i - 1 ) )
      end do

      end subroutine
      
