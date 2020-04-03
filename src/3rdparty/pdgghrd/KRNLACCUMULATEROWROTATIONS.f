      subroutine krnlaccumulaterowrotations(n, numseq, size, u, ldu, cs,
     $     ldcs, islastgroup )

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
*     DECLARATIONS
*     ============
*     

*
*     Scalar arguments.
*          
      logical islastgroup
      integer n, numseq, size, ldu, ldcs

*
*     Array arguments.
*     
      double precision u(ldu, *), cs(ldcs, *)

*
*     Local scalars.
*     
      integer seq, i1, i2, i, j1, j2, j
      double precision c, s, x, y

*
*     Local arrays.
*     
      integer, allocatable :: left(:), right(:)

*
*     EXECUTABLE STATEMENTS
*     =====================
*     

*
*     Set U := I.
*     
      call dlaset ('A', n, n, 0.0d0, 1.0d0, u, ldu)

*
*     Initialize row skyline for U.
*     
      allocate (left (n), right (n))
      do i = 1, n
         left (i) = i
         right (i) = i
      end do

*
*     Loop over all sequences of rotations.
*     
      do seq = 1, numseq
*
*        The range of rotations to apply is i1 : i2.
*     
         i1 = seq + 1
         if (islastgroup) then
            i2 = n
         else
            i2 = i1 + size - 1
         end if

*
*        Loop over all rotations to apply. 
*     
         do i = i2, i1, -1
*
*           The column range to update is j1 : j2.
*           
            j1 = left  (i - 1)
            j2 = right (i    )

*
*           Update the skyline.
*           
            left  (i    ) = left  (i - 1)
            right (i - 1) = right (i    )

*
*           Extract the rotation parameters.
*           
            c = cs (i, 2 * (seq - 1) + 1)
            s = cs (i, 2 * (seq - 1) + 2)

*
*           Loop over all columns to update.
*           
            do j = j1, j2
*
*              Apply the rotation to U(i-1:i, j).
*              
               x = u (i - 1, j)
               y = u (i    , j)
               u (i - 1, j) =   c * x + s * y
               u (i    , j) = - s * x + c * y
            end do


         end do
      end do

*
*     Deallocate skyline.
*     
      deallocate (left, right)

      end
