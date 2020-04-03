      subroutine krnlaccumulatecolumnrotations(n, numseq, size, u, ldu,
     $   cs, ldcs, islastgroup )

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
      integer, allocatable :: top(:), bot(:)

*
*     EXECUTABLE STATEMENTS
*     =====================
*     

*
*     Set U := I.
*     
      call dlaset ('A', n, n, 0.0d0, 1.0d0, u, ldu)

*
*     Initialize column skyline for U.
*     
      allocate (top (n), bot (n))
      do i = 1, n
         top (i) = i
         bot (i) = i
      end do

*
*     Loop over all sequences of rotations.
*     
      do seq = 1, numseq
*
*        The range of rotations to apply is j1 : j2.
*     
         j1 = seq + 1
         if (islastgroup) then
            j2 = n
         else
            j2 = j1 + size - 1
         end if

*
*        Loop over all rotations to apply. 
*     
         do j = j2, j1, -1
*
*           The row range to update is j1 : j2.
*           
            i1 = top (j - 1)
            i2 = bot (j    )

*
*           Update the skyline.
*           
            top (j    ) = top (j - 1)
            bot (j - 1) = bot (j    )

*
*           Extract the rotation parameters.
*           
            c = cs (j, 2 * (seq - 1) + 1)
            s = cs (j, 2 * (seq - 1) + 2)

*
*           Loop over all rows to update.
*           
            do i = i1, i2
*
*              Apply the rotation to U(i, j-1:j)
*              
               x = u(i, j - 1)
               y = u(i, j    )
               u(i, j - 1) = c * x - s * y
               u(i, j    ) = s * x + c * y
            end do

         end do
      end do

*
*     Deallocate skyline.
*     
      deallocate (top, bot)

      end

