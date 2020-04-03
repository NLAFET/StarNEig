      SUBROUTINE GRN2LRN( GINDX1, GINDX2, NB, NPROCS, MYROC, ISRCPROC, 
     $                    LINDX1, LINDX2 )
          IMPLICIT NONE
*     .. Scalar Arguments ..
          INTEGER         GINDX1, GINDX2, NB, NPROCS, MYROC, ISRCPROC, 
     $                    LINDX1, LINDX2
      
          INTEGER         ROCSRC
          CALL INFOG1L( GINDX1, NB, NPROCS, MYROC, ISRCPROC, 
     $                    LINDX1, ROCSRC)
          
          CALL INFOG1L( GINDX2+1, NB, NPROCS, MYROC, ISRCPROC, 
     $                    LINDX2, ROCSRC)
          LINDX2 = LINDX2 - 1
          
      END