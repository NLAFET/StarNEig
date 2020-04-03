***********************************************************************
*                                                                     *
*     UPDATEANDREDUCECOLUMN_ROOT.f:                                   *
*     Auxillary routine in the package PDGGHRD.                       *
*                                                                     *
*     Contributors: Björn Adlerborn                                   *
*                   Lars Karlsson                                     *
*                                                                     *
*     Department of Computing Science and HPC2N, Umea University      *
*                                                                     *
*                                                                     * 
***********************************************************************
      SUBROUTINE UPDATEANDREDUCECOLUMN_ROOT( A, DESCA, CS, IA0, JA, M
     $   ,SEQNUM, INFO )

      IMPLICIT NONE

*
*     PURPOSE
*     =======
*     
*     Updates the last DESCA( N_ ) - IA0 + 1 entries of the distributed column 
*     A( IA0:DESCA(N_), JA ) with respect to previous rotation
*     sequences. Then the last M - 1 entries of the column are reduced and
*     the resulting rotations replicated across the mesh.
*
*
*     ARGUMENTS
*     =========
*
*     A           (local input/output) DOUBLE PRECISION array
*                 dimension LLD_A x LOCc(M_A)
*
*                 The distributed matrix.
*
*     DESCA       (global and local input) INTEGER array 
*                 dimension 9
*     
*                 The array descriptor for A.
*     
*     CS          global and locak input ) INTGER array dimension
*                 DESCA(M_A) * 2 * SEQNUM
*
*                 ADD description
*
*     IA0         (global input) INTEGER
*     
*                 The row index of the top of the updated part of the column.
*             
*     JA          (global input) INTEGER
*     
*                 The column index of the column to update and reduce.
*     
*     M           (global input) INTEGER
*     
*                 The number of elements to update.
*
*     SEQNUM      (global input) 
*
*                 The rotation sequence number (starting with 1) of the
*                 sequence created by this routine. SEQNUM - 1 previous
*                 rotation sequences are applied in the update phase. 
*
*     INFO        (global output) INTEGER
*
*                 Status output.
*                 = 0:  successful exit.
*

      
*
*     DECLARATIONS
*     ============
*

*
*     Constants.
*    
      INTEGER CSRC_, CTXT_, LLD_, MB_, M_, NB_, N_, RSRC_
      DOUBLE PRECISION ZERO, ONE
      PARAMETER ( CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5 , NB_ = 6, RSRC_ =
     $   7, CSRC_ = 8, LLD_ = 9, ZERO = 0.0D+0, ONE = 1.0D+0 ) 
      
*
*     Scalar arguments.
*     
      INTEGER IA0, JA, M, INFO, SEQNUM
      
*
*     Array arguments.
*     
      INTEGER DESCA( * )      
      DOUBLE PRECISION A( * ), CS( * )

*
*     Local scalars.
*     
      INTEGER ICTXT, NPROW, NPCOL, MYROW, MYCOL, IAM, NPROCS, IERR,
     $   RINDX, PROW, PCOL, MB, NB, LSROW, LEROW, LCOL, N,
     $   NUMROWS, IA
      LOGICAL IAMROOT

*
*     Local arrays.
*     
      DOUBLE PRECISION, ALLOCATABLE :: tX( : )
      INTEGER tDESCX( 9 )

*
*     Externals.
*     
      INTEGER INDXG2P, INDXG2L, NUMROC
      DOUBLE PRECISION GETTOD
      EXTERNAL INDXG2P, INDXG2L, NUMROC, GETTOD
      EXTERNAL KRNLUPDATEANDREDUCECOLUMN

*
*     EXECUTABLE STATEMENTS
*     =====================
*



*
*     Reset INFO.
*     
      INFO = 0

*     
*     Extract the number of rows in A.
*     
      N = DESCA( M_ )

*     
*     The number of rows to update before reducing.
*     
      NUMROWS = N - IA0 + 1

*
*     The global row index of last element to perform a reducing
*     operation on (IA + 1 will be last element to be reduced, while IA
*     is updated).
*     
      IA = IA0 + (NUMROWS - M)
      
*     
*     Extract the distribution block size.
*     
      MB = DESCA( MB_ )
      NB = DESCA( NB_ )

*     
*     Get process mesh information.
*     
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      CALL BLACS_PINFO( IAM, NPROCS )     
      PROW = INDXG2P( IA, MB, 0, DESCA( RSRC_ ), NPROW )
      PCOL = INDXG2P( JA, NB, 0, DESCA( CSRC_ ), NPCOL )
      IAMROOT = ( PROW .EQ. MYROW .AND. PCOL .EQ. MYCOL )

*
*     Allocate temporary workspace on root.
*      
      IF ( IAMROOT ) THEN
         ALLOCATE ( tX( NUMROWS ) )
      ELSE
         ALLOCATE ( tX( 1 ) )
      END IF
      
*
*     Set up a descriptor to gather/scatter the column to/from root.
*      
      CALL DESCINIT( tDESCX, NUMROWS, 1, NUMROWS, 1, PROW, PCOL, ICTXT,
     $   NUMROWS, IERR ) 
      IF ( IERR .NE. 0 ) RETURN

*      
*     Gather the column on root. 
*      
      CALL PDCOPY( NUMROWS, A, IA0, JA, DESCA, 1, tX, 1, 1, tDESCX, 1 )

*     
*     Skip ahead unless root.
*
      IF ( .NOT. IAMROOT ) GOTO 100

*
*     Apply the previous rotations to the column and then reduce the column.
*
      CALL KRNLUPDATEANDREDUCECOLUMN( NUMROWS, SEQNUM, TX, CS( IA0 ), N 
     $   )
      
 100  CONTINUE

*
*     Scatter the updated column (except the zero part).
*     
      CALL PDCOPY( NUMROWS - (M - 1), tX, 1, 1, tDESCX, 1, A, IA0, JA,
     $   DESCA, 1 )

*     
*     Localize the row range IA + 1 : IA + M - 1 to LSROW : LEROW.
*     
      CALL GRN2LRN( IA + 1, IA + M - 1, MB, NPROW, MYROW, DESCA( RSRC_ )
     $   , LSROW, LEROW )

*
*     If I own part of the column, then ...
*     
      IF ( MYCOL .EQ. PCOL ) THEN
*
*        Zero out my part of the column.
*        
         LCOL = INDXG2L( JA, NB, 0, DESCA( CSRC_ ), NPCOL )
         DO RINDX = LSROW, LEROW
            A( RINDX + ( LCOL - 1) * DESCA( LLD_ ) ) = ZERO
         END DO
      END IF

*
*     Broadcast the new rotations to all.
*      
      IF ( IAMROOT .AND. NPROCS .GT. 1 ) THEN
         CALL DGEBS2D( ICTXT, 'A', ' ', M - 1, 2, CS( IA + 1 + ( SEQNUM
     $      - 1 ) * 2 * N ), N )
      ELSE IF ( NPROCS .GT. 1 ) THEN
         CALL DGEBR2D( ICTXT, 'A', ' ', M - 1, 2, CS( IA + 1 + ( SEQNUM
     $      - 1 ) * 2 * N ), N, PROW, PCOL )
      END IF

*
*     Deallocate temporary workspace.
*
      DEALLOCATE ( tX )

      
      RETURN
      
      END 

