***********************************************************************
*                                                                     *
*     UPDATEANDREDUCECOLUMN.f:                                        *
*     Auxillary routine in the package PDGGHRD.                       *
*                                                                     *
*     Contributors: Björn Adlerborn                                   *
*                   Lars Karlsson                                     *
*                                                                     *
*     Department of Computing Science and HPC2N, Umea University      *
*                                                                     *
*                                                                     * 
***********************************************************************
      SUBROUTINE UPDATEANDREDUCECOLUMN( A, DESCA, CS, IA0, JA, M
     $   , SEQNUM, INFO, VER )

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
*                 The number of elements touched by the reduction.
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
      
*
*     DECLARATIONS
*     ============
*

*
*     Constants.
*    
      INTEGER CSRC_, CTXT_, LLD_, MB_, M_, NB_, N_, RSRC_, UDEF
      DOUBLE PRECISION ZERO, ONE
      PARAMETER ( CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5 , NB_ = 6, RSRC_ =
     $   7, CSRC_ = 8, LLD_ = 9, ZERO = 0.0D+0, ONE = 1.0D+0, UDEF = -1)
      
*
*     Scalar arguments.
*     
      INTEGER IA0, JA, M, INFO, SEQNUM, VER
      
*
*     Array arguments.
*     
      INTEGER DESCA( * )      
      DOUBLE PRECISION A( * ), CS( * )

*
*     Local scalars.
*     
      INTEGER ICTXT, NPROW, NPCOL, MYROW, MYCOL, IAM, NPROCS, IERR,
     $   RINDX, MB, NB, LCOL, N,
     $   NUMROWS, IA
      LOGICAL DEBUG, DOBORDERUP, DOBORDERDW, DOLOCAL
      LOGICAL DOUPDATE
      INTEGER REDISTTONPROCS, REDISTBLKSZ, MYNUMROWS
      INTEGER SEQ, GROW, MYROW_NEW, MYCOL_NEW
      INTEGER NPCOL_NEW, NPROW_NEW, ICTXT_NEW, UP, DOWN, I, J, LROW
      INTEGER CURRCNT, LOCROW1, LOCROW2
      DOUBLE PRECISION C, S, TMP, TMPUP, TMPDW
      INTEGER MYFIRSTROW, MYLASTROW
      LOGICAL USEONLYONECOLUMN
      INTEGER PCOL
      
      
*
*     Local arrays.
*     
      DOUBLE PRECISION, ALLOCATABLE :: tX( : )
      INTEGER, ALLOCATABLE :: PMAP( : )
      INTEGER tDESCX( 9 )


*
*     Externals.
*     
      INTEGER INDXG2P, INDXG2L, NUMROC, ICEIL, BLACS_PNUM, INDXL2G
      EXTERNAL INDXG2P, INDXG2L,INDXL2G, NUMROC, ICEIL
      EXTERNAL KRNLUPDATEANDREDUCECOLUMN, BLACS_PNUM

*
*     EXECUTABLE STATEMENTS
*     =====================
*

      
      DEBUG = .false.
      IF ( VER .EQ. 0 ) THEN      
         USEONLYONECOLUMN = .TRUE.
      ELSE IF ( VER .EQ. 1 ) THEN
         USEONLYONECOLUMN = .FALSE.
      ELSE 
         CALL UPDATEANDREDUCECOLUMN_ROOT( A, DESCA, CS, IA0, JA, M
     $      ,SEQNUM, INFO )
         RETURN
      END IF
      
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
*     The global row index of first element touched by the reduction.
*     
      IA = IA0 + (NUMROWS - M)
      
*
*     Zero out CS.
*
      DO I = IA + 1, IA + M - 1
         CS( I + (2 * (SEQNUM - 1)) * N ) = 0.0D0
         CS( I + (2 * (SEQNUM - 1) + 1) * N ) = 0.0D0
      END DO
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

      PCOL = INDXG2P( JA, NB, UDEF, DESCA( CSRC_ ), NPCOL )

*
*     Set the number of processors particpating in the redistrbution of vector in A to  
*     update and annihilate
      IF (USEONLYONECOLUMN) THEN
         REDISTTONPROCS = NPROW
      ELSE
         REDISTTONPROCS = NPROW * NPCOL
      END IF
*
*     Allocate space for PMAP
*      
      ALLOCATE( PMAP( REDISTTONPROCS ) )

      CALL BLACS_GET(ICTXT, 10, ICTXT_NEW)
      CURRCNT = 0
      IF ( USEONLYONECOLUMN ) THEN
         DO I = 0, NPROW - 1
            CURRCNT = CURRCNT + 1
            PMAP( CURRCNT ) = BLACS_PNUM( ICTXT, I, PCOL )
            IF ( CURRCNT.EQ.REDISTTONPROCS) EXIT
         END DO
      ELSE
         DO I = 0, NPROW - 1
            DO J = 0, NPCOL - 1 
               CURRCNT = CURRCNT + 1
               PMAP( CURRCNT ) = BLACS_PNUM( ICTXT, I, J )
               IF ( CURRCNT.EQ.REDISTTONPROCS) EXIT
            END DO
            IF ( CURRCNT.EQ.REDISTTONPROCS) EXIT
         END DO
      END IF

      CALL BLACS_GRIDMAP( ICTXT_NEW, PMAP, REDISTTONPROCS, 
     $   REDISTTONPROCS, 1  )

      CALL BLACS_GRIDINFO( ICTXT_NEW, NPROW_NEW, NPCOL_NEW, MYROW_NEW,
     $   MYCOL_NEW )
          

*     Set the REDISTBLKSZ to use so that all participating processors 
*     will own at most 1 block of consequtive elements
      REDISTBLKSZ = ICEIL(NUMROWS, REDISTTONPROCS )

*     Calculate how many rows of the vector i will own
      IF ( MYROW_NEW .NE. -1 ) THEN
         MYNUMROWS = NUMROC( NUMROWS, REDISTBLKSZ, MYROW_NEW, 0,
     $      REDISTTONPROCS )
      ELSE
         MYNUMROWS = 0
      END IF
*
*     Set up a descriptor to gather/scatter the column to/from root.
*      
      if (debug) write(*,*)'% UPRECOL:  I have ', MYNUMROWS, 
     $   ' of X vec.', MYROW_NEW

      IF ( MYROW_NEW .NE. -1 ) THEN
         CALL DESCINIT( tDESCX, NUMROWS, 1, REDISTBLKSZ, 1, 0, 0, 
     $      ICTXT_NEW, MAX(MYNUMROWS, 1), IERR )
      ELSE
         tDESCX( CTXT_ ) = -1
      END IF
      
*
*     Allocate temporary workspace for the vector.
*      
      ALLOCATE ( tX( MAX( 1, MYNUMROWS ) ) )
      

*      
*     Gather the column onto the particpating processors. 
*
      CALL PDGEMR2D( NUMROWS, 1, A, IA0, JA, DESCA,
     $   tX, 1, 1, tDESCX, ICTXT )
      

*     Check if participating or not.
      IF ( MYROW_NEW .EQ. -1 ) THEN
         GOTO 999
      END IF

      
      UP = MODULO( MYROW_NEW - 1, REDISTTONPROCS )
      DOWN = MODULO( MYROW_NEW + 1, REDISTTONPROCS )

*     
*     Apply the previous rotations to the column and then reduce the column.
*

      IF ( DEBUG ) WRITE(*,*)'% UPRECOL : Looping sec=1..',SEQNUM
         
      DO SEQ = 1, SEQNUM 
*        Determine whether to do updates or annihilation         
         DOUPDATE = SEQ .NE. SEQNUM 
         
*        Localize the global range seq:numrows into LOCROW1:LOCROW2                  
         CALL GRN2LRN( SEQ, NUMROWS, REDISTBLKSZ, NPROW_NEW, MYROW_NEW,
     $      tDESCX( RSRC_ ), LOCROW1, LOCROW2 )
         
*        Initialize the global row pointer into CS          
         GROW = MIN( IA0 + NUMROWS - 1, IA0 + INDXL2G( LOCROW2,
     $      REDISTBLKSZ,MYROW_NEW,tDESCX( RSRC_ ), NPROW_NEW ) )
         
*        Find my global first row             
         MYFIRSTROW = IA0 + INDXL2G( LOCROW1, REDISTBLKSZ,MYROW_NEW
     $      ,tDESCX( RSRC_ ), NPROW_NEW ) - 1
*        Find my global last row          
         MYLASTROW = IA0 + INDXL2G( LOCROW2, REDISTBLKSZ,MYROW_NEW
     $      ,tDESCX( RSRC_ ), NPROW_NEW ) - 1
         
*        Decide whether to do crossborder with process below or not
         DOBORDERDW = LOCROW2 - LOCROW1 + 1 .GT. 0 .AND. MYLASTROW .LT.
     $      IA0 + NUMROWS - 1
*        Decide whether to do crossborder with process up or not          
         DOBORDERUP = LOCROW2 - LOCROW1 + 1 .GT. 0 .AND. MYFIRSTROW .GT.
     $      IA0 + SEQ - 1
*        Decide whether to do local updates/annihilation   
         DOLOCAL = MYNUMROWS .GT. 1
         
         if (debug) 
     $      write(*,*)'% UPRECOL :',
     $      SEQ, NUMROWS, GROW, LOCROW1, LOCROW2, MYROW_NEW
         
         IF ( DOBORDERDW ) THEN
            IF ( DEBUG ) WRITE(*,*)'% UPRECOL : Doing xborder down'
            LROW = LOCROW2
            IF ( DOUPDATE ) THEN
*              Send current element to below
               CALL DGESD2D( ICTXT_NEW, 1, 1, tX( LROW ), 1,
     $            DOWN, 0 )      
            END IF                      
*           Recv current element + 1 from below
            CALL DGERV2D( ICTXT_NEW, 1, 1, TMPDW, 1, DOWN, 0 )
*           Extract crossborder rotation
            IF ( DOUPDATE ) THEN 
*              Extract current rotation
               C = CS( GROW + ( SEQ - 1 ) * 2 * N )
               S = CS( GROW + ( SEQ - 1 ) * 2 * N + N )
*              apply the crossborder rotation 
               tX( LROW ) = C * tX( LROW ) + S * TMPDW 
            ELSE
*              Annihilate element row + 1 and update our row element
               TMP = tX( LROW )              
               CALL DLARTG( TMP, TMPDW, CS( GROW + 
     $            ( SEQ - 1 ) * 2 * N), CS( GROW + 
     $            ( SEQ - 1 ) * 2 * N + N ), tX( LROW ) )
            END IF
            GROW = GROW - 1
         END IF
         
*        Check if we should perform a local update
         IF ( DOLOCAL ) THEN
            IF ( DOUPDATE ) THEN
               DO LROW = LOCROW2, LOCROW1 + 1, -1
*                 Extract current rotation
                  C = CS( GROW + ( SEQ - 1 ) * 2 * N )
                  S = CS( GROW + ( SEQ - 1 ) * 2 * N + N )
*                 Apply the rotation 
                  TMP = tX ( LROW - 1 )
                  tX( LROW - 1 ) = C * TMP + S * tX ( LROW ) 
                  tX( LROW ) = C * tX ( LROW ) - S * TMP
                  GROW = GROW - 1
               END DO
            ELSE
               DO LROW = LOCROW2, LOCROW1 + 1, -1
*                 Annihilate LROW element row and update LROW- 1 element
                  TMP = tX( LROW - 1 )                
                  CALL DLARTG( TMP, tX( LROW ), 
     $               CS( GROW + ( SEQ - 1 ) * 2 * N), 
     $               CS( GROW + ( SEQ - 1 ) * 2 * N + N ), 
     $               tX( LROW - 1 ) )
                  GROW = GROW - 1
               END DO
            END IF
         END IF
         
*        Check if we should perform a crossborder with process above                  
         IF ( DOBORDERUP ) THEN
            LROW = LOCROW1
*           Send current element up
            CALL DGESD2D(ICTXT_NEW, 1, 1, tX( LROW ), 1, UP, 0 )
            IF ( DOUPDATE) THEN
*              Recv LROW-1 element from up
               CALL DGERV2D( ICTXT_NEW, 1, 1, TMPUP, 1, UP, 0)
*              Extract crossborder rotation
               C = CS( GROW + ( SEQ - 1 ) * 2 * N )
               S = CS( GROW + ( SEQ - 1 ) * 2 * N + N)
*              Apply the crossborder rotation 
               tX( LROW ) = C * tX ( LROW ) - S * TMPUP
            END IF 
         END IF
      END DO

*     
*     Broadcast the new rotations to all.
*
      CALL DGSUM2D(ICTXT_NEW, 'A', ' ',  M - 1, 2, CS( IA + 1 + ( SEQNUM
     $   - 1 ) * 2 * N), N, -1, -1)

 999  CONTINUE
      
*     
*     Scatter the updated column (except the zero part).
*     
      CALL PDGEMR2D( NUMROWS - ( M - 1 ), 1, tX, 1, 1, tDESCX,
     $   A, IA0, JA, DESCA, ICTXT )

*     
*     Exit from the temporary grid
*
      IF( MYROW_NEW .GE. 0 ) CALL BLACS_GRIDEXIT( ICTXT_NEW )

*
*     Broadcast on mesh rows.
*     
      IF ( USEONLYONECOLUMN ) THEN
         IF ( MYCOL .EQ. PCOL ) THEN
            CALL DGEBS2D( ICTXT, 'R', ' ', M - 1, 2, CS( IA + 1 +
     $         (SEQNUM- 1 ) * 2 * N), N )
         ELSE
            CALL DGEBR2D( ICTXT, 'R', ' ', M - 1, 2, CS( IA + 1 +
     $         (SEQNUM- 1 ) * 2 * N), N, MYROW, PCOL )
         END IF
      END IF
      
*     
*     Localize the row range IA + 1 : IA + M - 1 to LOCROW1 : LOCROW2.
*     
      CALL GRN2LRN( IA + 1, IA + M - 1, MB, NPROW, MYROW, DESCA( RSRC_ )
     $   , LOCROW1, LOCROW2 )

*     
*     If I own part of the column, then ...
*     
      IF ( MYCOL .EQ. PCOL )
     $   THEN
*
*        Zero out my part of the column.
*        
         LCOL = INDXG2L( JA, NB, 0, DESCA( CSRC_ ), NPCOL )
         DO RINDX = LOCROW1, LOCROW2
            A( RINDX + ( LCOL - 1) * DESCA( LLD_ ) ) = ZERO
         END DO
      END IF
      
      
*     
*     Deallocate temporary workspace.
*     
      DEALLOCATE ( tX, PMAP )

      
      RETURN
      
      END 


