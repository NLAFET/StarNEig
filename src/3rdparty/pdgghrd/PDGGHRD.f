***********************************************************************
*                                                                     *
*     PDGGHRD.f:                                                      *
*     Driver routine in the package PDGGHRD.                          *
*                                                                     *
*     Contributors: Björn Adlerborn                                   *
*                   Lars Karlsson                                     *
*                                                                     *
*     Department of Computing Science and HPC2N, Umea University      *
*                                                                     *
*                                                                     *
***********************************************************************
      SUBROUTINE PDGGHRD( COMPQ, COMPZ, N, ILO, IHI, A, DESCA, B,
     $   DESCB, Q, DESCQ, Z, DESCZ, WORK, LWORK, INFO )

      IMPLICIT NONE

*
*     PURPOSE
*     =======
*
*     Parallel, real, double precision, blocked routine that reduces a
*     pair of real matrices (A, B) to real upper Hessenberg-Triangular
*     form using orthogonal transformations, where A is a general matrix
*     and B is upper triangular: Q'' * A * Z = H and Q'' * B * Z = T,
*     where H is upper Hessenberg, T is upper triangular, and Q and Z
*     are orthogonal, and '' means transpose.
*
*     The orthogonal matrices Q and Z are determined as products of
*     Givens rotations.  They may either be formed explicitly, or they
*     may be postmultiplied into input matrices Q1 and Z1, so that
*
*     Q1 * A * Z1'' = (Q1*Q) * H * (Z1*Z)''
*     Q1 * B * Z1'' = (Q1*Q) * T * (Z1*Z)''
*
*
*     ARGUMENTS
*     =========
*
*     COMPQ   (global input) CHARACTER*1
*              = 'N': do not compute Q;
*              = 'I': Q is initialized to the unit matrix, and the
*                     orthogonal matrix Q is returned;
*              = 'V': Q must contain an orthogonal matrix Q1 on entry,
*                     and the product Q1*Q is returned.
*
*     COMPZ   (global input) CHARACTER*1
*             = 'N': do not compute Z;
*             = 'I': Z is initialized to the unit matrix, and the
*                    orthogonal matrix Z is returned;
*             = 'V': Z must contain an orthogonal matrix Z1 on entry,
*                    and the product Z1*Z is returned.
*
*     N       (global input) INTEGER
*             The order of the matrices A, B, Q and Z.  N >= 0.
*
*     ILO     (global input) INTEGER
*     IHI     (global input) INTEGER
*             It is assumed that A is already upper triangular in rows 
*             1:ILO-1 and IHI+1:N.

*             1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
*
*     A       (local input/output) DOUBLE PRECISION array,
*             dimension (LLD_A, LOCc(N)).
*             On entry, the N-by-N general matrix to be reduced.
*             On exit, the upper triangle and the first subdiagonal of A
*             are overwritten with the upper Hessenberg matrix H, and the
*             rest is set to zero.
*
*     DESCA   (global and local input) INTEGER array of dimension DLEN_.
*             The array descriptor for the distributed matrix A.
*
*     B       (local input/output) DOUBLE PRECISION array,
*             dimension (LLD_B, LOCc(N)).
*             On entry, the N-by-N upper triangular matrix B.
*             On exit, the upper triangular matrix T = Q'' B Z.  The
*             elements below the diagonal are set to zero.
*
*     DESCB   (global and local input) INTEGER array of dimension DLEN_.
*             The array descriptor for the distributed matrix B.
*
*     Q       (local input/output) DOUBLE PRECISION array,
*             dimension (LLD_Q, LOCc(N)).
*             If COMPQ='N': Q is not referenced.
*             If COMPQ='I': on entry, Q need not be set, and on exit it
*             contains the orthogonal matrix Q, where Q''
*             is the product of the Givens transformations
*             which are applied to A and B on the left.
*             If COMPQ='V':  on entry, Q must contain an orthogonal matrix
*             Q1, and on exit this is overwritten by Q1*Q.
*
*     DESCQ   (global and local input) INTEGER array of dimension DLEN_.
*             The array descriptor for the distributed matrix Q.
*
*     Z       (local input/output) DOUBLE PRECISION array,
*             dimension (LLD_Z, LOCc(N)).
*             If COMPZ='N': Z is not referenced.
*             If COMPZ='I': on entry, Z need not be set, and on exit it
*             contains the orthogonal matrix Z, which is
*             the product of the Givens transformations
*             which are applied to A and B on the right.
*             If COMPZ='V':  on entry, Z must contain an orthogonal matrix
*             Z1, and on exit this is overwritten by Z1*Z.
*
*     DESCZ   (global and local input) INTEGER array of dimension DLEN_.
*             The array descriptor for the distributed matrix Z.
*
*     WORK    (local workspace/local output) DOUBLE PRECISION array,
*             dimension (LWORK)
*             On exit, WORK(1) returns the minimal and optimal LWORK.
*
*     LWORK   (local or global input) INTEGER
*             The dimension of the array WORK, minium size MB_
*
*             If LWORK = -1, then LWORK is global input and a workspace
*             query is assumed; the routine only calculates the minimum
*             and optimal size for all work arrays. Each of these
*             values is returned in the first entry of the corresponding
*             work array, and no error message is issued by PXERBLA.
*
*
*     INFO    (global output) INTEGER
*             = 0:  successful exit.
*             < 0:  if INFO = -i, the i-th argument had an illegal value.
*
*

*
*     DECLARATIONS
*     ============
*

*
*     Constants.
*
      INTEGER BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DT_, LLD_, MB_, M_,
     $   NB_, N_, RSRC_
      DOUBLE PRECISION ZERO, ONE, UDEF
      PARAMETER ( ZERO = 0.0D+0, ONE = 1.0D+0, BLOCK_CYCLIC_2D = 1,
     $   DLEN_ = 9, DT_ = 1,CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5, NB_ = 6
     $   , RSRC_ = 7, CSRC_ = 8, LLD_ = 9, UDEF = -1 )

*
*     Scalar arguments.
*
      INTEGER N, INFO, ILO, IHI, LWORK
      CHARACTER COMPQ, COMPZ

*
*     Array arguments.
*
      DOUBLE PRECISION A( * ), B( * ), WORK( * ), Q( * ), Z( * )
      INTEGER DESCA( * ), DESCB( * ), DESCQ( * ), DESCZ( * )

*
*     Local scalars.
*
      INTEGER IAM, SYSPROCS, MYROW, MYCOL, NPROW, NPCOL, ICTXT, NB, MB,
     $   ICOMPQ, ICOMPZ, IERR, J, PANELCOLUMN, MAX_GROUPINFO_CNT,
     $   NUMGROUPS_ROW, NUMGROUPS_COL, NUMINNERITER, INNERITER, ROWBLKSZ
     $   , COLBLKSZ, TRAFPTR, NUM_VIRTUAL_ROWBLKS, NUM_VIRTUAL_COLBLKS,
     $   NUMITER, I, OUTERITER, COLREDUCER_VER, BLKSZ, PROC, 
     $   TNUMINNERITER
      LOGICAL DEBUG, LQUERY

*
*     Local arrays.
*
      INTEGER, ALLOCATABLE :: GROUPINFO_ROW( : , : ), GROUPINFO_COL(:
     $   , : )
      INTEGER PARAMS( 10 )
      DOUBLE PRECISION, ALLOCATABLE :: CSROW(:, :), CSCOL(:, :),
     $   TRAFS_ROW( : ), TRAFS_COL( : )

*
*     Externals.
*
      LOGICAL LSAME
      INTEGER NUMROC, ICEIL, INDXG2P
      EXTERNAL NUMROC, ICEIL, LSAME, INDXG2P

*
*     EXECUTABLE STATEMENTS
*     =====================
*

*     Quick return if possible.
*
      IF ( N.LE.1 ) RETURN

*
*     Get process mesh information.
*
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_PINFO( IAM, SYSPROCS )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )

*
*     Note: MB = NB is assumed (square blocks).
*
      NB = DESCA( NB_ )
      MB = NB


*     Choose the desired number of virtual column/row blocks.
      PARAMS( 1 ) = NPCOL * 1
      PARAMS( 2 ) = NPROW * 1
*     Number of inner iterations
      PARAMS( 3 ) = NB
*     Perform a sync after each internal call, set to 1 if so. Not USED
      PARAMS( 4 ) = 0
*     Columnreducer version, 0, 1, or 2
      PARAMS( 5 ) = 0
*     Param 6 is not used
      PARAMS( 6 ) = 0


*
*     Enable/disable debugging.
*
      DEBUG = .FALSE.


*
*     Disable debugging if not process 0 (if initially enabled).
*
      DEBUG = ( IAM .EQ. 0 .AND. DEBUG )


*
*     Set the desired number of virtual blocks.
*
      NUM_VIRTUAL_ROWBLKS = PARAMS( 1 )
      NUM_VIRTUAL_COLBLKS = PARAMS( 2 )

*
*     Set the number of inner iterations.
*
      NUMINNERITER = PARAMS( 3 )

*
*     Set the number of inner iterations to use for first block
*     Depends upon value of ILO - make sure we do not work cross process borders
*     since the values of CSROW should initially be stored on a single process column
*
      TNUMINNERITER = MIN( NUMINNERITER, NB - MOD( ILO - 1, NB ) )

*
*     Set version of column reducer to use
*
      COLREDUCER_VER = PARAMS( 5 ) 
*
*     Check for workspace query.
*
      LQUERY = ( LWORK .EQ. -1 )
      IF ( LQUERY ) THEN
*
*        Perform workspace query and return.
*
*        Workspace is only needed for debugging purpose.
*        The routine PDLAPRNT requires at least MB_ as workspace.
*
         WORK( 1 ) =  DESCA( MB_ )
         RETURN
      END IF

*
*     Allocate storage for row and column rotations.
*
      ALLOCATE( CSCOL( N, 2 * NUMINNERITER ) )
      ALLOCATE( CSROW( N, 2 * NUMINNERITER ) )

*
*     Clear out row and column rotations.
*
      DO J = 1, NUMINNERITER
         DO I = 1, N
            CSROW( I, 2 * (J - 1) + 1 ) = 1.0D0
            CSCOL( I, 2 * (J - 1) + 1 ) = 1.0D0
            CSROW( I, 2 * (J - 1) + 2 ) = 0.0D0
            CSCOL( I, 2 * (J - 1) + 2 ) = 0.0D0
         END DO
      END DO

*
*     Allocate storage for row and column rotation group information.
*
      MAX_GROUPINFO_CNT = ICEIL( N, NB ) + 1
      ALLOCATE( GROUPINFO_ROW( 12, MAX_GROUPINFO_CNT ) )
      ALLOCATE( GROUPINFO_COL( 12, MAX_GROUPINFO_CNT ) )

*
*     Convert COMPQ and COMPZ from CHARACTER to INTEGER.
*
      IF( LSAME( COMPQ, 'N' ) ) THEN
         ICOMPQ = 1
      ELSE IF( LSAME( COMPQ, 'V' ) ) THEN
         ICOMPQ = 2
      ELSE IF( LSAME( COMPQ, 'I' ) ) THEN
         ICOMPQ = 3
      ELSE
         ICOMPQ = 0
      END IF

      IF( LSAME( COMPZ, 'N' ) ) THEN
         ICOMPZ = 1
      ELSE IF( LSAME( COMPZ, 'V' ) ) THEN
         ICOMPZ = 2
      ELSE IF( LSAME( COMPZ, 'I' ) ) THEN
         ICOMPZ = 3
      ELSE
         ICOMPZ = 0
      END IF

*
*     Check the input parameters.
*
      INFO = 0
      IF( ICOMPQ .LE. 0 ) THEN
         INFO = -1
      ELSE IF( ICOMPZ .LE. 0 ) THEN
         INFO = -2
      ELSE IF( N .LT. 0 ) THEN
         INFO = -3
      ELSE IF( ILO .LT. 1 ) THEN
         INFO = -4
      ELSE IF( IHI .GT. N .OR. IHI .LT. ILO - 1 ) THEN
         INFO = -5
      ELSE IF ( DESCA( NB_ ) .NE. DESCA( MB_ ) ) THEN
         INFO = -6
      ELSE IF ( LWORK .LT. DESCA( MB_ ) ) THEN
         INFO = -7
      END IF
      IF( INFO .NE. 0 ) THEN
         CALL PXERBLA( ICTXT, 'PDGGHRD', -INFO )
         RETURN
      END IF

*
*     Initialize Q and Z if desired.
*
      IF( ICOMPQ.EQ.3 ) THEN
         CALL PDLASET( 'Full', N, N, ZERO, ONE, Q, 1, 1, DESCQ )
      END IF

      IF( ICOMPZ.EQ.3 ) THEN
         CALL PDLASET( 'Full', N, N, ZERO, ONE, Z, 1, 1, DESCZ )
      END IF

      NUMITER = (N + NUMINNERITER - 3) / NUMINNERITER

*
*     Set the number of inner iterations to use for first block
*     Depends upon value of ILO - make sure we do not work cross process borders
*     since the values of CSROW should initially be stored on a single process column
*        
      TNUMINNERITER = MIN( NUMINNERITER, NB - MOD( ILO - 1, NB ) )

*
*     Outer loop over column panels.
*
      PANELCOLUMN = ILO
      OUTERITER = 0
      DO WHILE ( PANELCOLUMN .LE. IHI - 2 )
         OUTERITER = OUTERITER + 1

         IF ( DEBUG ) WRITE(*,*)'% BEGIN OUTERITER', OUTERITER, ' of ', 
     $      NUMITER

*
*        Inner loop over columns within the panel.
*
         INNERITER = 0

         DO J = PANELCOLUMN, MIN( PANELCOLUMN + TNUMINNERITER - 1, 
     $      IHI - 2)
            INNERITER = INNERITER + 1

            IF ( DEBUG ) WRITE(*,*)'% BEGIN INNERITER', INNERITER

*
*           Apply INNERITER - 1 previous rotation sequences and reduce A( :, J ).
*
            IF ( DEBUG ) WRITE(*,*)'% ENTER UPDATEANDREDUCECOLUMN', 
     $         PANELCOLUMN + 1, J, IHI - J, INNERITER, ILO, IHI, N
*           Use parallel column reducer _only_ when enough parallelism
*           exists to use all processes. When the chosen reducer is the
*           full mesh reducer, then use the single column reducer for
*           intermediate cases.
            COLREDUCER_VER = 2
            IF ( PARAMS( 5 ) .EQ. 0 .AND. INNERITER .GE. NPROW ) THEN
               COLREDUCER_VER = 0
            END IF
            IF ( PARAMS( 5 ) .EQ. 1 .AND. INNERITER .GE. NPROW * NPCOL )
     $         THEN
               COLREDUCER_VER = 1
            ELSE IF ( PARAMS( 5 ) .EQ. 1 .AND. INNERITER .GE. NPROW)
     $            THEN
               COLREDUCER_VER = 0
            END IF


            CALL UPDATEANDREDUCECOLUMN( A, DESCA, CSROW, PANELCOLUMN + 1
     $         , J, N - J, INNERITER, IERR, COLREDUCER_VER )

            IF ( DEBUG ) WRITE(*,*)'% LEAVE UPDATEANDREDUCECOLUMN'

*
*           Adapt the virtual column block size.
*
            COLBLKSZ = MAX( 8, ICEIL(
     $         NUMROC( N, NB, MYCOL, DESCA( CSRC_ ), NPCOL ) -
     $         NUMROC( J, NB, MYCOL, DESCA( CSRC_ ), NPCOL ),
     $         2 * NUM_VIRTUAL_COLBLKS ) )
            
*           
*           Apply row rotations to B.
*
            IF ( DEBUG ) WRITE(*,*)'% ENTER ROWUPDATE(B)'
            CALL SLIVERROWUPDATE( IHI - J, N - J, B, J + 1, J + 1, 
     $         DESCB, COLBLKSZ, 
     $         CSROW( 1, 1 + 2 * ( INNERITER - 1 ) ), CSROW( 1, 2 + 2*
     $         (INNERITER - 1 ) ) )

            IF ( DEBUG ) WRITE(*,*)'% LEAVE ROWUPDATE(B)'

*
*           Adapt the virtual row block size.
*
            ROWBLKSZ = MAX( 8, ICEIL( N - PANELCOLUMN, 2 *
     $         NUM_VIRTUAL_ROWBLKS * NPROW ) )


*
*           Annihilate fill in B using column rotations.
*
            IF ( DEBUG ) WRITE(*,*)'% ENTER COLUMNUPDATE(B)'
            CALL SLIVERHESSCOLUMNUPDATE( IHI - PANELCOLUMN, 
     $         IHI - J,
     $         B, PANELCOLUMN + 1, J + 1, DESCB, CSCOL(1, 1 + 2
     $         *(INNERITER - 1)), CSCOL(1, 2 + 2 * (INNERITER - 1))
     $         )

            IF ( DEBUG ) WRITE(*,*)'% LEAVE COLUMNUPDATE(B)'

*
*           Adapt the row block sizes.
*
            ROWBLKSZ = MAX( 8, ICEIL(
     $         NUMROC( N, NB, MYROW, DESCA( RSRC_ ), NPROW ) -
     $         NUMROC( PANELCOLUMN, NB, MYROW, DESCA( RSRC_ ), NPROW ),
     $         NUM_VIRTUAL_ROWBLKS ) )

*
*           Apply column rotations to A.
*
            IF ( DEBUG ) WRITE(*,*)'% ENTER COLUMNUPDATE(A)'
            CALL SLIVERCOLUMNUPDATE( IHI - PANELCOLUMN, IHI - J, A,
     $         PANELCOLUMN + 1, J + 1, DESCA, CSCOL(1, 1 + 2 *
     $         (INNERITER - 1)), CSCOL(1, 2 + 2 * (INNERITER - 1)),
     $         ROWBLKSZ )
            IF ( DEBUG ) WRITE(*,*)'% LEAVE COLUMNUPDATE(A)'
            

         END DO
*
*        After each completed inner loop, do a global row gather of CSCOL.
*        We "do not" know who will use them, so all must have them.
*        Two Steps

*     
*        Step 1. Loop over all CSCOL values to set to zero if we dont "own" them
*         

         I = PANELCOLUMN + 1
         DO WHILE ( I .LE. IHI )
            BLKSZ = MIN( IHI - I + 1, NB - MOD( I - 1 , NB ) )
            PROC = INDXG2P( I , NB, UDEF, DESCA( CSRC_ ), NPCOL )
            IF ( MYCOL .NE. PROC ) then
               CSCOL( I : I + BLKSZ - 1, 1 : 2 * INNERITER ) = 0.0D+0
            END IF
            I = I + BLKSZ
         END DO
     
*    
*        Step2. Perform a global row sum of CSCOL so all will receive all elements
*     
         CALL DGSUM2D(ICTXT, 'R', ' ',  IHI - PANELCOLUMN, 
     $      2 * INNERITER, CSCOL( 1 + PANELCOLUMN, 1 ), N, 
     $      -1, -1 )

*
*        Query workspace for row rotation accumulation.
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER ACCUMULATEROWROTATIONS QRY'
         ALLOCATE( TRAFS_ROW( 1 ) )
         CALL ACCUMULATEROWROTATIONS( .TRUE., PANELCOLUMN + 1, IHI -
     $      PANELCOLUMN, INNERITER, DESCA, CSROW, N, NUMGROUPS_ROW,
     $      GROUPINFO_ROW, TRAFS_ROW, TRAFPTR, IERR )
         DEALLOCATE( TRAFS_ROW )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE ACCUMULATEROWROTATIONS QRY'

*
*        Allocate workspace for row rotation accumulation.
*
         IF ( TRAFPTR .GT. 1 ) THEN
*           Normal case.
            ALLOCATE( TRAFS_ROW ( TRAFPTR - 1 ) )
         ELSE
*           Corner case.
            ALLOCATE( TRAFS_ROW ( 1 ) )
         END IF
          
*
*        Accumulate row rotations.
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER ACCUMULATEROWROTATIONS'
         CALL ACCUMULATEROWROTATIONS( .FALSE., PANELCOLUMN + 1, IHI -
     $      PANELCOLUMN, INNERITER, DESCA, CSROW, N, NUMGROUPS_ROW,
     $      GROUPINFO_ROW, TRAFS_ROW, TRAFPTR, IERR )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE ACCUMULATEROWROTATIONS'
*
*        Adapt the virtual column block size.
*
         COLBLKSZ = MAX( 8, ICEIL(
     $      NUMROC( N, NB, MYCOL, DESCA( CSRC_ ), NPCOL ) -
     $      NUMROC( J - 1, NB, MYCOL, DESCA( CSRC_ ), NPCOL ),
     $      NUM_VIRTUAL_COLBLKS ) )

*
*        Apply row transformations to A.
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER BLOCKROWUPDATE(A)'
         IERR = 0
         CALL BLOCKSLIVERROWUPDATE( A, DESCA, J, N - J + 1, COLBLKSZ,
     $      NUMGROUPS_ROW, GROUPINFO_ROW, TRAFS_ROW, IERR )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE BLOCKROWUPDATE(A)'

*
*        Adapt the virtual column block size.
*
         COLBLKSZ = MAX( 8, ICEIL(
     $      NUMROC( N, NB, MYCOL, DESCA( CSRC_ ), NPCOL ),
     $      NUM_VIRTUAL_COLBLKS ) )

*
*        Apply row transformations to Q.
*

         IF ( DEBUG ) WRITE(*,*)'% ENTER BLOCKROWUPDATE(Q)'
         IERR = 0
         CALL BLOCKSLIVERROWUPDATE( Q, DESCQ, 1, N, 
     $      COLBLKSZ,
     $      NUMGROUPS_ROW, GROUPINFO_ROW, TRAFS_ROW, IERR )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE BLOCKROWUPDATE(Q)'

*
*        Deallocate workspace row transformations.
*
         DEALLOCATE( TRAFS_ROW )

*
*        Query workspace for column rotation accumulation.
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER ACCUMULATECOLUMNROTATIONS QRY'
         ALLOCATE(TRAFS_COL(1))
         CALL ACCUMULATECOLUMNROTATIONS( .TRUE., PANELCOLUMN + 1, IHI -
     $      PANELCOLUMN, INNERITER, DESCA, CSCOL, N, NUMGROUPS_COL,
     $      GROUPINFO_COL, TRAFS_COL, TRAFPTR, IERR )
         DEALLOCATE(TRAFS_COL)
         IF ( DEBUG ) WRITE(*,*)'% LEAVE ACCUMULATECOLUMNROTATIONS QRY'

*
*        Allocate workspace for column rotation accumulation.
*
         IF ( TRAFPTR .GT. 1 ) THEN
*           Normal case.
            ALLOCATE( TRAFS_COL ( TRAFPTR - 1 ) )
         ELSE
*           Corner case.
            ALLOCATE( TRAFS_COL ( 1 ) )
         END IF

*
*        Accumulate column rotations.
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER ACCUMULATECOLUMNROTATIONS'
         CALL ACCUMULATECOLUMNROTATIONS( .FALSE., PANELCOLUMN + 1, 
     $      IHI -
     $      PANELCOLUMN, INNERITER, DESCA, CSCOL, N, NUMGROUPS_COL,
     $      GROUPINFO_COL, TRAFS_COL, TRAFPTR, IERR )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE ACCUMULATECOLUMNROTATIONS'

*
*        Adapt the virtual row block size.
*
         ROWBLKSZ = MAX( 8, ICEIL(
     $      NUMROC( N, NB, MYROW, DESCA( RSRC_ ), NPROW ),
     $      NUM_VIRTUAL_ROWBLKS ) )

*
*        Apply column transformations to Z.
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER BLOCKCOLUMNUPDATE(Z)'
         CALL BLOCKSLIVERCOLUMNUPDATE( Z, DESCZ, 
     $      1, N, ROWBLKSZ,
     $      NUMGROUPS_COL, GROUPINFO_COL, TRAFS_COL, IERR )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE BLOCKCOLUMNUPDATE(Z)'


*
*        Adapt the virtual row block size.
*
         ROWBLKSZ = MAX( 8, ICEIL(
     $      NUMROC( PANELCOLUMN, NB, MYROW, DESCA( RSRC_ ), NPROW ),
     $      NUM_VIRTUAL_ROWBLKS ) )

*
*        Apply column transformations to A and B
*
         IF ( DEBUG ) WRITE(*,*)'% ENTER DUOBLOCKCOLUMNUPDATE(A,B)'
         CALL DUOBLOCKSLIVERCOLUMNUPDATE( A, DESCA, B, DESCB, 
     $      1, PANELCOLUMN, ROWBLKSZ,
     $      NUMGROUPS_COL, GROUPINFO_COL, TRAFS_COL, IERR )
         IF ( DEBUG ) WRITE(*,*)'% LEAVE DUOBLOCKCOLUMNUPDATE(A,B)'
         
*
*        Deallocate workspace for column transformations.
*
         DEALLOCATE( TRAFS_COL )

*
*        Advance to the next column panel.
*

         PANELCOLUMN = PANELCOLUMN + TNUMINNERITER
*
*        Reset TNUMINNERITER to the passed parameter NUMINNERITER
         TNUMINNERITER = NUMINNERITER

      END DO

      IF ( DEBUG ) WRITE (*,*) '% Deallocating'

*
*     Deallocate group information.
*
      DEALLOCATE( GROUPINFO_ROW, GROUPINFO_COL )

*
*     Deallocate row and column rotations.
*
      DEALLOCATE( CSROW, CSCOL )

      RETURN

      END


   
