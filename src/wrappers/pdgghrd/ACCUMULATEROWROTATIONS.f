***********************************************************************
*                                                                     *
*     ACCUMULATEROWROTATIONS.f:                                       *
*     Auxillary routine in the package PDGGHRD.                       *
*                                                                     *
*     Contributors: Björn Adlerborn                                   *
*                   Lars Karlsson                                     *
*                                                                     *
*     Department of Computing Science and HPC2N, Umea University      *
*                                                                     *
*                                                                     * 
***********************************************************************
      SUBROUTINE ACCUMULATEROWROTATIONS( QUERY, I0, M, NUMSEQ, DESC, CS,
     $   LDCS, NUMGROUPS, GROUPINFO, TRAFS, GTRAFPTR, INFO )
      
      IMPLICIT NONE
      
*
*     PURPOSE
*     =======
*
*     Forms and accumulates groups of rotations into orthogonal
*     matrices. 
*     
*
*     ARGUMENTS
*     =========
*
*     QUERY       (global input) LOGICAL
*
*                 Group information is computed and workspace
*                 requirements is returned when QUERY is set.
*
*     I0          (global input) INTEGER
*
*                 The first row affected by the rotations.
*     
*     M           (global input) INTEGER
*
*                 The number of rows affected by the rotations.
*
*     NUMSEQ      (global input) INTEGER
*
*                 The number of rotation sequences.
*
*     DESC        (global/local input) INTEGER array
*                 dimension 9
*
*                 Descriptor for the associated matrix to be updated
*                 from the left by the accumulated rotations. 
*     
*     CS          (global input) DOUBLE PRECISION array
*                 dimension DESC( M_ ) x 2 * NUMSEQ
*
*                 The rotations themselves. The C parameters are stored
*                 in the odd-numbered columns and the S parameters in
*                 the even-numbered columns.
*
*     LDCS        (local input) INTEGER
*
*                 The column stride of CS.
*
*     NUMGROUPS   (global input/output) INTEGER
*
*                 The number of groups. Output when QUERY = .TRUE. and
*                 input otherwise.
*
*     GROUPINFO   (global input/output) INTEGER array
*                 dimension 12 x NUMGROUPS
*
*                 Table of information about each rotation group. Each
*                 column corresponds to one group. The first (top-most)
*                 group corresponds to the first column. The rows of
*                 GROUPINFO have the following interpretation:
*     
*                 1.  FIRSTROW - The first row affected by the group.
*                 2.  MIDDLEROW - The row below the distribution boundary.
*                 3.  LASTROW - The last row affected by the group.
*                 4.  NUMROWS - The number of affected rows.
*                 5.  ROWSABOVE - The number of affected rows above the 
*                     distribution boundary.
*                 6.  ROWSBELOW - The number of affected rows below the 
*                     distribution boundary.
*                 7.  SIZE - The size of the group. 
*                 8.  ISLASTGROUP - 1: The last group, 0 otherwise.
*                 9.  HOMEMESHROW - The mesh row of the accumulating process.
*                 10. HOMEMESHCOL - The mesh column of the accumulating process.
*                 11. TRAFPTR - The location of the transformation matrix in 
*                     memory. 
*                 12. LDTRAF - The column stride of the transformation matrix.
*
*                 Upper bound on the size of GROUPINFO should be
*                 ICEIL ( DESCA( M_ ), DESC( MB_ ) ) + 1
*
*     TRAFS       (local output) DOUBLE PRECISION array
*                 dimension MAX( GTRAFPTR - 1, 1 )
*
*                 The transformation matrices needed by this process.
*
*                 Referenced only when QUERY = .FALSE.
*
*     GTRAFPTR    (local input/output) INTEGER
*
*                 The size of the TRAFS array.
*
*                 Output when QUERY = .TRUE. and input otherwise.
*
*     INFO        (global output) INTEGER
*
*                 Status output.
*                 = 0: successful exit.
*
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
      INTEGER M, I0, NUMSEQ, LDCS, INFO, GTRAFPTR
      LOGICAL QUERY

*
*     Array arguments.
*     
      INTEGER DESC( 9 ), GROUPINFO( 12, * )
      DOUBLE PRECISION CS( LDCS, * ), TRAFS ( * )

*
*     Local scalars.
*     
      INTEGER UDEF, ICTXT, NPROW, NPCOL, MYROW, MYCOL, IAM, NPROCS, MB,
     $   NUMGROUPS, LDTRAF, GROUP, DOWN, UP, FIRSTROW,
     $   MIDDLEROW ,LASTROW , SIZE, ROWSABOVE, ROWSBELOW, NUMROWS,
     $   HOMEMESHROW, HOMEMESHCOL, TRAFPTR
      LOGICAL ISLASTGROUP
      
*
*     Externals.
*     
      INTEGER INDXG2P, INDXG2L
      EXTERNAL INDXG2P, INDXG2L

*
*     EXECUTABLE STATEMENTS
*     =====================
*


*
*     Reset INFO.
*     
      INFO = 0


*
*     TODO Description of the relationship between variables and groups.
*     
*     Example of a regular group:
*
*     rots   row
*             1 = FIRSTROW
*     x       2
*     x x     3
*     x x x   4
*     --------- = distribution boundary
*     c c c c 5 = MIDDLEROW
*       x x x 6
*         x x 7
*           x 8 = LASTROW
*
*     SIZE = 4
*     ISLASTGROUP = .FALSE.
*     ROWSABOVE = 4
*     ROWSBELOW = 4
*     NUMROWS = 8
*

*
*     Get process mesh information.
*     
      ICTXT = DESC(CTXT_)
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      CALL BLACS_PINFO( IAM, NPROCS )

*
*     Get the mesh rows of the up and down neighbours.
*
      UP = MODULO( MYROW - 1, NPROW )
      DOWN = MODULO( MYROW + 1, NPROW )

*
*     Extract the distribution row block size.
*     
      MB = DESC( MB_ )

*
*     Skip the query part unless requested.
*     
      IF ( .NOT. QUERY )  GOTO 100

*
*     Reset the group counter.
* 
      NUMGROUPS = 0

*
*     Set the offset in memory of the first group.
*     
      GTRAFPTR = 1
      
*
*     Find the first affected row of the first (top-most) group.
*
      FIRSTROW = I0
      
*
*     Always use mesh column 0 as start for home mesh column.
*
      HOMEMESHCOL = 0
      
*     
*     Loop until all rotations have been grouped.
*
      ISLASTGROUP = .FALSE.
      DO WHILE ( .NOT. ISLASTGROUP )
*
*        Count the new group.
*        
         NUMGROUPS = NUMGROUPS + 1
         
*        
*        Locate the first row below the next distribution block boundary.
*        
         MIDDLEROW = 1 + MB * ((FIRSTROW + MB - 1) / MB)
*        Crop to submatrix boundary if necessary.
         IF ( MIDDLEROW .GT. I0 + M - 1 ) THEN
            MIDDLEROW = I0 + M
         END IF

*        
*        Compute the size of the group.
*        
         SIZE = MIDDLEROW - FIRSTROW

*        
*        Locate the last row affected by the group.
*        
         LASTROW = MIDDLEROW + NUMSEQ - 1
*        Crop to submatrix boundary if necessary.
         IF ( LASTROW .GT. I0 + M - 1 ) THEN
            LASTROW = I0 + M - 1
         END IF

*        
*        Determine if this is the last group.
*        
         ISLASTGROUP = (LASTROW .GE. I0 + M - 1)
         
*        
*        Compute the number of rows affected above and below the
*        distribution boundary, respectively.
*        
         ROWSABOVE = MIDDLEROW - FIRSTROW
         ROWSBELOW = LASTROW - MIDDLEROW + 1

*        
*        Compute the total number of affected rows. 
*        
         NUMROWS = LASTROW - FIRSTROW + 1

*
*        Locate the home mesh row of this group (i.e., the mesh row
*        below the distribution block boundary).
*
         HOMEMESHROW = MOD( 1 + INDXG2P( FIRSTROW, MB, UDEF, DESC(RSRC_)
     $      , NPROW ), NPROW )

*
*        Populate a column of the GROUPINFO array.
*        
         GROUPINFO( 1, NUMGROUPS ) = FIRSTROW
         GROUPINFO( 2, NUMGROUPS ) = MIDDLEROW
         GROUPINFO( 3, NUMGROUPS ) = LASTROW
         GROUPINFO( 4, NUMGROUPS ) = NUMROWS
         GROUPINFO( 5, NUMGROUPS ) = ROWSABOVE
         GROUPINFO( 6, NUMGROUPS ) = ROWSBELOW
         GROUPINFO( 7, NUMGROUPS ) = SIZE
         IF ( ISLASTGROUP ) THEN
            GROUPINFO( 8, NUMGROUPS ) = 1
         ELSE
            GROUPINFO( 8, NUMGROUPS ) = 0
         END IF
         GROUPINFO( 9, NUMGROUPS ) = HOMEMESHROW
         GROUPINFO( 10, NUMGROUPS ) = MOD( HOMEMESHCOL + ( NUMGROUPS -
     $      1) / NPCOL, NPCOL )
         IF ( MYROW .EQ. HOMEMESHROW .OR. MYROW .EQ. MODULO( HOMEMESHROW
     $        - 1, NPROW ) ) THEN
*     
*           I need local storage for the transformation.
*
            LDTRAF = NUMROWS
            GROUPINFO( 11, NUMGROUPS ) = GTRAFPTR
            GROUPINFO( 12, NUMGROUPS ) = LDTRAF
            GTRAFPTR = GTRAFPTR + (NUMROWS * NUMROWS)
         ELSE
*
*           I will not use the transformation, so don't allocate storage for it.
*           
            GROUPINFO( 11, NUMGROUPS ) = 1
            GROUPINFO( 12, NUMGROUPS ) = 1
         END IF
         
*        
*        Locate the first affected row in the next group.
*        
         FIRSTROW = MIDDLEROW    

*
*        Locate next home mesh column.
*
         HOMEMESHCOL = MOD( HOMEMESHCOL + 1, NPCOL )
      END DO
      
*
*     Query done; exit.
*
      RETURN

 100  CONTINUE

*
*     Non-query mode begins here.
*     

*     
*     Loop over all groups.
*     
      DO GROUP = 1, NUMGROUPS
*
*        Extract cached group information.
*        
         FIRSTROW    = GROUPINFO( 1, GROUP )
         MIDDLEROW   = GROUPINFO( 2, GROUP )
         LASTROW     = GROUPINFO( 3, GROUP )
         NUMROWS     = GROUPINFO( 4, GROUP )
         ROWSABOVE   = GROUPINFO( 5, GROUP )
         ROWSBELOW   = GROUPINFO( 6, GROUP )
         SIZE        = GROUPINFO( 7, GROUP )
         ISLASTGROUP = GROUPINFO( 8, GROUP ) .EQ. 1
         HOMEMESHROW = GROUPINFO( 9, GROUP )
         HOMEMESHCOL = GROUPINFO( 10, GROUP )
         TRAFPTR     = GROUPINFO( 11, GROUP )
         LDTRAF      = GROUPINFO( 12, GROUP )

*        
*        If I am the home process, then ...
*
         IF ( MYROW .EQ. HOMEMESHROW .AND. MYCOL. EQ. HOMEMESHCOL ) THEN
*           
*           Accumulate the group of rotations.
*           
            CALL KRNLACCUMULATEROWROTATIONS( NUMROWS, NUMSEQ, SIZE,
     $           TRAFS( TRAFPTR ), LDTRAF, CS( FIRSTROW, 1 ), LDCS,
     $           ISLASTGROUP )
         END IF
      END DO

*
*     Loop over the groups. 
*
      DO GROUP = 1, NUMGROUPS
*
*        Extract cached group information.
*        
         NUMROWS     = GROUPINFO( 4, GROUP )
         HOMEMESHROW = GROUPINFO( 9, GROUP )
         HOMEMESHCOL = GROUPINFO( 10, GROUP )
         TRAFPTR     = GROUPINFO( 11, GROUP )
         LDTRAF      = GROUPINFO( 12, GROUP )
         
*
*        If I am the home process, then ...
*        
         IF ( MYROW .EQ. HOMEMESHROW .AND. MYCOL .EQ. HOMEMESHCOL ) THEN
            IF ( NPROW .GT. 1 ) THEN
*
*              Send the transformation matrix up. 
*              
               CALL DGESD2D( ICTXT, NUMROWS, NUMROWS, TRAFS( TRAFPTR ),
     $            LDTRAF, UP, MYCOL )
            END IF

            IF ( NPCOL .GT. 1 ) THEN
*
*              Broadcast the transformation matrix along the mesh row.
*              
               CALL DGEBS2D( ICTXT, 'R', ' ', NUMROWS, NUMROWS,
     $            TRAFS(TRAFPTR ), LDTRAF )
            END IF

*
*        else If I am above the home process, then ...
*        
         ELSE IF ( MYROW .EQ. MODULO( HOMEMESHROW - 1, NPROW ) .AND. 
     $           MYCOL .EQ. HOMEMESHCOL ) THEN
            IF ( NPROW .GT. 1 ) THEN
*
*              Receive the transformation matrix from below.
*
               CALL DGERV2D( ICTXT, NUMROWS, NUMROWS, TRAFS( TRAFPTR ),
     $            LDTRAF, DOWN, MYCOL )
            END IF
            
            IF ( NPCOL .GT. 1 ) THEN
*
*              Broadcast the transformation matrix along the mesh row.
*              
               CALL DGEBS2D( ICTXT, 'R', ' ', NUMROWS, NUMROWS,
     $            TRAFS(TRAFPTR ), LDTRAF )
            END IF
            
*
*        else If I will need this transformation later but am not on the home
*        mesh column, then ...
*
         ELSE IF ( ( MYROW .EQ. HOMEMESHROW .OR. MYROW .EQ. MODULO(
     $         HOMEMESHROW - 1, NPROW ) ) .AND. MYCOL .NE. HOMEMESHCOL )
     $         THEN
            IF ( NPCOL .GT. 1 ) THEN
*
*              Broadcast receive the transformation matrix along the mesh row.
*              
               CALL DGEBR2D( ICTXT, 'R', ' ', NUMROWS, NUMROWS,
     $            TRAFS(TRAFPTR ), LDTRAF, MYROW, HOMEMESHCOL )
            END IF
         END IF
      END DO



      RETURN

      END

