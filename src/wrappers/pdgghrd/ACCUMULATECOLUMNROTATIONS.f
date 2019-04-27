***********************************************************************
*                                                                     *
*     ACCUMULATECOLUMNROTATIONS.f:                                    *
*     Auxillary routine in the package PDGGHRD.                       *
*                                                                     *
*     Contributors: Björn Adlerborn                                   *
*                   Lars Karlsson                                     *
*                                                                     *
*     Department of Computing Science and HPC2N, Umea University      *
*                                                                     *
*                                                                     * 
***********************************************************************
      SUBROUTINE ACCUMULATECOLUMNROTATIONS( QUERY, J0, N, NUMSEQ, DESC,
     $   CS, LDCS, NUMGROUPS, GROUPINFO, TRAFS, GTRAFPTR, INFO )
      
      IMPLICIT NONE
      
*
*     PURPOSE
*     =======
*
*     Forms and accumulates groups of rotations into orthogonal
*     matrices. TODO Expand the description of the purpose.
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
*     J0          (global input) INTEGER
*
*                 The first column affected by the rotations.
*     
*     N           (global input) INTEGER
*
*                 The number of columns affected by the rotations.
*
*     NUMSEQ      (global input) INTEGER
*
*                 The number of rotation sequences.
*
*     DESC        (global/local input) INTEGER array
*                 dimension 9
*
*                 Descriptor for the matrix to be updated from the right
*                 by the accumulated rotations.
*     
*     CS          (global input) DOUBLE PRECISION array
*                 dimension DESC( N_ ) x 2 * NUMSEQ
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
*                 column corresponds to one group. The first group
*                 corresponds to the first column. The rows of GROUPINFO
*                 have the following interpretation:
*     
*                 1.  FIRSTCOLUMN - The first column affected by the group.
*                 2.  MIDDLECOLUMN - The column to the right of the distribution boundary.
*                 3.  LASTCOLUMN - The last coulmn affected by the group.
*                 4.  NUMCOLUMNS - The number of affected columns.
*                 5.  COLUMNSLEFT - The number of affected columns left of the distribution boundary.
*                 6.  COULMNSRIGHT - The number of affected colmns right of the distribution boundary.
*                 7.  SIZE - The size of the group. 
*                 8.  ISLASTGROUP - 1: The last group, 0 otherwise.
*                 9.  HOMEMESHROW - The mesh row of the accumulating process.
*                 10. HOMEMESHCOL - The mesh column of the accumulating process.
*                 11. TRAFPTR - The location of the transformation matrix in memory. 
*                 12. LDTRAF - The column stride of the transformation matrix.
*
*                 An upper bound for the number of required columns in GROUPINFO is
*                 ICEIL ( DESCA( N_ ), DESC( NB_ ) ) + 1
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
      INTEGER N, J0, NUMSEQ, LDCS, INFO, GTRAFPTR
      LOGICAL QUERY


*
*     Array arguments.
*     
      INTEGER DESC( 9 ), GROUPINFO( 12, * )
      DOUBLE PRECISION CS( LDCS, * ), TRAFS ( * )

*
*     Local scalars.
*     
      INTEGER UDEF, ICTXT, NPROW, NPCOL, MYROW, MYCOL, IAM, NPROCS, NB,
     $   NUMGROUPS, LDTRAF, GROUP, LEFT, RIGHT, FIRSTCOLUMN,
     $   MIDDLECOLUMN ,LASTCOLUMN , SIZE, COLUMNSLEFT, COLUMNSRIGHT,
     $   NUMCOLUMNS, HOMEMESHROW, HOMEMESHCOL, TRAFPTR
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
*    
*     rots     x x x | c 
*                x x | c x 
*                  x | c x x
*                    | c x x x
*                    |     x x  
*                    |       x     
*     column 1 2 3 4 | 5 6 7 8
*            =       = =     =
*            F       d M     L
*            I       i I     A
*            R       s D     S
*            S       t D     T
*            T       . L     C
*            C       b E     O
*            O       o C     L
*            L       u O     U
*            U       n L     M
*            M       d U     N
*            N       a M
*                    r N
*                    y
*
*     SIZE = 4
*     ISLASTGROUP = .FALSE.
*     COLUMNSLEFT = 4
*     COLUMNSRIGHT = 4
*     NUMCOLUMNS = 8
*

*
*     Get process mesh information.
*     
      ICTXT = DESC( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      CALL BLACS_PINFO( IAM, NPROCS )

*
*     Get the mesh rows of the left and right neighbours.
*
      LEFT = MODULO( MYCOL - 1, NPCOL )
      RIGHT = MODULO( MYCOL + 1, NPCOL )

*
*     Extract the distribution column block size.
*     
      NB = DESC( NB_ )

*
*     Skip the query part unless requested.
*     
      IF ( .NOT. QUERY )  GOTO 100

*
*     Reset the group counter.
* 
      NUMGROUPS = 0

*
*     Set the offset in memory of the first transformation.
*     
      GTRAFPTR = 1
      
*
*     Find the first affected column of the first group.
*
      FIRSTCOLUMN = J0
      
*
*     Always use process 0 as start for home mesh row
*
      HOMEMESHROW = 0
      
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
*        Locate the first column on the right of the next distribution block boundary.
*        
         MIDDLECOLUMN = 1 + NB * ((FIRSTCOLUMN + NB - 1) / NB)
*        Crop to submatrix boundary if necessary.
         IF ( MIDDLECOLUMN .GT. J0 + N - 1 ) THEN
            MIDDLECOLUMN = J0 + N
         END IF

*        
*        Compute the size of the group.
*        
         SIZE = MIDDLECOLUMN - FIRSTCOLUMN

*        
*        Locate the last column affected by the group.
*        
         LASTCOLUMN = MIDDLECOLUMN + NUMSEQ - 1
*        Crop to submatrix boundary if necessary.
         IF ( LASTCOLUMN .GT. J0 + N - 1 ) THEN
            LASTCOLUMN = J0 + N - 1
         END IF

*        
*        Determine if this is the last group.
*        
         ISLASTGROUP = (LASTCOLUMN .EQ. J0 + N - 1)
         
*        
*        Compute the number of columns affected to the left and to the
*        right of the distribution boundary, respectively.
*        
         COLUMNSLEFT = MIDDLECOLUMN - FIRSTCOLUMN
         COLUMNSRIGHT = LASTCOLUMN - MIDDLECOLUMN + 1

*        
*        Compute the total number of affected columns. 
*        
         NUMCOLUMNS = LASTCOLUMN - FIRSTCOLUMN + 1

*
*        Locate the home mesh column of this group (i.e., the mesh
*        column to the right of the distribution block boundary).
*
         HOMEMESHCOL = MOD( 1 + INDXG2P( FIRSTCOLUMN, NB, UDEF,
     $      DESC(CSRC_ ), NPCOL ), NPCOL )

*
*        Populate a column of the GROUPINFO array.
*        
         GROUPINFO( 1, NUMGROUPS ) = FIRSTCOLUMN
         GROUPINFO( 2, NUMGROUPS ) = MIDDLECOLUMN
         GROUPINFO( 3, NUMGROUPS ) = LASTCOLUMN
         GROUPINFO( 4, NUMGROUPS ) = NUMCOLUMNS
         GROUPINFO( 5, NUMGROUPS ) = COLUMNSLEFT
         GROUPINFO( 6, NUMGROUPS ) = COLUMNSRIGHT
         GROUPINFO( 7, NUMGROUPS ) = SIZE
         IF ( ISLASTGROUP ) THEN
            GROUPINFO( 8, NUMGROUPS ) = 1
         ELSE
            GROUPINFO( 8, NUMGROUPS ) = 0
         END IF
         GROUPINFO( 9, NUMGROUPS ) = MOD( HOMEMESHROW + ( NUMGROUPS - 1)
     $      / NPROW, NPROW )
         GROUPINFO( 10, NUMGROUPS ) = HOMEMESHCOL
         IF ( MYCOL .EQ. HOMEMESHCOL .OR. MYCOL .EQ. MODULO( HOMEMESHCOL
     $      - 1, NPCOL ) ) THEN
*
*           I need local storage for the transformation.
*
            LDTRAF = NUMCOLUMNS
            GROUPINFO( 11, NUMGROUPS ) = GTRAFPTR
            GROUPINFO( 12, NUMGROUPS ) = LDTRAF
            GTRAFPTR = GTRAFPTR + (NUMCOLUMNS * NUMCOLUMNS)
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
         FIRSTCOLUMN = MIDDLECOLUMN    

*
*        Locate the next home mesh row.
*
         HOMEMESHROW = MOD( HOMEMESHROW + 1, NPROW )
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
         FIRSTCOLUMN   = GROUPINFO( 1, GROUP )
         MIDDLECOLUMN  = GROUPINFO( 2, GROUP )
         LASTCOLUMN    = GROUPINFO( 3, GROUP )
         NUMCOLUMNS    = GROUPINFO( 4, GROUP )
         COLUMNSLEFT   = GROUPINFO( 5, GROUP )
         COLUMNSRIGHT  = GROUPINFO( 6, GROUP )
         SIZE          = GROUPINFO( 7, GROUP )
         ISLASTGROUP   = GROUPINFO( 8, GROUP ) .EQ. 1
         HOMEMESHROW   = GROUPINFO( 9, GROUP )
         HOMEMESHCOL   = GROUPINFO( 10, GROUP )
         TRAFPTR       = GROUPINFO( 11, GROUP )
         LDTRAF        = GROUPINFO( 12, GROUP )

*        
*        If I am the home process, then ...
*
         IF ( MYROW .EQ. HOMEMESHROW .AND. MYCOL .EQ. HOMEMESHCOL ) THEN
*           
*           Reset the transformation matrix to the identity matrix.
*
            CALL KRNLACCUMULATECOLUMNROTATIONS( NUMCOLUMNS, NUMSEQ, SIZE
     $         , TRAFS( TRAFPTR ), LDTRAF, CS( FIRSTCOLUMN, 1 ), LDCS,
     $         ISLASTGROUP )
         END IF
      END DO

*
*     Loop over the groups. 
*
      DO GROUP = 1, NUMGROUPS
*
*        Extract cached group information.
*        
         NUMCOLUMNS  = GROUPINFO( 4, GROUP )
         HOMEMESHROW = GROUPINFO( 9, GROUP )
         HOMEMESHCOL = GROUPINFO( 10, GROUP )
         TRAFPTR     = GROUPINFO( 11, GROUP )
         LDTRAF      = GROUPINFO( 12, GROUP )
         
*
*        If I am the home process, then ...
*        
         IF ( MYROW .EQ. HOMEMESHROW .AND. MYCOL .EQ. HOMEMESHCOL ) THEN
            IF ( NPCOL .GT. 1 ) THEN
*              
*              Send the transformation matrix to the left. 
*              
               CALL DGESD2D( ICTXT, NUMCOLUMNS, NUMCOLUMNS,
     $            TRAFS(TRAFPTR), LDTRAF, MYROW, LEFT )
            END IF

            IF ( NPROW .GT. 1 ) THEN
*              
*              Broadcast the transformation matrix along the mesh column.
*              
               CALL DGEBS2D( ICTXT, 'C', ' ', NUMCOLUMNS, NUMCOLUMNS,
     $            TRAFS( TRAFPTR ), LDTRAF )
            END IF

*
*        else If I am left of the home process, then ...
*        
         ELSE IF ( MYCOL .EQ. MODULO( HOMEMESHCOL - 1, NPCOL ) .AND.
     $         MYROW .EQ. HOMEMESHROW ) THEN
            IF ( NPCOL .GT. 1 ) THEN
*
*              Receive the transformation matrix from the right.
*              
               CALL DGERV2D( ICTXT, NUMCOLUMNS, NUMCOLUMNS, TRAFS(
     $            TRAFPTR ), LDTRAF, MYROW, RIGHT )
            END IF

            IF ( NPROW .GT. 1 ) THEN
*              
*              Broadcast the transformation matrix along the mesh column.
*              
               CALL DGEBS2D( ICTXT, 'C', ' ', NUMCOLUMNS, NUMCOLUMNS,
     $            TRAFS( TRAFPTR ), LDTRAF )
            END IF
            
*
*        else If I will need this transformation later but am not on the home
*        mesh column, then ...
*
         ELSE IF ( ( MYCOL .EQ. HOMEMESHCOL .OR. MYCOL .EQ.
     $         MODULO(HOMEMESHCOL - 1, NPCOL ) ) .AND. MYROW .NE.
     $         HOMEMESHROW ) THEN
            IF ( NPROW .GT. 1 ) THEN
*
*              Broadcast receive the transformation matrix along the mesh row.
*
               CALL DGEBR2D( ICTXT, 'C', ' ', NUMCOLUMNS, NUMCOLUMNS,
     $            TRAFS( TRAFPTR ), LDTRAF, HOMEMESHROW, MYCOL )
            END IF
         END IF
      END DO

      RETURN

      END
