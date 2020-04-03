subroutine blockslivercolumnupdate( a, desca, IA, m, rowblksz, numgroups, groupinfo, trafs, info )

    implicit none
!      
!     PURPOSE
!     =========
!
!     TODO Describe the purpose.
!     
!     sub( A ) is given by rows 
!     
!     IA : IA + M - 1.
!     
!     The distribution of A partitions sub( A ) into block rows.
!     
!      
!     ARGUMENTS
!     =========
!
!     A        (local input/output) DOUBLE PRECISION array
!              dimension LLD_A x LOCc(N_A)
!
!              The distributed matrix A.
!     
!     DESCA    (global and local input) INTEGER array
!              dimension 9
!     
!              The array descriptor for A.
!     
!     IA       (global input) INTEGER
!     
!              The column location of sub( A ) in A.
!
!     M        (global input) INTEGER
!     
!              The number of columns of sub( A ).
!
!     
!     ROWBLKSZ (global input) INTEGER
!
!              The desired artificial column block size.
!              The column stride of CS.
!
!     NUMGROUPS (global input/output) INTEGER
!
!              The number of groups. Output when QUERY = .TRUE. and
!              input otherwise.
!
!     GROUPINFO (global input/output) INTEGER array
!                 dimension 12 x NUMGROUPS
!
!              Table of information about each rotation group. Each
!              column corresponds to one group. The first (top-most)
!              group corresponds to the first column. The rows of
!              GROUPINFO have the following interpretation:
!     
!              1.  FIRSTCOLUMN - The first colunmn affected by the group.
!              2.  MIDDLECOLUMN - The colunmn below the distribution boundary.
!              3.  LASTCOLUMN - The last colunmn affected by the group.
!              4.  NUMCOLUMN - The number of affected colunmns.
!              5.  COLUMNSLEFT - The number of affected colunmns left of the distribution boundary.
!              6.  COLUMNSRIGHT - The number of affected colunmns right of the distribution boundary.
!              7.  SIZE - The size of the group. 
!              8.  ISLASTGROUP - 1: The last group, 0 otherwise.
!              9.  HOMEMESHROW - The mesh row of the accumulating process.
!              10. HOMEMESHCOL - The mesh column of the accumulating process.
!              11. TRAFPTR - The location of the transformation matrix in memory. 
!              12. LDTRAF - The column stride of the transformation matrix.
!
!     TRAFS    (local input) DOUBLE PRECISION array
!
!              The transformation matrices needed by this process.
!
!     INFO     (global output) INTEGER
!
!              Status output.
!              = 0: successful exit.
!
!
      

  ! DECLARATIONS
  ! ============
  
  ! SCALAR ARGUMENTS

  integer :: m, ia, rowblksz, numgroups, info

  ! ARRAY ARGUMENTS

  integer :: descA(*)
  double precision :: a(*), trafs(*)
  integer :: groupinfo( 12, * )

  ! CONSTANTS
  
  integer, parameter :: ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5, nb_ = 6, rsrc_ = 7, csrc_ = 8, lld_ = 9
  double precision, parameter :: one = 1.0D+0, zero = 0.0D+0
  ! DERIVED TYPES

  type sliver
     integer :: height, r1l, r2l, firstcol, middlecol, slot, currgrp
  end type sliver

  type slot
     integer :: time, stacksize
     integer, pointer :: stack(:)
  end type slot

  ! LOCAL SCALARS

  integer :: numslivers, numslots, timestep
  integer :: udef
  integer :: ictxt, nprow, npcol, myrow, mycol, iam, nprocs, left, right, proc
  integer :: mb, nb, lldA
  integer :: locrow1, locrow2
  integer :: myrightslot, myleftslot, mylocslot
  integer :: myrightsliver, myleftsliver, mylocsliver
  integer :: numcross, numlocal
  integer :: i, j, k, l
  integer :: tmpsliver
  integer :: c1l, c2l, numlocrow
  integer :: trafptr, ldtraf
  integer :: firstcol, middlecol
  integer :: colsleft, colsright, numcols, ldbuf

  logical :: hasrightslot, haslefttslot, haslocslot
  logical :: terminate
  logical :: docross, dolocal
  ! LOCAL ARRAYS
  
  type(sliver), allocatable :: slivers(:)
  type(slot), allocatable :: slots(:)
  double precision, allocatable :: leftbuf(:, :), rightbuf(:, :)

  ! EXTERNALS

  integer, external :: indxl2g, indxg2l, indxg2p, iceil, numroc
  double precision, external:: gettod

  ! INTRINSICS
  
  ! EXECUTABLE STATEMENTS
  ! =====================

  info = 0

  udef = 0


  ! Get process mesh information.
  ictxt = descA(ctxt_)
  call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)
  call blacs_pinfo(iam, nprocs)


  ! Locate neighbours.
  right = modulo(mycol + 1, npcol)
  left = modulo(mycol - 1, npcol)
  

  ! Get the distribution block size.
  mb = descA(mb_)
  nb = descA(nb_)

  ! Get the stride for A and B.
  lldA = descA(lld_)

  ! Allocate temporary buffers.

  ! Find the number of local rows in A. 

  numlocrow = numroc( desca( m_ ), mb, myrow, desca( rsrc_ ), nprow)

      
  ldbuf = max( 1, rowblksz )
  allocate(leftbuf(ldbuf, 2*nb), rightbuf(ldbuf, 2*nb))

  ! Localize the column range JA : IA + M - 1 into locrow1 : locrow2.
  call grn2lrn(IA, IA + m - 1, nb, nprow, myrow, descA(rsrc_), locrow1, locrow2)

  ! Count the number of slivers.
  numslivers = ((locrow2 - locrow1) / rowblksz) + 1

  ! Allocate the sliver states.
  allocate(slivers(numslivers))

  ! Count the number of slots.
  numslots = 2 * npcol

  ! Locate my slots.
  mylocslot = 2 * mycol
  myrightslot = modulo(mylocslot + 1, numslots)
  myleftslot = modulo(mylocslot - 1, numslots)
  mylocslot = mylocslot + 1
  myleftslot = myleftslot + 1
  myrightslot = myrightslot + 1

  ! Allocate the slots.
  allocate( slots ( numslots ) )
  do i = 1, numslots
     allocate(slots( i )%stack( numslivers ))
  end do

  ! Extract cached group information 
  middlecol =   groupinfo( 2, numgroups )
  firstcol =    groupinfo( 1, numgroups ) 
  
  ! Initialize the sliver states.
  do i = 1, numslivers
     if (i == 1) then
        slivers(i)%r1l = locrow1
     else
        slivers(i)%r1l = slivers(i - 1)%r2l + 1
     end if
     slivers( i )%r2l = min(locrow2, slivers( i )%r1l + rowblksz - 1)
     slivers( i )%height = slivers( i )%r2l - slivers( i )%r1l + 1

     
     
     slivers( i )%firstcol = firstcol
     slivers( i )%middlecol = middlecol     

     proc = modulo( 1 + indxg2p( slivers( i )%firstcol, nb, udef, descA( csrc_ ), npcol ), npcol )
     slivers( i )%slot = 2 * proc + 1    

     slivers( i )%currgrp = numgroups

  end do

  ! Initialize the slots.
  do i = 1, numslots
     slots(i)%time = 0
     slots(i)%stacksize = 0
     slots(i)%stack(1:numslivers) = 0
  end do

  ! Add the slivers to the slots.
  do i = 1, numslivers
     ! Skip if the sliver is already depleted.
     if (slivers(i)%currgrp < 1) then
        cycle
     end if

     ! Add the sliver to its slot.
     j = slivers(i)%slot
     k = slots(j)%stacksize + 1
     do while (k > 1)
        if (slivers(slots(j)%stack(k - 1))%middlecol <= slivers(i)%middlecol) then
           exit
        end if
        slots(j)%stack(k) = slots(j)%stack(k - 1)
        k = k - 1
     end do
     slots(j)%stack(k) = i
     slots(j)%stacksize = slots(j)%stacksize + 1
  end do

  ! MAIN LOOP
  
  timestep = 0
  do
     timestep = timestep + 1

     ! COUNT THE NUMBER OF ACTIVATABLE CROSS-BORDER AND LOCAL SLOTS

     numlocal = 0
     do i = 1, numslots, 2
        if (slots(i)%stacksize > 0) then
           numlocal = numlocal + 1
        end if
     end do
     numcross = 0
     do i = 2, numslots, 2
        if (slots(i)%stacksize > 0) then
           numcross = numcross + 1
        end if
     end do

     ! CHOOSE SLOT TYPE

     if (numcross >= numlocal) then
        docross = .true.
        dolocal = .false.
     else
        docross = .false.
        dolocal = .true.
     end if

     ! ACTIVATE SLOTS

     do i = 1, numslots
        ! Skip if empty.
        if (slots(i)%stacksize == 0) then
           cycle
        end if

        ! Skip if the wrong type.
        if ((docross .and. modulo(i, 2) == 1) .or. (dolocal .and. modulo(i, 2) == 0)) then
           cycle;
        end if

        ! Activate this slot.
        slots(i)%time = timestep
     end do

     ! SCHEDULE THE LOCALLY ACTIVE SLOTS

     ! Determine which slots are active locally.
     haslefttslot = (slots(myleftslot)%time == timestep)
     hasrightslot = (slots(myrightslot)%time == timestep)
     haslocslot = (slots(mylocslot)%time == timestep)

     ! Determine which slivers are active locally.
     if (haslefttslot) myleftsliver = slots(myleftslot)%stack(slots(myleftslot)%stacksize)
     if (hasrightslot) myrightsliver = slots(myrightslot)%stack(slots(myrightslot)%stacksize)
     if (haslocslot) mylocsliver = slots(mylocslot)%stack(slots(mylocslot)%stacksize)

     
     ! Step 1 (left cross-border): Send.
     if (.not. haslefttslot) goto 200
     
     
     ! Extract cached group information 
     colsright = groupinfo( 6, slivers(myleftsliver)%currgrp )

     
     c2l = indxg2l(slivers(myleftsliver)%middlecol, nb, udef, desca(csrc_), npcol)
     if ( colsright .gt. 0 ) then
        call dgesd2d( ictxt, slivers(myleftsliver)%height, colsright, A(slivers(myleftsliver)%r1l + ( c2l - 1 ) * lldA), &
             lldA, myrow, left )
     end if
     ! Step 2 (right cross-border): Send.
     200  continue
     

     
     if (.not. hasrightslot) goto 500 
     ! Extract cached group information 
     trafptr =      groupinfo( 11, slivers(myrightsliver)%currgrp )
     ldtraf =       groupinfo( 12, slivers(myrightsliver)%currgrp )
     colsleft =     groupinfo( 5, slivers(myrightsliver)%currgrp )
     colsright =    groupinfo( 6, slivers(myrightsliver)%currgrp )
     numcols =      groupinfo( 4, slivers(myrightsliver)%currgrp )  
     
     c1l = indxg2l( slivers(myrightsliver)%firstcol, nb, udef, desca(csrc_), npcol )

     if ( colsright .gt. 0 ) then
        call dgesd2d( ictxt, slivers(myrightsliver)%height, colsleft, A(slivers(myrightsliver)%r1l  + ( c1l - 1 ) * lldA), &
             lldA, myrow, right )
     end if

     ! Step 3 (right cross-border): Receive A.
     if ( colsright .gt. 0 ) then
        call dgerv2d( ictxt, slivers(myrightsliver)%height, colsright , rightbuf(1, 1 + colsleft) , ldbuf, myrow, right )
     end if
     ! Step 3a (right cross-border): copy my part of A to rightbuf
     call dlacpy( 'A', slivers(myrightsliver)%height, colsleft, A(slivers(myrightsliver)%r1l + ( c1l  - 1 ) * lldA ), lldA, & 
          rightbuf, ldbuf ) 
     
     ! Step 4 (right cross-border): Update A.
     call dgemm( 'N', 'N', slivers(myrightsliver)%height, colsleft , numcols, one, rightbuf, ldbuf, trafs( trafptr ), &
         ldtraf, zero, A(slivers(myrightsliver)%r1l + ( c1l - 1 ) * lldA ), lldA ) 

     
     ! Step 5 (left cross-border): 
     500 continue
     if (.not. haslefttslot) goto 700
     ! Extract cached group information 
     trafptr =      groupinfo( 11, slivers(myleftsliver)%currgrp )
     ldtraf =       groupinfo( 12, slivers(myleftsliver)%currgrp )
     colsleft =     groupinfo( 5, slivers(myleftsliver)%currgrp )
     colsright =    groupinfo( 6, slivers(myleftsliver)%currgrp )
     numcols =      groupinfo( 4, slivers(myleftsliver)%currgrp )  
                
     if ( colsright .gt. 0 ) then           
        ! Receive A.
        call dgerv2d(ictxt, slivers(myleftsliver)%height, colsleft, leftbuf, ldbuf, myrow, left ) 

        ! Step 5a (left cross-border): copy my part of A to leftbuf
        call dlacpy( 'A', slivers(myleftsliver)%height, colsright, A(slivers(myleftsliver)%r1l + ( c2l - 1 ) * lldA ), & 
             lldA, leftbuf( 1, 1 + colsleft ), ldbuf )     

        ! Step 6 (left cross-border): Update.
        call dgemm( 'N', 'N', slivers(myleftsliver)%height, colsright, numcols, one, leftbuf, ldbuf, & 
             trafs( trafptr + colsleft * ldtraf ), ldtraf, zero, A(slivers(myleftsliver)%r1l  + & 
             ( c2l  - 1 ) * lldA ), lldA )     
     end if
     ! Step 7 (local): Update.
     700 continue
     if (.not. haslocslot) goto 800

     ! Done.
800  continue
     
     ! UPDATE THE SLIVERS AND SLOTS


     
     do i = 1, numslots
        ! Skip if not active.
        if (slots(i)%time /= timestep) then
           cycle
        end if
        
        ! Locate the active sliver.
        j = slots(i)%stack(slots(i)%stacksize)

        ! Update the slot.
        slivers(j)%slot = 1 + modulo(slivers(j)%slot - 2, numslots)
        ! Remove the sliver from its current slot.
        slots(i)%stacksize = slots(i)%stacksize - 1
        ! Update the sliver.
        if ( docross ) then
           slivers(j)%currgrp = slivers(j)%currgrp - 1
           ! Skip if depleted.
           if (slivers(j)%currgrp < 1) then
              cycle
           end if
           slivers(j)%middlecol = groupinfo( 2, slivers(j)%currgrp )
           slivers(j)%firstcol = groupinfo( 1, slivers(j)%currgrp )       
        end if
        ! Move the sliver to its new slot.
        l = slivers(j)%slot
        if (l > i .and. slots(l)%time == timestep) then
           ! Temporarily remove the top element.
           tmpsliver = slots(l)%stack(slots(l)%stacksize)
           slots(l)%stacksize = slots(l)%stacksize - 1
        end if
        k = slots(l)%stacksize + 1
        do while (k > 1)
           if (slivers(slots(l)%stack(k - 1))%middlecol <= slivers(j)%middlecol) then
              exit
           end if
           slots(l)%stack(k) = slots(l)%stack(k - 1)
           k = k - 1
        end do
        slots(l)%stack(k) = j
        slots(l)%stacksize = slots(l)%stacksize + 1
        if (l > i .and. slots(l)%time == timestep) then
           ! Replace the old top element.
           slots(l)%stacksize = slots(l)%stacksize + 1
           slots(l)%stack(slots(l)%stacksize) = tmpsliver
        end if
     end do

     ! DETECT TERMINATION

     terminate = .true.
     do i = 1, numslots
        if (slots(i)%stacksize > 0) then
           terminate = .false.
           exit                      
        end if
     end do
     if (terminate) then
        exit
     end if
  end do


  ! Deallocate the slots.
  do i = 1, numslots
     deallocate(slots(i)%stack)
  end do
  deallocate(slots)

  ! Deallocate the sliver states.
  deallocate(slivers)

  ! Deallocate temporary buffers.
  deallocate(rightbuf, leftbuf)

end subroutine blockslivercolumnupdate
