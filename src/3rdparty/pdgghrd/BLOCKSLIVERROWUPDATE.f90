subroutine blocksliverrowupdate( a, desca, ja, n, colblksz, numgroups, groupinfo, trafs, info )

    implicit none
!      
!     PURPOSE
!     =========
!
!     TODO Describe the purpose.
!     
!     sub( A ) is given by columns
!     
!     JA : JA + N - 1.
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
!     JA       (global input) INTEGER
!     
!              The column location of sub( A ) in A.
!
!     N        (global input) INTEGER
!     
!              The number of columns of sub( A ).
!
!     
!     COLBLKSZ (global input) INTEGER
!
!              The desired artificial column block size.
!              The column stride of CS.
!
!     NUMGROUPS (global input) INTEGER
!
!              The number of groups. 
!
!     GROUPINFO (global input/output) INTEGER array
!                 dimension 12 x NUMGROUPS
!
!              Table of information about each rotation group. Each
!              column corresponds to one group. The first (top-most)
!              group corresponds to the first column. The rows of
!              GROUPINFO have the following interpretation:
!     
!              1.  FIRSTROW - The first row affected by the group.
!              2.  MIDDLEROW - The row below the distribution boundary.
!              3.  LASTROW - The last row affected by the group.
!              4.  NUMROWS - The number of affected rows.
!              5.  ROWSABOVE - The number of affected rows above the distribution boundary.
!              6.  ROWSBELOW - The number of affected rows below the distribution boundary.
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

  integer :: n, ja, colblksz, numgroups, info

  ! ARRAY ARGUMENTS

  integer :: descA(*)
  double precision :: a(*), trafs(*)
  integer :: groupinfo( 12, * )

  ! CONSTANTS
  
  integer, parameter :: ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5, nb_ = 6, rsrc_ = 7, csrc_ = 8, lld_ = 9
  double precision, parameter :: one = 1.0D+0, zero = 0.0D+0
  ! DERIVED TYPES

  type sliver
     integer :: width, c1l, c2l, firstrow, middlerow, slot, currgrp
  end type sliver

  type slot
     integer :: time, stacksize
     integer, pointer :: stack(:)
  end type slot

  ! LOCAL SCALARS

  integer :: numslivers, numslots, timestep
  integer :: udef
  integer :: ictxt, nprow, npcol, myrow, mycol, iam, nprocs, down, up, proc
  integer :: mb, nb, lldA
  integer :: loccol1, loccol2
  integer :: mybotslot, mytopslot, mylocslot
  integer :: mybotsliver, mytopsliver, mylocsliver
  integer :: numcross, numlocal
  integer :: i, j, k, l
  integer :: tmpsliver
  integer :: r1l, r2l
  integer :: trafptr, ldtraf
  integer :: firstrow, middlerow 
  integer :: rowsabove, rowsbelow, numrows, ldbuf

  logical :: terminate
  logical :: docross, dolocal
  logical :: hasbotslot, hastopslot, haslocslot
  ! LOCAL ARRAYS
  
  type(sliver), allocatable :: slivers(:)
  type(slot), allocatable :: slots(:)
  double precision, allocatable :: topbuf(:, :), botbuf(:, :)

  ! EXTERNALS

  integer, external :: indxl2g, indxg2l, indxg2p, iceil, numroc

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
  down = modulo(myrow + 1, nprow)
  up   = modulo(myrow - 1, nprow)

  ! Get the distribution block size.
  mb = descA(mb_)
  nb = descA(nb_)

  ! Get the stride for A.
  lldA = descA(lld_)

  ! Allocate temporary buffers.
  ldbuf = 2*nb
  allocate(topbuf(ldbuf, colblksz), botbuf(ldbuf, colblksz))

  ! Localize the column range JA : JA + N - 1 into loccol1 : loccol2.
  call grn2lrn(jA, jA + n - 1, nb, npcol, mycol, descA(csrc_), loccol1, loccol2)

  ! Count the number of slivers.
  numslivers = ((loccol2 - loccol1) / colblksz) + 1

  ! Allocate the sliver states.
  allocate(slivers(numslivers))

  ! Count the number of slots.
  numslots = 2 * nprow

  ! Locate my slots.
  mylocslot = 2 * myrow
  mybotslot = modulo(mylocslot + 1, numslots)
  mytopslot = modulo(mylocslot - 1, numslots)
  mylocslot = mylocslot + 1
  mytopslot = mytopslot + 1
  mybotslot = mybotslot + 1

  ! Allocate the slots.
  allocate( slots ( numslots ) )
  do i = 1, numslots
     allocate(slots( i )%stack( numslivers ))
  end do

  ! Extract cached group information 
  middlerow =   groupinfo( 2, numgroups )
  firstrow =    groupinfo( 1, numgroups ) 
  
  ! Initialize the sliver states.
  do i = 1, numslivers
     if (i == 1) then
        slivers(i)%c1l = loccol1
     else
        slivers(i)%c1l = slivers(i - 1)%c2l + 1
     end if
     slivers( i )%c2l = min(loccol2, slivers( i )%c1l + colblksz - 1)
     slivers( i )%width = slivers( i )%c2l - slivers( i )%c1l + 1

     
     
     slivers( i )%firstrow = firstrow
     slivers( i )%middlerow = middlerow     

     proc = modulo( 1 + indxg2p( slivers( i )%firstrow, mb, udef, descA( rsrc_ ), nprow ), nprow )
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
        if (slivers(slots(j)%stack(k - 1))%middlerow <= slivers(i)%middlerow) then
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
     hastopslot = (slots(mytopslot)%time == timestep)
     hasbotslot = (slots(mybotslot)%time == timestep)
     haslocslot = (slots(mylocslot)%time == timestep)

     ! Determine which slivers are active locally.
     if (hastopslot) mytopsliver = slots(mytopslot)%stack(slots(mytopslot)%stacksize)
     if (hasbotslot) mybotsliver = slots(mybotslot)%stack(slots(mybotslot)%stacksize)
     if (haslocslot) mylocsliver = slots(mylocslot)%stack(slots(mylocslot)%stacksize)

     
     ! Step 1 (top cross-border): Send.
     if (.not. hastopslot) goto 200
     
     
     ! Extract cached group information 
     rowsbelow = groupinfo( 6, slivers(mytopsliver)%currgrp )

     
     r2l = indxg2l(slivers(mytopsliver)%middlerow, mb, udef, desca(rsrc_), nprow)
     if ( rowsbelow .gt. 0 ) then
        call dgesd2d(ictxt, rowsbelow, slivers(mytopsliver)%width, A(r2l + (slivers(mytopsliver)%c1l - 1) * lldA), & 
             lldA, up, mycol)
     end if
     ! Step 2 (bottom cross-border): Send.
     200  continue
     

     
     if (.not. hasbotslot) goto 500 
     ! Extract cached group information 
     trafptr =      groupinfo( 11, slivers(mybotsliver)%currgrp )
     ldtraf =       groupinfo( 12, slivers(mybotsliver)%currgrp )
     rowsabove =    groupinfo( 5, slivers(mybotsliver)%currgrp )
     rowsbelow =    groupinfo( 6, slivers(mybotsliver)%currgrp )
     numrows =      groupinfo( 4, slivers(mybotsliver)%currgrp )  
     
     r1l = indxg2l( slivers(mybotsliver)%firstrow, mb, udef, desca(rsrc_), nprow )
     if ( rowsbelow .gt. 0 ) then
        call dgesd2d( ictxt, rowsabove, slivers(mybotsliver)%width, A(r1l + (slivers(mybotsliver)%c1l - 1) * lldA), & 
             lldA, down, mycol )
     end if
     ! Step 3 (bottom cross-border): Receive.
     if ( rowsbelow .gt. 0 ) then
        call dgerv2d( ictxt, rowsbelow, slivers(mybotsliver)%width, botbuf(1 + rowsabove, 1) , ldbuf, down, mycol )
     end if
     ! Step 3a (bottom cross-border): copy my part of A to botbuf
     call dlacpy( 'A', rowsabove, slivers(mybotsliver)%width, A(r1l + ( slivers(mybotsliver)%c1l - 1 ) * lldA ), & 
          lldA, botbuf, ldbuf ) 
     
     
     call dgemm( 'N', 'N', rowsabove, slivers(mybotsliver)%width, numrows, one, trafs( trafptr ), ldtraf , & 
          botbuf, ldbuf, zero, A(r1l + ( slivers(mybotsliver)%c1l - 1 ) * lldA ), lldA ) 
     
     
     ! Step 5 (top cross-border): Receive.
     500 continue
     if (.not. hastopslot) goto 700
     ! Extract cached group information 
     trafptr =      groupinfo( 11, slivers(mytopsliver)%currgrp )
     ldtraf =       groupinfo( 12, slivers(mytopsliver)%currgrp )
     rowsabove =    groupinfo( 5, slivers(mytopsliver)%currgrp )
     rowsbelow =    groupinfo( 6, slivers(mytopsliver)%currgrp )
     numrows =      groupinfo( 4, slivers(mytopsliver)%currgrp )  
                
     if ( rowsbelow .gt. 0 ) then           
        call dgerv2d(ictxt, rowsabove, slivers(mytopsliver)%width, topbuf, ldbuf, up, mycol)

        ! Step 5a (top cross-border): copy my part of A to topbuf
        call dlacpy( 'A', rowsbelow, slivers(mytopsliver)%width, A(r2l + ( slivers(mytopsliver)%c1l - 1 ) * lldA ), & 
             lldA, topbuf( 1 + rowsabove, 1 ), ldbuf )     

        ! Step 6 (top cross-border): Update.
        call dgemm( 'N', 'N', rowsbelow, slivers(mytopsliver)%width, numrows, one, trafs( trafptr + rowsabove ), & 
             ldtraf, topbuf, ldbuf, zero, A(r2l + ( slivers(mytopsliver)%c1l - 1 ) * lldA ), lldA )     
        
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
           slivers(j)%middlerow = groupinfo( 2, slivers(j)%currgrp )
           slivers(j)%firstrow = groupinfo( 1, slivers(j)%currgrp )       
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
           if (slivers(slots(l)%stack(k - 1))%middlerow <= slivers(j)%middlerow) then
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
  deallocate(topbuf, botbuf)

  
end subroutine blocksliverrowupdate
