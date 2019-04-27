subroutine sliverrowupdate( m, n, A, iA, jA, descA, colblksz, cvec, svec )
  
  implicit none

  ! DECLARATIONS
  ! ============
  
  ! SCALAR ARGUMENTS

  integer :: m, n, iA, jA, colblksz

  ! ARRAY ARGUMENTS

  integer :: descA(*)
  double precision :: A(*), cvec(*), svec(*) 

  ! CONSTANTS
  
  integer, parameter :: ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5, nb_ = 6, rsrc_ = 7, csrc_ = 8, lld_ = 9
  
  ! DERIVED TYPES

  type sliver
     integer :: width, c1l, c2l, r1g, r2g, slot
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

  logical :: hasbotslot, hastopslot, haslocslot, docross, dolocal, terminate

  ! LOCAL ARRAYS
  
  type(sliver), allocatable :: slivers(:)
  type(slot), allocatable :: slots(:)
  double precision, allocatable :: topbuf(:), botbuf(:)

  ! EXTERNALS

  integer, external :: indxl2g, indxg2l, indxg2p, iceil, numroc

  ! INTRINSICS
  
  ! EXECUTABLE STATEMENTS
  ! =====================

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
  allocate(topbuf(colblksz), botbuf(colblksz))

  ! Localize the column range JA : JA + N - 1 into loccol1 : loccol2.
  call grn2lrn(jA, jA + n - 1, nb, npcol, mycol, descA(csrc_), loccol1, loccol2)

  ! Count the number of slivers.
  numslivers = (loccol2 - loccol1) / colblksz + 1

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
  allocate(slots(numslots))
  do i = 1, numslots
     allocate(slots(i)%stack(numslivers))
  end do

  ! Initialize the sliver states.
  do i = 1, numslivers
     if (i == 1) then
        slivers(i)%c1l = loccol1
     else
        slivers(i)%c1l = slivers(i - 1)%c2l + 1
     end if
     slivers(i)%c2l = min(loccol2, slivers(i)%c1l + colblksz - 1)
     slivers(i)%width = slivers(i)%c2l - slivers(i)%c1l + 1
     slivers(i)%r2g = min(iA + m - 1, indxl2g(slivers(i)%c2l, nb, mycol, descA(csrc_), npcol) + 1)
     proc = indxg2p(slivers(i)%r2g, mb, udef, descA(rsrc_), nprow)
     slivers(i)%slot = 2 * proc + 1
     slivers(i)%r1g = max(iA, slivers(i)%r2g - modulo(slivers(i)%r2g - 1, mb))
     if (slivers(i)%r1g == slivers(i)%r2g) then
        slivers(i)%slot = 1 + modulo(slivers(i)%slot - 2, numslots)
        slivers(i)%r1g = max(iA, slivers(i)%r2g - 1)
     end if
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
     if (slivers(i)%r2g - slivers(i)%r1g + 1 < 2) then
        cycle
     end if

     ! Add the sliver to its slot.
     j = slivers(i)%slot
     k = slots(j)%stacksize + 1
     do while (k > 1)
        if (slivers(slots(j)%stack(k - 1))%r2g <= slivers(i)%r2g) then
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
           cycle
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
     r2l = indxg2l(slivers(mytopsliver)%r2g, mb, udef, udef, nprow)
     call dgesd2d(ictxt, 1, slivers(mytopsliver)%width, A(r2l + (slivers(mytopsliver)%c1l - 1) * lldA), lldA, up, mycol)

     ! Step 2 (bottom cross-border): Send.
     200 continue
     if (.not. hasbotslot) goto 500
     r1l = indxg2l(slivers(mybotsliver)%r1g, mb, udef, udef, nprow)
     call dgesd2d(ictxt, 1, slivers(mybotsliver)%width, A(r1l + (slivers(mybotsliver)%c1l - 1) * lldA), lldA, down, mycol)

     ! Step 3 (bottom cross-border): Receive.
     call dgerv2d(ictxt, 1, slivers(mybotsliver)%width, botbuf, 1, down, mycol)

     ! Step 4 (bottom cross-border): Update.
     call drot(slivers(mybotsliver)%width, A(r1l +(slivers(mybotsliver)%c1l - 1) * lldA), lldA, botbuf, 1, &
          cvec(slivers(mybotsliver)%r2g), svec(slivers(mybotsliver)%r2g))

     
     ! Step 5 (top cross-border): Receive.
     500 continue
     if (.not. hastopslot) goto 700
     call dgerv2d(ictxt, 1, slivers(mytopsliver)%width, topbuf, 1, up, mycol)

     ! Step 6 (top cross-border): Update.
     call drot(slivers(mytopsliver)%width, topbuf, 1, A(r2l + (slivers(mytopsliver)%c1l - 1) * lldA), lldA, &
          cvec(slivers(mytopsliver)%r2g), svec(slivers(mytopsliver)%r2g))

     ! Step 7 (local): Update.
     700 continue
     if (.not. haslocslot) goto 800
     r1l = indxg2l(slivers(mylocsliver)%r1g, mb, udef, udef, nprow)
     r2l = indxg2l(slivers(mylocsliver)%r2g, mb, udef, udef, nprow)
     call krnlrowupdate(r2l - r1l + 1, slivers(mylocsliver)%width, A(r1l + (slivers(mylocsliver)%c1l - 1) * lldA), lldA, &
          cvec(slivers(mylocsliver)%r1g), svec(slivers(mylocsliver)%r1g) )

     ! Done.
     800 continue
     
     ! UPDATE THE SLIVERS AND SLOTS

     do i = 1, numslots
        ! Skip if not active.
        if (slots(i)%time /= timestep) then
           cycle
        end if

        ! Locate the active sliver.
        j = slots(i)%stack(slots(i)%stacksize)
        
        ! Remove the sliver from its current slot.
        slots(i)%stacksize = slots(i)%stacksize - 1

        ! Update the sliver.
        slivers(j)%slot = 1 + modulo(slivers(j)%slot - 2, numslots)
        slivers(j)%r2g = slivers(j)%r1g
        if (modulo(slivers(j)%slot, 2) == 0) then
           slivers(j)%r1g = max(iA, slivers(j)%r2g - 1)
        else
           slivers(j)%r1g = max(iA, slivers(j)%r2g - mb + 1)
        end if

        ! Skip if the sliver is depleted.
        if (slivers(j)%r2g - slivers(j)%r1g + 1 < 2) then
           cycle
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
           if (slivers(slots(l)%stack(k - 1))%r2g <= slivers(j)%r2g) then
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

  
end subroutine sliverrowupdate

