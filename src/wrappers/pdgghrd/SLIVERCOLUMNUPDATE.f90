subroutine slivercolumnupdate(m, n, A, iA, jA, descA, cvec, svec, rowblksz)
  
  implicit none

  ! DECLARATIONS
  ! ============
  
  ! SCALAR ARGUMENTS

  integer :: m, n, iA, jA, rowblksz

  ! ARRAY ARGUMENTS

  integer :: descA(*)
  double precision :: A(*), cvec(*), svec(*) 

  ! CONSTANTS
  
  integer, parameter :: ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5, nb_ = 6, rsrc_ = 7, csrc_ = 8, lld_ = 9
  
  ! DERIVED TYPES

  type sliver
     integer :: c1g, c2g
     integer :: r1l, r2l
     integer :: height
     integer :: slot
  end type sliver

  type slot
     integer :: time
     integer, allocatable :: stack(:)
     integer :: stacksize
  end type slot

  ! LOCAL SCALARS

  integer :: numslivers, numslots, timestep
  integer :: udef
  integer :: ictxt, nprow, npcol, myrow, mycol, iam, nprocs, left, right, proc
  integer :: mb, nb, lldA  
  integer :: myleftslot, myrightslot, mylocslot
  integer :: myleftsliver, myrightsliver, mylocsliver
  integer :: i, k, l, q 
  integer :: tmpsliver
  integer :: c1l, c2l, r1l, r2l
  integer :: numactiveslots
  integer :: numcross, numlocal

  logical :: hasleftslot, hasrightslot, haslocslot
  logical :: terminate
  logical :: docross, dolocal
  
  ! LOCAL ARRAYS
  
  type(sliver), allocatable :: slivers(:)
  type(slot), allocatable :: slots(:)
  double precision, allocatable :: leftbuf(:), rightbuf(:)

  ! EXTERNALS

  integer, external :: indxl2g, indxg2l, indxg2p, iceil, numroc

  ! EXECUTABLE STATEMENTS
  ! =====================


  udef = 0


  ! Get process mesh information.
  ictxt = descA(ctxt_)
  call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)
  call blacs_pinfo(iam, nprocs)



  ! Locate neighbours.
  right = modulo(mycol + 1, npcol)
  left  = modulo(mycol - 1, npcol)

  ! Get the distribution block size.
  mb = descA(mb_)
  nb = descA(nb_)

  ! Get the stride for A.
  lldA = descA(lld_)

  ! Allocate temporary buffers.
  allocate(leftbuf(rowblksz), rightbuf(rowblksz))

  ! Count the number of slivers.
  call grn2lrn(iA, iA + m - 1, mb, nprow, myrow, descA(rsrc_), r1l, r2l)
  numslivers = (r2l - r1l) / rowblksz + 1

  ! Allocate the sliver states.
  allocate(slivers(numslivers))

  ! Count the number of slots.
  numslots = 2 * npcol

  ! Locate my slots.
  mylocslot   = 2 * mycol
  myrightslot = modulo(mylocslot + 1, numslots)
  myleftslot  = modulo(mylocslot - 1, numslots)
  mylocslot   = mylocslot + 1
  myrightslot = myrightslot + 1
  myleftslot  = myleftslot + 1

  ! Allocate the slots.
  allocate(slots(numslots))
  do i = 1, numslots
     allocate(slots(i)%stack(numslivers))
  end do

  ! Initialize the sliver states.
  do i = 1, numslivers
     if (i == 1) then
        slivers(i)%r1l = r1l
     else
        slivers(i)%r1l = slivers(i - 1)%r2l + 1
     end if
     slivers(i)%r2l = min(r2l, slivers(i)%r1l + rowblksz - 1)
     slivers(i)%height = slivers(i)%r2l - slivers(i)%r1l + 1
     slivers(i)%c2g = min(jA + n - 1, (((jA + n - 1) - 1) / nb) * nb + nb)
     slivers(i)%c1g = max(jA, ((slivers(i)%c2g - 1) / nb) * nb + 1)
     proc = indxg2p(slivers(i)%c2g, nb, udef, descA(csrc_), npcol)
     slivers(i)%slot = 2 * proc + 1
     if (slivers(i)%c1g == slivers(i)%c2g) then
        slivers(i)%c1g = max(jA, slivers(i)%c2g - 1)
        slivers(i)%slot = 1 + modulo(slivers(i)%slot - 2, numslots)
     end if
  end do

  ! Initialize the slots.
  do i = 1, numslots
     slots(i)%time = 0
     slots(i)%stack(1:numslivers) = 0
     slots(i)%stacksize = 0
  end do

  ! Add the slivers to the slots.
  do i = 1, numslivers
     ! Skip if the sliver is already depleted.
     if (slivers(i)%c2g - slivers(i)%c1g + 1 < 2) then
        cycle
     end if

     ! Add the sliver to its slot.
     k = slivers(i)%slot
     l = slots(k)%stacksize + 1
     do while (l > 1)
        if (slivers(slots(k)%stack(l - 1))%c2g <= slivers(i)%c2g) then
           exit
        end if
        slots(k)%stack(l) = slots(k)%stack(l - 1)
        l = l - 1
     end do
     slots(k)%stack(l) = i
     slots(k)%stacksize = slots(k)%stacksize + 1
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


     ! CHOOSE SLOT TYPES

     if (numcross >= numlocal) then
        docross = .true.
        dolocal = .false.
     else
        docross = .false.
        dolocal = .true.
     end if

     ! ACTIVATE SLOTS

     numactiveslots = 0
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
        numactiveslots = numactiveslots + 1
     end do

     ! SCHEDULE THE LOCALLY ACTIVE SLOTS

     ! Determine which slots are active locally.
     hasleftslot  = (slots(myleftslot)%time  == timestep)
     hasrightslot = (slots(myrightslot)%time == timestep)
     haslocslot   = (slots(mylocslot)%time   == timestep)

     ! Determine which slivers are active locally.
     if (hasleftslot)  myleftsliver  = slots(myleftslot)%stack (slots(myleftslot)%stacksize)
     if (hasrightslot) myrightsliver = slots(myrightslot)%stack(slots(myrightslot)%stacksize)
     if (haslocslot)   mylocsliver   = slots(mylocslot)%stack  (slots(mylocslot)%stacksize)

     ! Step 1 (left cross-border): Send.
     if (.not. hasleftslot) goto 200
     c2l = indxg2l(slivers(myleftsliver)%c2g, nb, udef, udef, npcol)
     call dgesd2d(ictxt, slivers(myleftsliver)%height, 1, &
          A(slivers(myleftsliver)%r1l + (c2l - 1) * lldA), lldA, myrow, left)

     ! Step 2 (right cross-border): Send.
     200 continue
     if (.not. hasrightslot) goto 500
     c1l = indxg2l(slivers(myrightsliver)%c1g, nb, udef, udef, npcol)
     call dgesd2d(ictxt, slivers(myrightsliver)%height, 1, &
          A(slivers(myrightsliver)%r1l + (c1l - 1) * lldA), lldA, myrow, right)

     ! Step 3 (right cross-border): Receive.
     call dgerv2d(ictxt, slivers(myrightsliver)%height, 1, rightbuf, rowblksz, myrow, right)

     ! Step 4 (right cross-border): Update.
     call drot(slivers(myrightsliver)%height, rightbuf, 1, A(slivers(myrightsliver)%r1l +(c1l - 1) * lldA), 1, &
          cvec(slivers(myrightsliver)%c2g), svec(slivers(myrightsliver)%c2g))

     
     ! Step 5 (left cross-border): Receive.
     500 continue
     if (.not. hasleftslot) goto 700
     call dgerv2d(ictxt, slivers(myleftsliver)%height, 1, leftbuf, rowblksz, myrow, left)

     ! Step 6 (left cross-border): Update.
     call drot(slivers(myleftsliver)%height, A(slivers(myleftsliver)%r1l + (c2l - 1) * lldA), 1, leftbuf, 1, &
          cvec(slivers(myleftsliver)%c2g), svec(slivers(myleftsliver)%c2g))


     ! Step 7 (local): Update.
     700 continue
     if (.not. haslocslot) goto 800
     c1l = indxg2l(slivers(mylocsliver)%c1g, nb, udef, udef, npcol)
     c2l = indxg2l(slivers(mylocsliver)%c2g, nb, udef, udef, npcol)
     call krnlcolumnupdate(slivers(mylocsliver)%height, c2l - c1l + 1, A(slivers(mylocsliver)%r1l + (c1l - 1) * lldA), lldA, &
          cvec(slivers(mylocsliver)%c1g), svec(slivers(mylocsliver)%c1g))

     ! Done.
     800 continue
     
     ! UPDATE THE SLIVERS AND SLOTS

     do i = 1, numslots
        ! Skip if the sliver is not active.
        if (slots(i)%time /= timestep) then
           cycle
        end if

        ! Locate the active sliver.
        k = slots(i)%stack(slots(i)%stacksize)
        
        ! Remove the sliver for a moment (possibly permanently).
        slots(i)%stacksize = slots(i)%stacksize - 1

        ! Update the sliver.
        slivers(k)%slot = 1 + modulo(slivers(k)%slot - 2, numslots)
        slivers(k)%c2g = max(jA, slivers(k)%c1g)
        if (modulo(slivers(k)%slot, 2) == 0) then
           ! Cross-border.
           slivers(k)%c1g = max(jA, slivers(k)%c2g - 1)
        else
           ! Local.
           slivers(k)%c1g = max(jA, slivers(k)%c2g - nb + 1)
        end if

        ! Skip if the sliver has now become depleted.
        if (slivers(k)%c2g - slivers(k)%c1g + 1 < 2) then
           cycle
        end if

        ! Move the sliver to its new slot. 
        l = slivers(k)%slot
        if (l > i .and. slots(l)%time == timestep) then
           ! Temporarily remove the top element on the new slot.
           tmpsliver = slots(l)%stack(slots(l)%stacksize)
           slots(l)%stacksize = slots(l)%stacksize - 1
        end if
        q = slots(l)%stacksize + 1
        do while (q > 1)
           if (slivers(slots(l)%stack(q - 1))%c2g <= slivers(k)%c2g) then
              exit
           end if
           slots(l)%stack(q) = slots(l)%stack(q - 1)
           q = q - 1
        end do
        slots(l)%stack(q) = k
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
  deallocate(leftbuf, rightbuf)


  
end subroutine slivercolumnupdate
