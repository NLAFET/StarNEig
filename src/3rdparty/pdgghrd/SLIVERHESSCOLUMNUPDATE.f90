subroutine sliverhesscolumnupdate(m, n, A, iA, jA, descA &
     , cvec, svec )

  implicit none

  ! DECLARATIONS
  ! ============

  ! SCALAR ARGUMENTS

  integer :: m, n, iA, jA

  ! ARRAY ARGUMENTS

  integer :: descA(*)
  double precision :: A(*), cvec(*), svec(*)

  ! CONSTANTS

  integer, parameter :: ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5 &
       , nb_ = 6, rsrc_ = 7, csrc_ = 8, lld_ = 9

  ! DERIVED TYPES

  type sliver
     integer :: c1g, c2g, r1g, r2g
     integer :: r1l, r2l
     integer :: height
     integer :: slot
     integer :: meshrow
  end type sliver

  type slot
     integer :: time
     integer, allocatable :: avail(:), pending(:)
     integer :: numavail, numpending
     integer :: meshrow
  end type slot

  ! LOCAL SCALARS

  integer :: numslivers, numslots, timestep
  integer :: firstavailrot
  integer :: udef
  integer :: ictxt, nprow, npcol, myrow, mycol, iam, nprocs, left, right, proc
  integer :: row1, row2, row3, locrow1, locrow2, locrow3, loccol
  integer :: mb, nb, lldA
  integer :: myleftslot, myrightslot, mylocslot
  integer :: myleftsliver, myrightsliver, mylocsliver, diagsliver
  integer :: i, j, k, kk, l, q
  integer :: tmpsliver
  integer :: c1l, c2l
  integer :: numactiveslots
  logical :: dodiag
  logical :: isavail, ispending
  logical :: hasleftslot, hasrightslot, haslocslot


  logical :: terminate
  double precision ::  tmp2
  double precision, dimension(1) :: tmp

  ! LOCAL ARRAYS

  type(sliver), allocatable :: slivers(:)
  type(slot), allocatable :: slots(:, :)
  double precision, allocatable :: leftbuf(:), rightbuf(:)
  integer, allocatable :: numcross(:), numlocal(:)
  logical, allocatable :: docross(:), dolocal(:)

  ! EXTERNALS

  integer, external :: indxl2g, indxg2l, indxg2p, iceil, numroc

  double precision one, zero
  parameter (one = 1.0d+0, zero=0.0D+0)

  ! EXECUTABLE STATEMENTS
  ! =====================



  udef = 0


  cvec(1:descA(m_)) = 1.0d0
  svec(1:descA(m_)) = 0.0d0

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

  ! Allocate misc arrays.
  allocate(numcross(nprow), numlocal(nprow))
  allocate(docross(nprow), dolocal(nprow))

  ! Allocate temporary buffers.
  allocate(leftbuf(mb), rightbuf(mb))

  ! Initialize firstavailrot.
  firstavailrot = jA + n

  ! Count the number of slivers.
  numslivers = ((iA + m - 1) - 1) / mb - (iA - 1) / mb + 1

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
  allocate(slots(nprow, numslots))
  do i = 1, nprow
     do j = 1, numslots
        allocate(slots(i, j)%avail(numslivers), slots(i, j)%pending(numslivers))
     end do
  end do

  ! Initialize the sliver states.
  do i = 1, numslivers
     if (i == 1) then
        slivers(i)%r1g = iA
     else
        slivers(i)%r1g = slivers(i - 1)%r2g + 1
     end if
     slivers(i)%r2g = min(iA + m - 1, ((slivers(i)%r1g - 1) / mb) * mb + mb)
     slivers(i)%height = slivers(i)%r2g - slivers(i)%r1g + 1
     slivers(i)%c2g = min(jA + n - 1, (((jA + n - 1) - 1) / nb) * nb + nb)
     slivers(i)%c1g = max(jA, ((slivers(i)%c2g - 1) / nb) * nb + 1)
     proc = indxg2p(slivers(i)%c2g, nb, udef, descA(csrc_), npcol)
     slivers(i)%slot = 2 * proc + 1
     if (slivers(i)%c1g == slivers(i)%c2g) then
        slivers(i)%c1g = max(jA, slivers(i)%c2g - 1)
        slivers(i)%slot = 1 + modulo(slivers(i)%slot - 2, numslots)
     end if
     slivers(i)%r1l = indxg2l(slivers(i)%r1g, mb, udef, udef, nprow)
     slivers(i)%r2l = indxg2l(slivers(i)%r2g, mb, udef, udef, nprow)
     slivers(i)%meshrow = 1 + indxg2p(slivers(i)%r1g, mb, udef, descA(rsrc_), nprow)
  end do

  ! Initialize the slots.
  do i = 1, nprow
     do j = 1, numslots
        slots(i, j)%time = 0
        slots(i, j)%avail(1:numslivers) = 0
        slots(i, j)%pending(1:numslivers) = 0
        slots(i, j)%numavail = 0
        slots(i, j)%numpending = 0
        slots(i, j)%meshrow = i
     end do
  end do

  ! Add the slivers to the slots.
  do i = 1, numslivers
     ! Skip if the sliver is already depleted.
     if (slivers(i)%c2g - slivers(i)%c1g + 1 < 2) then
        cycle
     end if

     ! Add the sliver to its slot.
     j = slivers(i)%meshrow
     k = slivers(i)%slot
     if (slivers(i)%c2g == slivers(i)%r2g .and. firstavailrot == slivers(i)%c2g + 1) then
        ! Diagonal and available (only one).
        slots(j, k)%avail(1) = i
        slots(j, k)%numavail = 1
     else
        ! Pending.
        l = slots(j, k)%numpending + 1
        do while (l > 1)
           if (slivers(slots(j, k)%pending(l - 1))%c2g <= slivers(i)%c2g) then
              exit
           end if
           slots(j, k)%pending(l) = slots(j, k)%pending(l - 1)
           l = l - 1
        end do
        slots(j, k)%pending(l) = i
        slots(j, k)%numpending = slots(j, k)%numpending + 1
     end if
  end do

  ! MAIN LOOP

  timestep = 0
  do
     timestep = timestep + 1


     ! COUNT THE NUMBER OF ACTIVATABLE CROSS-BORDER AND LOCAL SLOTS

     do i = 1, nprow
        numlocal(i) = 0
        do j = 1, numslots, 2
           if (slots(i, j)%numavail > 0) then
              numlocal(i) = numlocal(i) + 1
           end if
        end do
        numcross(i) = 0
        do j = 2, numslots, 2
           if (slots(i, j)%numavail > 0) then
              numcross(i) = numcross(i) + 1
           end if
        end do
     end do

     ! CHOOSE SLOT TYPES

     do i = 1, nprow
        if (numcross(i) >= numlocal(i)) then
           docross(i) = .true.
           dolocal(i) = .false.
        else
           docross(i) = .false.
           dolocal(i) = .true.
        end if
     end do

     ! ACTIVATE SLOTS

     numactiveslots = 0
     do i = 1, nprow
        do j = 1, numslots
           ! Skip if empty.
           if (slots(i, j)%numavail == 0) then
              cycle
           end if

           ! Skip if the wrong type.
           if ((docross(i) .and. modulo(j, 2) == 1) .or. (dolocal(i) .and. modulo(j, 2) == 0)) then
              cycle
           end if

           ! Activate this slot.
           slots(i, j)%time = timestep
           numactiveslots = numactiveslots + 1
        end do
     end do

     ! SCHEDULE THE LOCALLY ACTIVE SLOTS

     ! Determine which slots are active locally.
     hasleftslot  = (slots(myrow + 1, myleftslot)%time  == timestep)
     hasrightslot = (slots(myrow + 1, myrightslot)%time == timestep)
     haslocslot   = (slots(myrow + 1, mylocslot)%time   == timestep)
     ! Determine if there is a diagonal block active.
     dodiag = .false.
     L1 : do i = 1, nprow
        do j = 1, numslots
           if (slots(i, j)%time /= timestep) then
              cycle
           end if
           k = slots(i, j)%avail(slots(i, j)%numavail)
           if (slivers(k)%c2g == slivers(k)%r2g) then
              dodiag = .true.
              diagsliver = k
              exit L1
           end if
        end do
     end do L1

     ! Determine which slivers are active locally.
     if (hasleftslot)  myleftsliver  = slots(myrow + 1, myleftslot)%avail (slots(myrow + 1, myleftslot)%numavail)
     if (hasrightslot) myrightsliver = slots(myrow + 1, myrightslot)%avail(slots(myrow + 1, myrightslot)%numavail)
     if (haslocslot)   mylocsliver   = slots(myrow + 1, mylocslot)%avail  (slots(myrow + 1, mylocslot)%numavail)


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
     call dgerv2d(ictxt, slivers(myrightsliver)%height, 1, rightbuf, mb, myrow, right)

     ! Step 4 (right cross-border): Update.
     if ( cvec(slivers(myrightsliver)%c2g) .ne. one .and. svec(slivers(myrightsliver)%c2g) .ne. zero ) then
        call drot(slivers(myrightsliver)%height, rightbuf, 1, A(slivers(myrightsliver)%r1l +(c1l - 1) * lldA), 1, &
             cvec(slivers(myrightsliver)%c2g), svec(slivers(myrightsliver)%c2g))

     end if
     ! Step 5 (left cross-border): Receive.
     500 continue
     if (.not. hasleftslot) goto 810
     call dgerv2d(ictxt, slivers(myleftsliver)%height, 1, leftbuf, mb, myrow, left)

     ! Step 6 (left cross-border): Update.
     if ( cvec(slivers(myleftsliver)%c2g) .ne. one .and. svec(slivers(myleftsliver)%c2g) .ne. zero ) then
        call drot(slivers(myleftsliver)%height, A(slivers(myleftsliver)%r1l + (c2l - 1) * lldA), 1, leftbuf, 1, &
             cvec(slivers(myleftsliver)%c2g), svec(slivers(myleftsliver)%c2g))
     end if

     ! Step 8a (diag): Send and receive subdiagonal element.
     810 continue
     ! Skip if not doing diagonal in this timestep.
     if (.not. dodiag) goto 700
     ! The diagonal block is dense on rows row1 : row2 - 1 and Hessenberg on rows row2 : row3.
     row1 = slivers(diagsliver)%r1g
     row3 = slivers(diagsliver)%r2g
     row2 = row3 - (slivers(diagsliver)%c2g - slivers(diagsliver)%c1g + 1) + 1
     ! Special fix if height of block is only 1
     if (row1 == row3) row2 = row1

     ! The mesh column that owns the diagonal.
     proc = (slivers(diagsliver)%slot - 1) / 2
     ! The above doesnt always work for the diagonal block, using c2g for determining proc should be fool proof.
     proc = indxg2p(slivers(diagsliver)%c2g, nb, udef, descA(csrc_), npcol)

     if (myrow + 1 == slivers(diagsliver)%meshrow .and. modulo(mycol + 1, npcol) == proc) then
        ! Owns the subdiagonal element on the other side of the block boundary (if any).
        if (slivers(diagsliver)%c1g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%c1g > jA) then
           ! There is a subdiagonal element; locate it.
           locrow2 = indxg2l(row2, mb, udef, udef, nprow)
           loccol  = indxg2l(slivers(diagsliver)%c1g - 1, nb, udef, udef, npcol)
           ! Send the element to the right.
           call dgesd2d(ictxt, 1, 1, A(locrow2 + (loccol - 1) * lldA), lldA, myrow, right)
           ! Set it to zero.
           A(locrow2 + (loccol - 1) * lldA) = 0.0d+0
        else if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height ==1 ) then
           ! There is a subdiagonal element; locate it.
           locrow2 = indxg2l(row2, mb, udef, udef, nprow)
           loccol  = indxg2l(slivers(diagsliver)%c1g, nb, udef, udef, npcol)
           ! Send the element to the right.
           call dgesd2d(ictxt, 1, 1, A(locrow2 + (loccol - 1) * lldA), lldA, myrow, right)
           ! Set it to zero.
           A(locrow2 + (loccol - 1) * lldA) = 0.0d+0
        end if
     end if
     if (myrow + 1 == slivers(diagsliver)%meshrow .and. mycol == proc) then
        ! Owns the diagonal block; reduce the Hessenberg part.
        if (slivers(diagsliver)%c1g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%c1g > jA ) then
           ! There is a subdiagonal element; receive it from the left.
           ! Store in the variable tmp, used below at 820, see "Reduce the subdiagonal element"
           call dgerv2d(ictxt, 1, 1, tmp, 1, myrow, left)
        else if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height == 1 ) then
           call dgerv2d(ictxt, 1, 1, tmp, 1, myrow, left)
        end if
     end if

     ! Step 7 (local): Update.
     700 continue
     if (.not. haslocslot) goto 820
     c1l = indxg2l(slivers(mylocsliver)%c1g, nb, udef, udef, npcol)
     c2l = indxg2l(slivers(mylocsliver)%c2g, nb, udef, udef, npcol)
     if (slivers(mylocsliver)%c2g == slivers(mylocsliver)%r2g) goto 820

     call krnlcolumnupdate(slivers(mylocsliver)%height, c2l - c1l + 1, A(slivers(mylocsliver)%r1l + (c1l - 1) * lldA), lldA, &
          cvec(slivers(mylocsliver)%c1g), svec(slivers(mylocsliver)%c1g) )

     ! Step 8b (diag): Reduce diagonal block.
     820 continue
     ! Skip if not doing diagonal in this timestep.
     if (.not. dodiag) goto 830

     c1l = indxg2l(slivers(diagsliver)%c1g, nb, udef, udef, npcol)
     c2l = indxg2l(slivers(diagsliver)%c2g, nb, udef, udef, npcol)
     if (myrow + 1 == slivers(diagsliver)%meshrow .and. mycol == proc) then
        ! Owns the diagonal block; reduce the Hessenberg part.
        locrow1 = indxg2l(row1, mb, udef, udef, nprow)
        locrow2 = indxg2l(row2, mb, udef, udef, nprow)
        locrow3 = indxg2l(row3, mb, udef, udef, nprow)

        call krnlcolumnannihilate(locrow3 - locrow2 + 1, A(locrow2 + (c1l - 1) * lldA), lldA, &
             cvec(slivers(diagsliver)%c1g), svec(slivers(diagsliver)%c1g) )
        ! Update the dense part (if any).
        if (locrow2 - locrow1 > 0) then
           call krnlcolumnupdate(locrow2 - locrow1, c2l - c1l + 1, A(locrow1 + (c1l - 1) * lldA), lldA, &
                cvec(slivers(diagsliver)%c1g), svec(slivers(diagsliver)%c1g) )
        end if
        if (slivers(diagsliver)%c1g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%c1g > jA) then
           ! Reduce the subdiagonal element.
           tmp2 = A(locrow2 + (c1l - 1) * lldA )
           call dlartg(tmp2, tmp, cvec(slivers(diagsliver)%c1g), svec(slivers(diagsliver)%c1g), A(locrow2 + (c1l - 1) * lldA))
        else if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height ==1 ) then
           ! Reduce the subdiagonal element.
           tmp2 = A(locrow2 + (c2l - 1) * lldA )
           call dlartg(tmp2, tmp, cvec(slivers(diagsliver)%c2g), svec(slivers(diagsliver)%c2g), A(locrow2 + (c2l - 1) * lldA))
        end if
     end if

     ! Step 8c (diag): Broadcast new rotations.
     830 continue
     ! Skip if not doing diagonal in this timestep.
     if (.not. dodiag) goto 900
     ! Broadcast new rotations along mesh column.
     if (myrow + 1 == slivers(diagsliver)%meshrow .and. mycol == proc) then
        if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height ==1) then
           call dgebs2d(ictxt, 'C', ' ', 1, 1, cvec(slivers(diagsliver)%c2g), 1)
           call dgebs2d(ictxt, 'C', ' ', 1, 1, svec(slivers(diagsliver)%c2g), 1)
        else
           call dgebs2d(ictxt, 'C', ' ', 1, slivers(diagsliver)%c2g - slivers(diagsliver)%c1g + 1, cvec(slivers(diagsliver)%c1g), 1)
           call dgebs2d(ictxt, 'C', ' ', 1, slivers(diagsliver)%c2g - slivers(diagsliver)%c1g + 1, svec(slivers(diagsliver)%c1g), 1)
        end if
     else if (mycol .eq. proc ) then
        if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height ==1) then
           call dgebr2d(ictxt, 'C', ' ', 1, 1, cvec(slivers(diagsliver)%c2g), 1, slivers(diagsliver)%meshrow - 1, proc)
           call dgebr2d(ictxt, 'C', ' ', 1, 1, svec(slivers(diagsliver)%c2g), 1, slivers(diagsliver)%meshrow - 1, proc)
        else
           call dgebr2d(ictxt, 'C', ' ', 1, slivers(diagsliver)%c2g - slivers(diagsliver)%c1g + 1, cvec(slivers(diagsliver)%c1g), &
                1, slivers(diagsliver)%meshrow - 1, proc)
           call dgebr2d(ictxt, 'C', ' ', 1, slivers(diagsliver)%c2g - slivers(diagsliver)%c1g + 1, svec(slivers(diagsliver)%c1g), &
                1, slivers(diagsliver)%meshrow - 1, proc)
        end if
     end if
     ! Send/receive left cross-border rotation over the boundary.
     if (mycol == proc) then
        if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height ==1) then
           call dgesd2d(ictxt, 1 , 1, cvec(slivers(diagsliver)%c2g), 1, myrow, left)
           call dgesd2d(ictxt, 1 , 1, svec(slivers(diagsliver)%c2g), 1, myrow, left)
        else if ( slivers(diagsliver)%c1g > jA ) then
           call dgesd2d(ictxt, 1 , 1, cvec(slivers(diagsliver)%c1g), 1, myrow, left)
           call dgesd2d(ictxt, 1 , 1, svec(slivers(diagsliver)%c1g), 1, myrow, left)
        end if
     end if
     if (modulo(mycol + 1, npcol) == proc) then
        if (slivers(diagsliver)%c2g == slivers(diagsliver)%r1g .and. slivers(diagsliver)%height ==1) then
           call dgerv2d(ictxt, 1 , 1, cvec(slivers(diagsliver)%c2g), 1, myrow, right)
           call dgerv2d(ictxt, 1 , 1, svec(slivers(diagsliver)%c2g), 1, myrow, right)
        else if ( slivers(diagsliver)%c1g > jA ) then
           call dgerv2d(ictxt, 1 , 1, cvec(slivers(diagsliver)%c1g), 1, myrow, right)
           call dgerv2d(ictxt, 1 , 1, svec(slivers(diagsliver)%c1g), 1, myrow, right)
        end if
     end if
     ! Update firstavailrot.
     firstavailrot = max(jA + 1, slivers(diagsliver)%c1g)

     ! Done.
     900 continue

     ! UPDATE THE SLIVERS AND SLOTS

     do i = 1, nprow
        do j = 1, numslots
           ! Skip if the sliver is not active.
           if (slots(i, j)%time /= timestep) then
              cycle
           end if

           ! Locate the active sliver.
           k = slots(i, j)%avail(slots(i, j)%numavail)

           ! Remove the sliver for a moment (possibly permanently).
           slots(i, j)%numavail = slots(i, j)%numavail - 1

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
           ! Account for Hessenberg structure.
           slivers(k)%c1g = max(slivers(k)%c1g, slivers(k)%r1g)

           ! Skip if the sliver has now become depleted.
           if (slivers(k)%c2g - slivers(k)%c1g + 1 < 2) then
              cycle
           end if

           ! Determine if the sliver is still available or now pending.
           if (slivers(k)%c2g == slivers(k)%r2g .or. slivers(k)%c1g + 1 >= firstavailrot) then
              ! Still available.
              isavail = .true.
              ispending = .false.
           else
              ! Now pending.
              isavail = .false.
              ispending = .true.
           end if

           ! Move the sliver to its new slot.
           l = slivers(k)%slot
           if (isavail .and. l > j .and. slots(i, l)%time == timestep) then
              ! Temporarily remove the top element on the new slot.
              tmpsliver = slots(i, l)%avail(slots(i, l)%numavail)
              slots(i, l)%numavail = slots(i, l)%numavail - 1
           end if
           if (isavail) then
              q = slots(i, l)%numavail + 1
           else if (ispending) then
              q = slots(i, l)%numpending + 1
           end if
           do while (q > 1)
              if (isavail) then
                 if (slivers(slots(i, l)%avail(q - 1))%c2g < slivers(k)%c2g .or. &
                      (slivers(slots(i, l)%avail(q - 1))%c2g == slivers(k)%c2g .and. &
                      slivers(slots(i, l)%avail(q - 1))%r1g < slivers(k)%r1g)) then
                    exit
                 end if
                 slots(i, l)%avail(q) = slots(i, l)%avail(q - 1)
              else if (ispending) then
                 if (slivers(slots(i, l)%pending(q - 1))%c2g < slivers(k)%c2g .or. &
                      (slivers(slots(i, l)%pending(q - 1))%c2g == slivers(k)%c2g .and. &
                      slivers(slots(i, l)%pending(q - 1))%r1g < slivers(k)%r1g)) then
                    exit
                 end if
                 slots(i, l)%pending(q) = slots(i, l)%pending(q - 1)
              end if
              q = q - 1
           end do
           if (isavail) then
              slots(i, l)%avail(q) = k
              slots(i, l)%numavail = slots(i, l)%numavail + 1
           else if (ispending) then
              slots(i, l)%pending(q) = k
              slots(i, l)%numpending = slots(i, l)%numpending + 1
           end if
           if (isavail .and. l > j .and. slots(i, l)%time == timestep) then
              ! Replace the old top element.
              slots(i, l)%numavail = slots(i, l)%numavail + 1
              slots(i, l)%avail(slots(i, l)%numavail) = tmpsliver
           end if
        end do
     end do

     ! TRANSFER NEWLY AVAILABLE SLIVERS FROM PENDING TO AVAIL

     do i = 1, nprow
        do j = 1, numslots
           k = slots(i, j)%numpending
           do while (k >= 1)
              l = slots(i, j)%pending(k)
              isavail = .false.
              if (slivers(l)%c2g == slivers(l)%r2g) then
                 ! Newly available diagonal.
                 isavail = .true.
              else if (slivers(l)%c1g + 1 >= firstavailrot) then
                 ! Newly available off-diagonal.
                 isavail = .true.
              end if
              if (isavail) then
                 ! Move from pending to avail.
                 slots(i, j)%numpending = slots(i, j)%numpending - 1
                 do kk = k, slots(i, j)%numpending
                    slots(i, j)%pending(kk) = slots(i, j)%pending(kk + 1)
                 end do
                 kk = slots(i, j)%numavail + 1
                 do while (kk > 1)
                    if (slivers(slots(i, j)%avail(kk - 1))%c2g < slivers(l)%c2g .or. &
                         (slivers(slots(i, j)%avail(kk - 1))%c2g == slivers(l)%c2g .and. &
                         slivers(slots(i, j)%avail(kk - 1))%r1g < slivers(l)%r1g)) then
                       exit
                    end if
                    slots(i, j)%avail(kk) = slots(i, j)%avail(kk - 1)
                    kk = kk - 1
                 end do
                 slots(i, j)%avail(kk) = l
                 slots(i, j)%numavail = slots(i, j)%numavail + 1
              end if
              k = k - 1
           end do
        end do
     end do

     ! DETECT TERMINATION

     terminate = .true.
     do i = 1, nprow
        do j = 1, numslots
           if (slots(i, j)%numavail > 0) then
              terminate = .false.
              exit
           end if
        end do
     end do
     if (terminate) then
        exit
     end if
  end do


  ! Deallocate misc arrays.
  deallocate(numcross, numlocal)
  deallocate(docross, dolocal)

  ! Deallocate the slots.
  do i = 1, nprow
     do j = 1, numslots
        deallocate(slots(i, j)%avail, slots(i, j)%pending)
     end do
  end do
  deallocate(slots)

  ! Deallocate the sliver states.
  deallocate(slivers)

  ! Deallocate temporary buffers.
  deallocate(leftbuf, rightbuf)


end subroutine sliverhesscolumnupdate
