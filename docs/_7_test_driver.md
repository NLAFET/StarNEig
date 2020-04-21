# Test program

The test program provides an unified interface for testing the entire library.
Most command line arguments have default values so only few of them have to be
set in most situations. General usage information can be printed as follows:

```
$ ./starneig-test
Usage: ./starneig-test (options)

Global options:
  --mpi -- Enable MPI
  --mpi-mode [serialized,multiple] -- MPI mode
  --seed (num) -- Random number generator seed
  --experiment (experiment) -- Experiment module
  --test-workers [(num),default] -- Test program StarPU worker count
  --blas-threads [(num),default] -- Test program BLAS thread count
  --lapack-threads [(num),default] -- LAPACK solver thread count
  --scalapack-threads [(num),default] -- ScaLAPACK solver thread count

Available experiment modules:
    'hessenberg' : Hessenberg reduction experiment
    'schur' : Schur reduction experiment
    'reorder' : Eigenvalue reordering experiment
    'eigenvectors' : Eigenvectors experiment
    'full-chain' : Full chain experiment
    'partial-hessenberg' : Partial Hessenberg reduction experiment
    'validator' : Validation experiment
```

The `--mpi` option enables the MPI support and `--seed (num)` option initializes
the random number generator with a given seed. Available experiment modules are
listed below the global options and the desired experiment module is selected
with the `--experiment (experiment)` option. For example, the Hessenberg
reduction specific experiment module usage information can be printed as
follows:

```
$ ./starneig-test --experiment hessenberg
Usage: ./starneig-test (options)

Global options:
  --mpi -- Enable MPI
  --mpi-mode [serialized,multiple] -- MPI mode
  --seed (num) -- Random number generator seed
  --experiment (experiment) -- Experiment module
  --test-workers [(num),default] -- Test program StarPU worker count
  --blas-threads [(num),default] -- Test program BLAS thread count
  --lapack-threads [(num),default] -- LAPACK solver thread count
  --scalapack-threads [(num),default] -- ScaLAPACK solver thread count

Available experiment modules:
    'hessenberg' : Hessenberg reduction experiment
    'schur' : Schur reduction experiment
    'reorder' : Eigenvalue reordering experiment
    'eigenvectors' : Eigenvectors experiment
    'full-chain' : Full chain experiment
    'partial-hessenberg' : Partial Hessenberg reduction experiment
    'validator' : Validation experiment

Experiment module (hessenberg) specific options:
  --data-format (format) -- Data format
  --init (initializer) -- Initialization module
  --solver (solver) -- Solver module
  --hooks (hook:mode, ...) -- Hooks and modes
  --no-reinit -- Do not reinitialize after each repetition
  --repeat (num) -- Repeated experiment
  --warmup (num) -- Perform "warmups"
  --keep-going -- Try to recover from a solver failure
  --abort -- Call abort() in failure
...
```

The overall design of the test program is modular. Each experiment module is
built on *initializers*, *solvers* and *hooks*. Each experiment module contains
several of each, thus allowing a user to initialize the inout data in various
ways and compare different solvers with each other. Each of these building
blocks can be configured with various command line arguments. However, in most
situations only the problem dimension `--n (num)` needs to be specified.

Hooks are used to test and validate the output of the solver. For example,
`--hooks hessenberg residual print` option enables hooks that

 - check whether the output matrix is in upper Hessenberg form;
 - computes the related residuals using Frobenius norm (reported in terms of
   unit roundoff error) and checks that they are within the permissible limits;
   and
 - prints the output matrices.

Certain general purpose initializers allow a user to read the input data from
a disk (`read-mtx` and `read-raw`) and output data can be stored to a disk
using a suitable post-processing hook (`store-raw`).

The test program supports various data formats. For example, shared memory
experiments are usually performed using the `pencil-local` data format which
stores the matrices continuously in the main memory. Distributed memory
experiments are performed using either a StarNEig library specific distributed
data format (`pencil-starneig`) and the BLACS data format
(`pencil-starneig-blacs`). The test program will detect the correct data format
automatically. The test program is also is capable of converting the data
between various data formats. The related data converter modules can be in most
cases configured using additional command line arguments. For example, the
distributed data formats distribute the data in rectangular sections and the
section size can be set with command line arguments `--section-height (num)` and
`--section-width (num)`).

## Performance models

The StarPU performance models must be calibrated before the software can
function efficiently on heterogeneous platforms (CPUs+GPUs). The calibration
is triggered automatically if the models are not calibrated well enough for a
given problem size. This can impact the execution time negatively. The test
program provides an easy-to-use solution for this problem:
```
$ ./starneig-test ... --warmup 3
```
The `--warmup (number)` argument causes the test program to perform a number of
warm-up runs before the actual execution time measurement takes place.

Please see the StarPU handbook for further instructions:
http://starpu.gforge.inria.fr/doc/html/Scheduling.html

## Examples

 - Reorder a 4000 x 4000 matrix using the StarNEig implementation:

```
$ ./starneig-test --experiment reorder --n 4000
TEST: --seed 1585762840 --experiment reorder --test-workers default --blas-threads default --lapack-threads default --scalapack-threads default --data-format pencil-local --init default --n 4000 --complex-distr uniform --complex-ratio 0.500000 --zero-ratio 0.010000 --inf-ratio 0.010000 --data-distr default --section-height default --section-width default --select-ratio 0.350000 --solver starneig --cores default --gpus default --tile-size default --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal eigenvalues:normal analysis:normal reordering:normal residual:normal --eigenvalues-fail-threshold 10000 --eigenvalues-warn-threshold 1000 --reordering-fail-threshold 10000 --reordering-warn-threshold 1000 --residual-fail-threshold 10000 --residual-warn-threshold 500 --repeat 1 --warmup 0
THREADS: Using 6 StarPU worker threads during initialization and validation.
THREADS: Using 6 BLAS threads during initialization and validation.
THREADS: Using 6 BLAS threads in LAPACK solvers.
THREADS: Using 1 BLAS threads in ScaLAPACK solvers.
INIT...
PREPARE...
PROCESS...
[starneig][message] Setting tile size to 192.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 1701 MS
FINALIZE...
EIGENVALUES CHECK: mean = 0 u, min = 0 u, max = 0 u
EIGENVALUES ANALYSIS: zeros = 36, infinities = 0, indefinites = 0
EIGENVALUES ANALYSIS: close zeros = 0, close infinities = 0, close indefinites = 0
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 71 u, min = 0 u, max = 526 u
|Q ~A Q^T - A| / |A| = 314 u
|Q Q^T - I| / |I| = 140 u
================================================================
TIME = 1701 MS [avg 1701 MS, cv 0.00, min 1701 MS, max 1701 MS]
NO FAILED SCHUR FORM TESTS
EIGENVALUES CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (MEANS): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MAX): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES ANALYSIS (ZEROS): [avg 36.0, cv 0.00, min 36, max 36]
EIGENVALUES ANALYSIS (CLOSE ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 71 u, cv 0.00, min 71 u, max 71 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 526 u, cv 0.00, min 526 u, max 526 u]
|Q ~A Q^T - A| / |A| = [avg 314 u, cv 0.00, min 314 u, max 314 u]
|Q Q^T - I| / |I| = [avg 140 u, cv 0.00, min 140 u, max 140 u]
```

 - Reorder a 4000 x 4000 matrix stencil \f$(A,B)\f$ using the StarNEig
   implementation, initialize the random number generator using the seed
   `1480591971`, fortify the matrix stencil \f$(A,B)\f$ against failed swaps,
   and disable GPUs:

```
$ ./starneig-test --experiment reorder --n 4000 --generalized --seed 1480591971 --fortify --gpus 0
TEST: --seed 1480591971 --experiment reorder --test-workers default --blas-threads default --lapack-threads default --scalapack-threads default --data-format pencil-local --init default --n 4000 --generalized --complex-distr uniform --fortify --data-distr default --section-height default --section-width default --select-ratio 0.350000 --solver starneig --cores default --gpus 0 --tile-size default --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal eigenvalues:normal analysis:normal reordering:normal residual:normal --eigenvalues-fail-threshold 10000 --eigenvalues-warn-threshold 1000 --reordering-fail-threshold 10000 --reordering-warn-threshold 1000 --residual-fail-threshold 10000 --residual-warn-threshold 500 --repeat 1 --warmup 0
THREADS: Using 6 StarPU worker threads during initialization and validation.
THREADS: Using 6 BLAS threads during initialization and validation.
THREADS: Using 6 BLAS threads in LAPACK solvers.
THREADS: Using 1 BLAS threads in ScaLAPACK solvers.
INIT...
PREPARE...
PROCESS...
[starneig][message] Setting tile size to 192.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 7472 MS
FINALIZE...
EIGENVALUES CHECK: mean = 0 u, min = 0 u, max = 0 u
EIGENVALUES ANALYSIS: zeros = 0, infinities = 0, indefinites = 0
EIGENVALUES ANALYSIS: close zeros = 0, close infinities = 0, close indefinites = 0
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 23 u, min = 0 u, max = 169 u
|Q ~A Z^T - A| / |A| = 46 u
|Q ~B Z^T - B| / |B| = 63 u
|Q Q^T - I| / |I| = 44 u
|Z Z^T - I| / |I| = 43 u
================================================================
TIME = 7472 MS [avg 7472 MS, cv 0.00, min 7472 MS, max 7472 MS]
NO FAILED SCHUR FORM TESTS
EIGENVALUES CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (MEANS): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MAX): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES ANALYSIS (ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 23 u, cv 0.00, min 23 u, max 23 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 169 u, cv 0.00, min 169 u, max 169 u]
|Q ~A Z^T - A| / |A| = [avg 46 u, cv 0.00, min 46 u, max 46 u]
|Q ~B Z^T - B| / |B| = [avg 63 u, cv 0.00, min 63 u, max 63 u]
|Q Q^T - I| / |I| = [avg 44 u, cv 0.00, min 44 u, max 44 u]
|Z Z^T - I| / |I| = [avg 43 u, cv 0.00, min 43 u, max 43 u]
```
 - Reduce a dense matrix to upper Hessenberg form, validate the output and store
   the output matrices to the disk:

```
$ ./starneig-test --experiment hessenberg --n 4000 --hooks hessenberg residual store-raw --store-raw-output hessenberg_%s.dat
TEST: --seed 1585762935 --experiment hessenberg --test-workers default --blas-threads default --lapack-threads default --scalapack-threads default --data-format pencil-local --init default --n 4000 --data-distr default --section-height default --section-width default --solver starneig --cores default --gpus default --tile-size default --panel-width default --hooks hessenberg:normal residual:normal store-raw:normal --residual-fail-threshold 10000 --residual-warn-threshold 500 --store-raw-output hessenberg_%s.dat --repeat 1 --warmup 0
THREADS: Using 6 StarPU worker threads during initialization and validation.
THREADS: Using 6 BLAS threads during initialization and validation.
THREADS: Using 6 BLAS threads in LAPACK solvers.
THREADS: Using 1 BLAS threads in ScaLAPACK solvers.
INIT...
PREPARE...
PROCESS...
[starneig][message] Setting tile size to 336.
[starneig][message] Setting panel width to 288.
EXPERIMENT TIME = 13121 MS
FINALIZE...
|Q ~A Q^T - A| / |A| = 15 u
|Q Q^T - I| / |I| = 11 u
WRITING TO hessenberg_A.dat...
WRITING TO hessenberg_Q.dat...
WRITING TO hessenberg_CA.dat...
================================================================
TIME = 13121 MS [avg 13121 MS, cv 0.00, min 13121 MS, max 13121 MS]
NO FAILED HESSENBERG FORM TESTS
|Q ~A Q^T - A| / |A| = [avg 15 u, cv 0.00, min 15 u, max 15 u]
|Q Q^T - I| / |I| = [avg 11 u, cv 0.00, min 11 u, max 11 u]
```

 - Read an upper Hessenberg matrix from the disk, reduce it to Schur form and
   set tile size to `128`:

```
$ ./starneig-test --experiment schur --init read-raw --input hessenberg_%s.dat --tile-size 128
TEST: --seed 1585762972 --experiment schur --test-workers default --blas-threads default --lapack-threads default --scalapack-threads default --data-format pencil-local --init read-raw --input hessenberg_%s.dat --data-distr default --section-height default --section-width default --solver starneig --cores default --gpus default --iteration-limit default --tile-size 128 --small-limit default --aed-window-size default --aed-shift-count default --aed-nibble default --aed-parallel-soft-limit default --aed-parallel-hard-limit default --window-size default --shifts-per-window default --update-width default --update-height default --left-threshold default --right-threshold default --inf-threshold default --hooks schur:normal eigenvalues:normal known-eigenvalues:normal analysis:normal residual:normal --eigenvalues-fail-threshold 10000 --eigenvalues-warn-threshold 1000 --known-eigenvalues-fail-threshold 1000000 --known-eigenvalues-warn-threshold 10000 --residual-fail-threshold 10000 --residual-warn-threshold 500 --repeat 1 --warmup 0
THREADS: Using 6 StarPU worker threads during initialization and validation.
THREADS: Using 6 BLAS threads during initialization and validation.
THREADS: Using 6 BLAS threads in LAPACK solvers.
THREADS: Using 1 BLAS threads in ScaLAPACK solvers.
INIT...
READING FROM hessenberg_A.dat...
READING A 4000 X 4000 MATRIX ...
READING FROM hessenberg_Q.dat...
READING A 4000 X 4000 MATRIX ...
READING FROM hessenberg_CA.dat...
READING A 4000 X 4000 MATRIX ...
PREPARE...
PROCESS...
[starneig][message] Using AED windows size 320.
[starneig][message] Using 240 shifts.
EXPERIMENT TIME = 9479 MS
FINALIZE...
EIGENVALUES CHECK: mean = 0 u, min = 0 u, max = 0 u
KNOWN EIGENVALUES CHECK: The stored pencil does not contain the known eigenvalues. Skipping.
EIGENVALUES ANALYSIS: zeros = 0, infinities = 0, indefinites = 0
EIGENVALUES ANALYSIS: close zeros = 0, close infinities = 0, close indefinites = 0
|Q ~A Q^T - A| / |A| = 68 u
|Q Q^T - I| / |I| = 89 u
================================================================
TIME = 9479 MS [avg 9479 MS, cv 0.00, min 9479 MS, max 9479 MS]
NO FAILED SCHUR FORM TESTS
EIGENVALUES CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (MEANS): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MAX): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES ANALYSIS (ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
|Q ~A Q^T - A| / |A| = [avg 68 u, cv 0.00, min 68 u, max 68 u]
|Q Q^T - I| / |I| = [avg 89 u, cv 0.00, min 89 u, max 89 u]
```

## Distributed memory example

 - Reorder a 4000 x 4000 matrix using the StarNEig implementation, use two MPI
   ranks, use three CPU cores per rank, distribute the matrix in 1024 x 1024
   sections, and use tile size 256:

```
$ mpirun -n 2 --map-by :PE=3 ./starneig-test --mpi --experiment reorder --n 4000 --section-height 1024 --section-width 1024 --tile-size 256
```

Rank 0 output:

```
MPI INIT...
TEST: --mpi --mpi-mode serialized --seed 1585763077 --experiment reorder --test-workers default --blas-threads default --lapack-threads default --scalapack-threads default --data-format pencil-starneig-blacs --init default --n 4000 --complex-distr uniform --complex-ratio 0.500000 --zero-ratio 0.010000 --inf-ratio 0.010000 --data-distr default --section-height 1024 --section-width 1024 --select-ratio 0.350000 --solver starneig --cores default --gpus default --tile-size 256 --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal eigenvalues:normal analysis:normal reordering:normal residual:normal --eigenvalues-fail-threshold 10000 --eigenvalues-warn-threshold 1000 --reordering-fail-threshold 10000 --reordering-warn-threshold 1000 --residual-fail-threshold 10000 --residual-warn-threshold 500 --repeat 1 --warmup 0
THREADS: Using 3 StarPU worker threads during initialization and validation.
THREADS: Using 3 BLAS threads during initialization and validation.
THREADS: Using 3 BLAS threads in LAPACK solvers.
THREADS: Using 1 BLAS threads in ScaLAPACK solvers.
INIT...
PREPARE...
PROCESS...
[starneig][message] Attempting to set tile size to 256.
[starneig][message] Setting tile size to 256.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 3320 MS
FINALIZE...
EIGENVALUES CHECK: mean = 0 u, min = 0 u, max = 0 u
EIGENVALUES ANALYSIS: zeros = 37, infinities = 0, indefinites = 0
EIGENVALUES ANALYSIS: close zeros = 0, close infinities = 0, close indefinites = 0
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 70 u, min = 0 u, max = 527 u
|Q ~A Q^T - A| / |A| = 314 u
|Q Q^T - I| / |I| = 139 u
================================================================
TIME = 3319 MS [avg 3320 MS, cv 0.00, min 3320 MS, max 3320 MS]
NO FAILED SCHUR FORM TESTS
EIGENVALUES CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (MEANS): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MAX): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES ANALYSIS (ZEROS): [avg 37.0, cv 0.00, min 37, max 37]
EIGENVALUES ANALYSIS (CLOSE ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 70 u, cv 0.00, min 70 u, max 70 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 527 u, cv 0.00, min 527 u, max 527 u]
|Q ~A Q^T - A| / |A| = [avg 314 u, cv 0.00, min 314 u, max 314 u]
|Q Q^T - I| / |I| = [avg 139 u, cv 0.00, min 139 u, max 139 u]
```

Rank 1 output:

```
MPI INIT...
TEST: --mpi --mpi-mode serialized --seed 1585763077 --experiment reorder --test-workers default --blas-threads default --lapack-threads default --scalapack-threads default --data-format pencil-starneig-blacs --init default --n 4000 --complex-distr uniform --complex-ratio 0.500000 --zero-ratio 0.010000 --inf-ratio 0.010000 --data-distr default --section-height 1024 --section-width 1024 --select-ratio 0.350000 --solver starneig --cores default --gpus default --tile-size 256 --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal eigenvalues:normal analysis:normal reordering:normal residual:normal --eigenvalues-fail-threshold 10000 --eigenvalues-warn-threshold 1000 --reordering-fail-threshold 10000 --reordering-warn-threshold 1000 --residual-fail-threshold 10000 --residual-warn-threshold 500 --repeat 1 --warmup 0
THREADS: Using 3 StarPU worker threads during initialization and validation.
THREADS: Using 3 BLAS threads during initialization and validation.
THREADS: Using 3 BLAS threads in LAPACK solvers.
THREADS: Using 1 BLAS threads in ScaLAPACK solvers.
INIT...
PREPARE...
PROCESS...
[starneig][message] Attempting to set tile size to 256.
[starneig][message] Setting tile size to 256.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 3320 MS
FINALIZE...
EIGENVALUES CHECK: mean = 0 u, min = 0 u, max = 0 u
EIGENVALUES ANALYSIS: zeros = 37, infinities = 0, indefinites = 0
EIGENVALUES ANALYSIS: close zeros = 0, close infinities = 0, close indefinites = 0
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 70 u, min = 0 u, max = 527 u
|Q ~A Q^T - A| / |A| = 314 u
|Q Q^T - I| / |I| = 139 u
================================================================
TIME = 3319 MS [avg 3320 MS, cv 0.00, min 3320 MS, max 3320 MS]
NO FAILED SCHUR FORM TESTS
EIGENVALUES CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES CHECK (MEANS): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES CHECK (MAX): [avg 0 u, cv 0.00, min 0 u, max 0 u]
EIGENVALUES ANALYSIS (ZEROS): [avg 37.0, cv 0.00, min 37, max 37]
EIGENVALUES ANALYSIS (CLOSE ZEROS): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INFINITIES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
EIGENVALUES ANALYSIS (CLOSE INDEFINITES): [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 70 u, cv 0.00, min 70 u, max 70 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 527 u, cv 0.00, min 527 u, max 527 u]
|Q ~A Q^T - A| / |A| = [avg 314 u, cv 0.00, min 314 u, max 314 u]
|Q Q^T - I| / |I| = [avg 139 u, cv 0.00, min 139 u, max 139 u]
```
