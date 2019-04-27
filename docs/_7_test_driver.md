# Test program

The test program provides an unified interface to test all software components.
Most command line arguments have default values and in most cases it is not
necessary to set more than a few. General usage information can be printed as
follows:

```
$ ./starneig-test
Usage: ./starneig-test (options)

Global options:
  --mpi -- Enable MPI
  --mpi-mode [serialized,multiple] -- MPI mode
  --seed (num) -- Random number generator seed
  --blas-threads [(num),default] -- BLAS thread count
  --omp-threads [(num),default] -- OpenMP thread count
  --experiment (experiment) -- Experiment module

Available experiment modules:
    'hessenberg' : Hessenberg reduction experiment
    'schur' : Schur reduction experiment
    'reorder' : Eigenvalue reordering experiment
    'eigenvectors' : Eigenvectors experiment
    'full-chain' : Full chain experiment
    'partial-hessenberg' : Partial Hessenberg reduction experiment
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
  --blas-threads [(num),default] -- BLAS thread count
  --omp-threads [(num),default] -- OpenMP thread count
  --experiment (experiment) -- Experiment module

Available experiment modules:
    'hessenberg' : Hessenberg reduction experiment
    'schur' : Schur reduction experiment
    'reorder' : Eigenvalue reordering experiment
    'eigenvectors' : Eigenvectors experiment
    'full-chain' : Full chain experiment
    'partial-hessenberg' : Partial Hessenberg reduction experiment

Experiment module (hessenberg) specific options:
  --data-format (format) -- Data format
  --init (initializer) -- Initialization module
  --solver (solver) -- Solver module
  --hooks (hook:mode, ...) -- Hooks and modes
...
```

The overall design of the test program is modular. Each experiment module is
build on *initializers*, *solvers* and *hooks*. Each experiment module contains
several of each allowing a user to initialize the data in various ways and
compare different solvers with each other. Each of these building blocks can be
configured with various parameters. However, in most cases only the problem
dimension `--n (num)` needs to be specified.

Hooks are used to test and validate the output of the software components. For
example, `--hooks hessenberg residual print` option enables hooks that

 - check whether the output matrix is in upper Hessenberg form,
 - computes the related residuals using Frobenius norm (reported in terms of
   unit roundoff error) and checks that they are within the permissible limits
   and
 - prints the input and output matrices.

Certain general purpose initializers allow a user to read the input data from
the disk (`read-mtx` and `read-raw`) and output data can be stored to the disk
with a suitable hook (`store-raw`).

The test program supports various data formats. For example, shared memory
experiments are usually performed using the `pencil-local` data format that
stores the matrices continuously in the local main memory. Distributed memory
experiments are performed using either StarNEig library specific distributed
data format (`pencil-starneig`) and BLACS data format (`pencil-starneig-blacs`).
The test program will detect the correct data format automatically. The test
program is also is capable of converting between various data formats. The
related data converter modules can be in most cases configured using additional
command line arguments. For example, the distributed data formats distribute the
data in sections and the section size can be set with command line arguments
(`--section-height (num)` and `--section-width (num)`).

## Performance models

The StarPU performance models must be calibrated before the software can
function efficiently on heterogeneous platforms (CPUs+GPUs). The calibration
is triggered automatically if the models are not calibrated well enough for a
given problem size. This may impact the execution time negatively. The test
program provides an easy to use solution for this problem:
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
TEST: --seed 1549961762 --experiment reorder --omp-threads default --blas-threads default --data-format pencil-local --init default --n 4000 --complex-distr uniform --select-distr uniform --zero-ratio 0.010000 --complex-ratio 0.500000 --select-ratio 0.350000 --solver starneig --cores default --gpus default --tile-size default --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal reordering:normal residual:normal --reordering-fail-threshold 10000 --reordering-warn-threshold 100000 --residual-fail-threshold 1000 --residual-warn-threshold 500 --repeat 1 --warmup 0
INIT...
PREPARE...
PROCESS...
[starneig][message] Setting tile size to 192.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 1407 MS
FINALIZE...
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 21 u, min = 0 u, max = 1954 u
|Q ~A Q^T - A| / |A| = 47 u
|Q Q^T - I| / |I| = 33 u
================================================================
TIME = 1407 MS [avg 1407 MS, cv 0.00, min 1407 MS, max 1407 MS]
NO FAILED SCHUR FORM TESTS
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 21 u, cv 0.00, min 21 u, max 21 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 1954 u, cv 0.00, min 1954 u, max 1954 u]
|Q ~A Q^T - A| / |A| = [avg 47 u, cv 0.00, min 47 u, max 47 u]
|Q Q^T - I| / |I| = [avg 33 u, cv 0.00, min 33 u, max 33 u]
```

 - Reorder a 4000 x 4000 matrix stencil \f$(A,B)\f$ using the StarNEig
   implementation, initialize the random number generator using the seed
   `1480591971`, fortify the matrix stencil \f$(A,B)\f$ against failed swaps,
   and disable GPUs:

```
$ ./starneig-test --experiment reorder --n 4000 --generalized --seed 1480591971 --fortify --gpus 0
TEST: --seed 1480591971 --experiment reorder --omp-threads default --blas-threads default --data-format pencil-local --init default --n 4000 --generalized --complex-distr uniform --select-distr uniform --zero-ratio 0.010000 --inf-ratio 0.010000 --fortify --complex-ratio 0.500000 --select-ratio 0.350000 --solver starneig --cores default --gpus 0 --tile-size default --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal reordering:normal residual:normal --reordering-fail-threshold 10000 --reordering-warn-threshold 100000 --residual-fail-threshold 1000 --residual-warn-threshold 500 --repeat 1 --warmup 0
INIT...
PREPARE...
PROCESS...
[starneig][message] Setting tile size to 192.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 4641 MS
FINALIZE...
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 52 u, min = 0 u, max = 912 u
|Q ~A Z^T - A| / |A| = 46 u
|Q ~B Z^T - B| / |B| = 59 u
|Q Q^T - I| / |I| = 41 u
|Z Z^T - I| / |I| = 43 u
================================================================
TIME = 4641 MS [avg 4641 MS, cv 0.00, min 4641 MS, max 4641 MS]
NO FAILED SCHUR FORM TESTS
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 52 u, cv 0.00, min 52 u, max 52 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 912 u, cv 0.00, min 912 u, max 912 u]
|Q ~A Z^T - A| / |A| = [avg 46 u, cv 0.00, min 46 u, max 46 u]
|Q ~B Z^T - B| / |B| = [avg 59 u, cv 0.00, min 59 u, max 59 u]
|Q Q^T - I| / |I| = [avg 41 u, cv 0.00, min 41 u, max 41 u]
|Z Z^T - I| / |I| = [avg 43 u, cv 0.00, min 43 u, max 43 u]
```
 - Reduce a dense matrix to upper Hessenberg form, validate the output and store
   the output matrices to the disk:

```
$ ./starneig-test --experiment hessenberg --n 4000 --hooks hessenberg residual store-raw --store-raw-output hessenberg_%s.dat
TEST: --seed 1549962246 --experiment hessenberg --omp-threads default --blas-threads default --data-format pencil-local --init default --n 4000 --solver starneig --cores default --gpus default --tile-size default --panel-width default --parallel-worker-size default --hooks hessenberg:normal residual:normal store-raw:normal --residual-fail-threshold 1000 --residual-warn-threshold 500 --store-raw-output hessenberg_%s.dat --repeat 1 --warmup 0
INIT...
PREPARE...
PROCESS...
[starneig][message] Setting tile size to 320.
[starneig][message] Setting panel width to 160.
EXPERIMENT TIME = 7283 MS
FINALIZE...
|Q ~A Q^T - A| / |A| = 12 u
|Q Q^T - I| / |I| = 8 u
WRITING TO hessenberg_A.dat...
WRITING TO hessenberg_Q.dat...
================================================================
TIME = 7283 MS [avg 7283 MS, cv 0.00, min 7283 MS, max 7283 MS]
NO FAILED HESSENBERG FORM TESTS
|Q ~A Q^T - A| / |A| = [avg 12 u, cv 0.00, min 12 u, max 12 u]
|Q Q^T - I| / |I| = [avg 8 u, cv 0.00, min 8 u, max 8 u]
```

 - Read an upper Hessenberg matrix from the disk, reduce it to Schur form and
   set tile size to `128`:

```
$ ./starneig-test --experiment schur --init read-raw --input hessenberg_%s.dat --tile-size 128
TEST: --seed 1549962455 --experiment schur --omp-threads default --blas-threads default --data-format pencil-local --init read-raw --input hessenberg_%s.dat --solver starneig --cores default --gpus default --iteration-limit default --tile-size 128 --small-limit default --aed-parallel-soft-limit default --aed-parallel-hard-limit default --aed-window-size default --aed-shift-count default --window-size default --shifts-per-window default --update-width default --update-height default --hooks schur:normal residual:normal --residual-fail-threshold 1000 --residual-warn-threshold 500 --repeat 1 --warmup 0
INIT...
READING FROM hessenberg_A.dat...
READING 4000 X 4000 MATRIX ...
READING FROM hessenberg_Q.dat...
READING 4000 X 4000 MATRIX ...
PREPARE...
PROCESS...
[starneig][message] Using AED windows size 320.
[starneig][message] Using 240 shifts.
EXPERIMENT TIME = 9742 MS
FINALIZE...
|Q ~A Q^T - A| / |A| = 138 u
|Q Q^T - I| / |I| = 95 u
================================================================
TIME = 9742 MS [avg 9742 MS, cv 0.00, min 9742 MS, max 9742 MS]
NO FAILED SCHUR FORM TESTS
|Q ~A Q^T - A| / |A| = [avg 138 u, cv 0.00, min 138 u, max 138 u]
|Q Q^T - I| / |I| = [avg 95 u, cv 0.00, min 95 u, max 95 u]
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
THREADS: Found 3 PUs.
TEST: --mpi --mpi-mode serialized --seed 1551464108 --experiment reorder --omp-threads default --blas-threads default --data-format pencil-starneig-blacs --init default --n 4000 --complex-distr uniform --select-distr uniform --complex-ratio 0.500000 --zero-ratio 0.010000 --inf-ratio 0.010000 --select-ratio 0.350000 --data-distr default --section-height 1024 --section-width 1024 --solver starneig --cores default --gpus default --tile-size 256 --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal reordering:normal residual:normal --reordering-fail-threshold 10000 --reordering-warn-threshold 100000 --residual-fail-threshold 1000 --residual-warn-threshold 500 --repeat 1 --warmup 0
INIT...
PREPARE...
PROCESS...
[starneig][message] Attempting to set tile size to 256.
[starneig][message] Setting tile size to 256.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 1821 MS
FINALIZE...
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 26 u, min = 0 u, max = 5807 u
|Q ~A Q^T - A| / |A| = 47 u
|Q Q^T - I| / |I| = 33 u
================================================================
TIME = 1821 MS [avg 1821 MS, cv 0.00, min 1821 MS, max 1821 MS]
NO FAILED SCHUR FORM TESTS
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 26 u, cv 0.00, min 26 u, max 26 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 5807 u, cv 0.00, min 5807 u, max 5807 u]
|Q ~A Q^T - A| / |A| = [avg 47 u, cv 0.00, min 47 u, max 47 u]
|Q Q^T - I| / |I| = [avg 33 u, cv 0.00, min 33 u, max 33 u]
```

Rank 1 output:

```
MPI INIT...
THREADS: Found 3 PUs.
TEST: --mpi --mpi-mode serialized --seed 1551464108 --experiment reorder --omp-threads default --blas-threads default --data-format pencil-starneig-blacs --init default --n 4000 --complex-distr uniform --select-distr uniform --complex-ratio 0.500000 --zero-ratio 0.010000 --inf-ratio 0.010000 --select-ratio 0.350000 --data-distr default --section-height 1024 --section-width 1024 --solver starneig --cores default --gpus default --tile-size 256 --window-size default --values-per-chain default --small-window-size default --small-window-threshold default --update-width default --update-height default --plan default --blueprint default --hooks schur:normal reordering:normal residual:normal --reordering-fail-threshold 10000 --reordering-warn-threshold 100000 --residual-fail-threshold 1000 --residual-warn-threshold 500 --repeat 1 --warmup 0
INIT...
PREPARE...
PROCESS...
[starneig][message] Attempting to set tile size to 256.
[starneig][message] Setting tile size to 256.
[starneig][message] Using multi-part task insertion plan.
[starneig][message] Using two-pass backward dummy blueprint.
[starneig][message] Using "rounded" window size.
EXPERIMENT TIME = 1778 MS
FINALIZE...
REORDERING CHECK: Checking selected eigenvalues...
REORDERING CHECK: Checking other eigenvalues...
REORDERING CHECK: mean = 26 u, min = 0 u, max = 5807 u
|Q ~A Q^T - A| / |A| = 47 u
|Q Q^T - I| / |I| = 33 u
================================================================
TIME = 1777 MS [avg 1778 MS, cv 0.00, min 1778 MS, max 1778 MS]
NO FAILED SCHUR FORM TESTS
REORDERING CHECK (WARNINGS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (FAILS): 0 runs effected [avg 0.0, cv 0.00, min 0, max 0]
REORDERING CHECK (MEANS): [avg 26 u, cv 0.00, min 26 u, max 26 u]
REORDERING CHECK (MIN): [avg 0 u, cv 0.00, min 0 u, max 0 u]
REORDERING CHECK (MAX): [avg 5807 u, cv 0.00, min 5807 u, max 5807 u]
|Q ~A Q^T - A| / |A| = [avg 47 u, cv 0.00, min 47 u, max 47 u]
|Q Q^T - I| / |I| = [avg 33 u, cv 0.00, min 33 u, max 33 u]
```
