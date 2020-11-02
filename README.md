# Introduction

StarNEig library aims to provide a complete task-based software stack for
solving **dense** **nonsymmetric** (generalized) eigenvalue problems. The
library is built on top of the [StarPU](http://starpu.gforge.inria.fr/) runtime
system and targets both **shared memory** and **distributed memory** machines.
Some components of the library support **GPUs**.

The four main components of the library are:

 - **Hessenberg(-triangular) reduction**: A dense matrix (or a dense matrix
   pair) is reduced to upper Hessenberg (or Hessenberg-triangular) form.
 - **Schur reduction (QR/QZ algorithm)**: A upper Hessenberg matrix (or a
   Hessenberg-triangular matrix pair) is reduced to (generalized) Schur form.
   The (generalized) eigenvalues can be determined from the diagonal blocks.
 - **Eigenvalue reordering and deflating subspaces**: Reorders a user-selected
   set of (generalized) eigenvalues to the upper left corner of an updated
   (generalized) Schur form.
 - **Computation of eigenvectors**: Computes (generalized) eigenvectors for a
   user-selected set of (generalized) eigenvalues.

The library has been developed as a part of the [NLAFET](https://www.nlafet.eu/)
project. The project has received funding from the European Union’s Horizon 2020
research and innovation programme under grant agreement No. 671633. Support has
also been received from eSSENCE, a collaborative e-Science programme funded by
the Swedish Government via the Swedish Research Council (VR), and VR Grant
E0485301. The development and performance evaluations were performed on
resources provided by the Swedish National Infrastructure for Computing (SNIC)
at HPC2N partially funded by VR through grant agreement No. 2016-07213. The
library is published under open-source [BSD 3-Clause license](LICENSE.md).

Please cite the following article when referring to StarNEig:
> Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Task-based,
> GPU-accelerated and Robust Library for Solving Dense Nonsymmetric Eigenvalue
> Problems*, Concurrency and Computation: Practice and Experience, 2020,
> doi: [10.1002/cpe.5915](https://doi.org/10.1002/cpe.5915)

Please see [publications](PUBLICATIONS.md) and [authors](AUTHORS.md).

## Performance

Performance comparisons against MAGMA (GPU) and ScaLAPACK (distributed memory),
and strong scalability on shared and distributed memory machines:

![](docs/figures/performance.png)

Also, see following publications:

 - Mirko Myllykoski: *A Task-based Multi-shift QR/QZ Algorithm with Aggressive
   Early Deflation*, [arXiv:2007.03576](https://arxiv.org/abs/2007.03576)
 - Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Task-based,
   GPU-accelerated and Robust Library for Solving Dense Nonsymmetric Eigenvalue
   Problems*, Concurrency and Computation: Practice and Experience, 2020,
   doi: [10.1002/cpe.5915](https://doi.org/10.1002/cpe.5915)
 - Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen, Angelika Schwarz,
   Bo Kågström: *D2.7 Eigenvalue solvers for nonsymmetric problems*, public
   NLAFET deliverable, 2019
   ([download](http://www.nlafet.eu/wp-content/uploads/2019/04/D2.7-EVP-solvers-evaluation-final.pdf))

## Current status (stable series)

The library currently supports only real arithmetic (real input and output
matrices but **real and/or complex** eigenvalues and eigenvectors). In addition,
some interface functions are implemented as LAPACK and ScaLAPACK wrapper
functions.

Standard eigenvalue problems:

| Component             |  Shared memory  | Distributed memory |      CUDA      |
|-----------------------|:---------------:|:------------------:|:--------------:|
| Hessenberg reduction  |  **Complete**   |      ScaLAPACK     | **Single GPU** |
| Schur reduction       |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvalue reordering |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvectors          |  **Complete**   |        ---         |      ---       |

Generalized eigenvalue problems:

| Component             |  Shared memory  | Distributed memory |      CUDA      |
|-----------------------|:---------------:|:------------------:|:--------------:|
| HT reduction          |     LAPACK      |     3rd party      |      ---       |
| Schur reduction       |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvalue reordering |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvectors          |  **Complete**   |        ---         |      ---       |

Please see [changelog](CHANGELOG.md) and [known problems](KNOWN_PROBLEMS.md).

## Documentation:

The StarNEig User's Guide is available in both HTML and PDF formats at
https://nlafet.github.io/StarNEig. The PDF version is also available under
[releases](https://github.com/NLAFET/StarNEig/releases).

## Installation

Prebuild binary packages (Ubuntu 18.04 and Ubuntu 20.04) are available under
[releases](https://github.com/NLAFET/StarNEig/releases) and can be installed
with the following command:

```
$ sudo apt install ./StarNEig-v0.xx.yy-ubuntu-vv.uu.deb
```

The binary packages rely on mainstream StarPU packages and do not necessary
provide full functionality.

For full functionality, it is recommended that StarNEig (and StarPU) are
compiled from the source code, see below and/or the StarNEig User's Guide.
Please consider using one of the tested
[release](https://github.com/NLAFET/StarNEig/releases) versions.

### Dependencies

Library dependencies:

 - Linux
 - CMake 3.3 or newer
 - Portable Hardware Locality (hwloc)
 - Starpu 1.2 or 1.3
    - Newer versions require the user set the `STARPU_LIBRARIES`,
      `STARPU_MPI_LIBRARIES` and `STARPU_INCLUDE_PATH` environmental variables.
 - OpenBLAS, MKL, GotoBLAS or single-threaded BLAS library
 - LAPACK
 - MPI (optional)
 - CUDA + cuBLAS (optional)
 - ScaLAPACK + BLACS (optional)

Test program and example code dependencies:

 - pkg-config
 - GNU Scientific Library (optional)
 - MAGMA (optional)

### Configure, build and install

To compile, validate and install StarNEig, execute in the same directory as this
`README.md` file:
```
$ mkdir build
$ cd build/
$ cmake ../
$ make
$ make test
$ sudo make install
```

## Example

The following example demonstrates how a dense matrix `A` is reduced to real
Schur form:

~~~~~~~~~~~~~~~{.c}
// my_program.c
#include <starneig/starneig.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int n = 3000;
    srand((unsigned) time(NULL));

    // generate a random matrix A
    int ldA = ((n/8)+1)*8;
    double *A = malloc(n*ldA*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A[j*ldA+i] = 2.0*rand()/RAND_MAX - 1.0;

    // generate an identity matrix Q
    int ldQ = ((n/8)+1)*8;
    double *Q = malloc(n*ldA*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            Q[j*ldQ+i] = i == j ? 1.0 : 0.0;

    // allocate space for the eigenvalues
    double *real = malloc(n*sizeof(double));
    double *imag = malloc(n*sizeof(double));

    // initialize the StarNEig library
    starneig_node_init(STARNEIG_USE_ALL, STARNEIG_USE_ALL, STARNEIG_HINT_SM);

    // reduce matrix A to real Schur form S = Q^T A Q
    starneig_SEP_SM_Reduce(
        n, A, ldA, Q, ldQ, real, imag, NULL, NULL, NULL, NULL);

    // de-initialize the StarNEig library
    starneig_node_finalize();

    free(A); free(Q); free(real); free(imag);

    return 0;
}
~~~~~~~~~~~~~~~

Compile:
```
$ gcc -o my_program my_program.c -lstarneig
```
or:
```
$ gcc -o my_program my_program.c $(pkg-config --cflags starneig) $(pkg-config --libs starneig)
```
