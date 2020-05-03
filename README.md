# Introduction

StarNEig library aims to provide a complete task-based software stack for
solving **dense** **nonsymmetric** (generalized) eigenvalue problems. The
library is built on top of the [StarPU](http://starpu.gforge.inria.fr/)
runtime system and targets both shared memory and distributed memory machines.
Some components of the library support GPUs.

The four main components of the library are:

 - **Hessenberg(-triangular) reduction**: A dense matrix (or a dense matrix
   pair) is reduced to upper Hessenberg (or Hessenberg-triangular) form.
 - **Schur reduction (QR/QZ algorithm)**: A upper Hessenberg matrix (or a
   Hessenberg-triangular matrix pair) is reduced to (generalized) Schur form.
   The (generalized) eigenvalues can be determined from the diagonal blocks.
 - **Eigenvalue reordering**: Reorders a user-selected set of (generalized)
   eigenvalues to the upper left corner of an updated (generalized) Schur form.
 - **Eigenvectors**: Computes (generalized) eigenvectors for a user-selected
   set of (generalized) eigenvalues.

**A brief summary of the StarNEig library** can be found from a recent poster:
*Task-based, GPU-accelerated and Robust Algorithms for Solving Dense
Nonsymmetric Eigenvalue Problems*, Swedish eScience Academy, Lund, Sweden,
October 15-16, 2019
([download](http://www.nlafet.eu/starneig/escience_poster.pdf))

The library has been developed as a part of the NLAFET project. The project has
received funding from the European Union’s Horizon 2020 research and innovation
programme under grant agreement No. 671633. Support has also been received
from eSSENCE, a collaborative e-Science programme funded by the Swedish
Government via the Swedish Research Council (VR), and VR Grant E0485301.

The library is open source and published under
[BSD 3-Clause license](LICENSE.md).

Please cite the following article when refering to StarNEig:
> Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Introduction to
> StarNEig — A Task-based Library for Solving Nonsymmetric Eigenvalue Problems*,
> In Parallel Processing and Applied Mathematics, 13th International Conference,
> PPAM 2019, Bialystok, Poland, September 8–11, 2019, Revised Selected Papers,
> Part I, Lecture Notes in Computer Science, Vol. 12043, Wyrzykowski R., Deelman
> E., Dongarra J., Karczewski K. (eds), Springer International Publishing, pp.
> 70-81, 2020, doi:
> [10.1007/978-3-030-43229-4_7](https://doi.org/10.1007/978-3-030-43229-4_7)

Please see [publications](PUBLICATIONS.md) and [authors](AUTHORS.md).

## Current status (development series)

The library currently supports only real arithmetic (real input and output
matrices but real and/or complex eigenvalues and eigenvectors). In addition,
some interface functions are implemented as LAPACK and ScaLAPACK wrapper
functions.

Standard eigenvalue problems:

| Component             |  Shared memory  | Distributed memory |      CUDA      |
|-----------------------|:---------------:|:------------------:|:--------------:|
| Hessenberg reduction  |  **Complete**   |   *Experimental*   | *Experimental* |
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

HTML and PDF documentation can be found from https://nlafet.github.io/StarNEig
and under [releases](https://github.com/NLAFET/StarNEig/releases).

## Quickstart guide

### Dependencies

Library dependencies:

 - Linux
 - CMake 3.3 or newer
 - Portable Hardware Locality (hwloc)
 - Starpu 1.2 or 1.3
    - Newer versions require minor changes to `src/CMakeLists.txt`;
      `SUPPORTED_STARPU`)
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

Execute in the same directory as this `README.md` file:
```
$ mkdir build
$ cd build/
$ cmake ../
$ make
$ make test
$ sudo make install
```

### Example

The following example demonstrates how a dense matrix `A` is reduced to real
Schur form:

~~~~~~~~~~~~~~~{.c}
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
            A[j*ldA+i] = C[j*ldC+i] = 2.0*rand()/RAND_MAX - 1.0;

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
    starneig_node_init(-1, -1, STARNEIG_HINT_SM);

    // reduce matrix A to real Schur form S = Q^T A Q
    starneig_SEP_SM_Reduce(
        n, A, ldA, Q, ldQ, real, imag, NULL, NULL, NULL, NULL);

    // de-initialize the StarNEig library
    starneig_node_finalize();

    free(A); free(Q); free(real); free(imag);

    return 0;
}
~~~~~~~~~~~~~~~
