# Introduction

StarNEig library aims to provide a full suite of algorithms for solving
**non-symmetric** (generalized) eigenvalue problems. The library is built on top
of the *StarPU* runtime system and targets both shared memory and distributed
memory machines. Some components of the library support GPUs.

The four main components of the library are:

 - **Hessenberg(-triangular) reduction**: A dense matrix (or a dense matrix
   pencil) is reduced to upper Hessenberg (or Hessenberg-triangular) form.
 - **Schur reduction**: A upper Hessenberg matrix (or a Hessenberg-triangular
   matrix pencil) is reduced to (generalized) Schur form. The (generalized)
   eigenvalues can be determined from the diagonal blocks.
 - **Eigenvalue reordering**: Reorders a user-selected set of (generalized)
   eigenvalues to the upper left corner of an updated (generalized) Schur form.
 - **Eigenvectors**: Computes (generalized) eigenvectors for a user-selected
   set of (generalized) eigenvalues.

The library has been developed as a part of the NLAFET project. The project has
received funding from the European Union’s Horizon 2020 research and innovation
programme under grant agreement No. 671633. Support has also been received
from eSSENCE, a collaborative e-Science programme funded by the Swedish
Government via the Swedish Research Council (VR).

## Current status

The library is currently in a beta state and only real arithmetic is supported.
In addition, some interface functions are implemented as LAPACK and ScaLAPACK
wrappers.

Current status with standard eigenvalue problems:

| Component              |   Shared memory  | Distributed memory | Accelerators (GPUs)  |
|------------------------|:----------------:|:------------------:|:--------------------:|
| Hessenberg reduction   |   **Complete**   | ScaLAPACK wrapper  |     **Single GPU**   |
| Schur reduction        |   **Complete**   |    *Experimental*  |     *Experimental*   |
| Eigenvalue reordering  |   **Complete**   |    **Complete**    |     *Experimental*   |
| Eigenvectors           |   **Complete**   |Waiting integration |      Not planned     |

Current status with generalized eigenvalue problems:

| Component                         |      Shared memory     | Distributed memory | Accelerators (GPUs)  |
|-----------------------------------|:----------------------:|:------------------:|:--------------------:|
| Hessenberg-triangular reduction   |LAPACK wrapper / Planned| ScaLAPACK wrapper  |      Not planned     |
| Generalized Schur reduction       |      **Complete**      |    *Experimental*  |     *Experimental*   |
| Generalized eigenvalue reordering |      **Complete**      |    **Complete**    |     *Experimental*   |
| Generalized eigenvectors          |      **Complete**      |Waiting integration |      Not planned     |

## Documentation:

 - [Installation](docs/_1_installation.md)
 - [Test driver](docs/_7_test_driver.md)

### Doxygen documentation

Dependencies:

 - CMake 3.3 or newer
 - Doxygen
 - Latex + pdflatex

The documentation can be build as follows (execute in the same directory as this
`README.md` file):

```
$ mkdir build_doc
$ cd build_doc/
$ cmake ../docs/
$ make
```

The PDF documentation is copied to `build_doc/starneig_manual.pdf`. The HTML
documentation is available at `build_doc/html` directory.

## Quickstart guide

### Dependencies

Library dependencies:

 - Linux (not tested in Window or Mac OS X)
 - CMake 3.3 or newer
 - Portable Hardware Locality (hwloc)
 - Starpu 1.2 or 1.3 (newer versions require minor changes to
   `src/CMakeLists.txt`; `SUPPORTED_STARPU`)
 - BLAS (preferably a multi-threaded variant that has an option to change the
   thread count)
 - LAPACK
 - MPI (optional)
 - CUDA (optional)
 - ScaLAPACK (optional)

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
