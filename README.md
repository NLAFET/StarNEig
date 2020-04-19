# Introduction

![StarNEig tests](https://github.com/NLAFET/StarNEig/workflows/StarNEig%20tests/badge.svg) ![StarNEig manual](https://github.com/NLAFET/StarNEig/workflows/StarNEig%20manual/badge.svg) ![StarNEig version](https://github.com/NLAFET/StarNEig/workflows/StarNEig%20version/badge.svg)

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

The library is open source and published under BSD 3-Clause licence.

Please cite the following article when refering to StarNEig:
> Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Introduction to StarNEig — A Task-based Library for Solving Nonsymmetric Eigenvalue Problems*, In Parallel Processing and Applied Mathematics, 13th International Conference, PPAM 2019, Bialystok, Poland, September 8–11, 2019, Revised Selected Papers, Part I, Lecture Notes in Computer Science, Vol. 12043, Wyrzykowski R., Deelman E., Dongarra J., Karczewski K. (eds), Springer International Publishing, pp. 70-81, 2020, doi: [10.1007/978-3-030-43229-4_7](https://doi.org/10.1007/978-3-030-43229-4_7)

## Current status

The library is currently in a beta state and only real arithmetic is supported.
In addition, some interface functions are implemented as LAPACK and ScaLAPACK
wrappers.

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
| HT reduction          |     LAPACK      |      ScaLAPACK     |      ---       |
| Schur reduction       |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvalue reordering |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvectors          |  **Complete**   |        ---         |      ---       |

Please see the *Known problems* section in the StarNEig manual.

## Related publications

### Research papers

 - Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Task-based, GPU-accelerated and Robust Library for Solving Dense Nonsymmetric Eigenvalue Problems*, Invited article submitted to Concurrency and Computation: Practice and Experience, [arXiv:2002.05024](https://arxiv.org/abs/2002.05024)
- Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Introduction to StarNEig — A Task-based Library for Solving Nonsymmetric Eigenvalue Problems*, In Parallel Processing and Applied Mathematics, 13th International Conference, PPAM 2019, Bialystok, Poland, September 8–11, 2019, Revised Selected Papers, Part I, Lecture Notes in Computer Science, Vol. 12043, Wyrzykowski R., Deelman E., Dongarra J., Karczewski K. (eds), Springer International Publishing, pp. 70-81, 2020, doi: [10.1007/978-3-030-43229-4_7](https://doi.org/10.1007/978-3-030-43229-4_7)
- Carl Christian Kjelgaard Mikkelsen, Mirko Myllykoski: *Parallel Robust Computation of Generalized Eigenvectors of Matrix Pencils*, presented at PPAM 2019, In Parallel Processing and Applied Mathematics, 13th International Conference, PPAM 2019, Bialystok, Poland, September 8–11, 2019, Revised Selected Papers, Part I, Lecture Notes in Computer Science, Vol. 12043, Wyrzykowski R., Deelman E., Dongarra J., Karczewski K. (eds), Springer International Publishing, pp. 58-69, 2020, doi: [10.1007/978-3-030-43229-4_6](https://doi.org/10.1007/978-3-030-43229-4_6)
 - Carl Christian Kjelgaard Mikkelsen, Angelika Schwarz, and Lars Karlsson: *Parallel Robust Solution of Triangular Linear Systems*, Concurrency and Computation: Practice and Experience, 31 (19), 2019, doi: [10.1016/j.parco.2019.04.001](https://doi.org/10.1016/j.parco.2019.04.001)
 - Mirko Myllykoski: *A Task-Based Algorithm for Reordering the Eigenvalues of a Matrix in Real Schur Form*, In Parallel Processing and Applied Mathematics, 12th International Conference, PPAM 2017, Lublin, Poland, September 10-13, 2017, Revised Selected Papers, Part I, Lecture Notes in Computer Science, Vol. 10777, Wyrzykowski R., Dongarra J., Deelman E., Karczewski K. (eds), Springer International Publishing, pp. 207-216, 2018, doi: [10.1007/978-3-319-78024-5_19](https://doi.org/10.1007/978-3-319-78024-5_19)
 - Carl Christian Kjelgaard Mikkelsen, Lars Karlsson. *Blocked Algorithms for Robust Solution of Triangular Linear Systems*, In Parallel Processing and Applied Mathematics, 12th International Conference, PPAM 2017, Lublin, Poland, September 10-13, 2017, Revised Selected Papers, Part I, Lecture Notes in Computer Science, Vol. 10777, Wyrzykowski R., Dongarra J., Deelman E., Karczewski K. (eds), Springer International Publishing, pp. 207-216, 2018, doi: [10.1007/978-3-319-78024-5_7](https://doi.org/10.1007/978-3-319-78024-5_7)

### Reports, deliverables etc

 - Angelika Schwarz, Carl Christian Kjelgaard Mikkelsen, Lars Karlsson: *Robust Parallel Eigenvector Computation For the Non-Symmetric Eigenvalue Problem*, Report UMINF 20.02, Department of Computing Science, Umeå University, SE-901 87 Umeå, Sweden, 2020 ([download](https://webapps.cs.umu.se/uminf/index.cgi?year=2020&number=2))
 - Angelika Schwarz: *Towards efficient overflow-free solvers for systems of triangular type*, Licentiate thesis, Department of computing science, Umeå University, ISSN: 0348-0542, 2019
 - Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen, Angelika Schwarz, Bo Kågström: *D2.7 Eigenvalue solvers for nonsymmetric problems*, public NLAFET deliverable, 2019 ([download](http://www.nlafet.eu/wp-content/uploads/2019/04/D2.7-EVP-solvers-evaluation-final.pdf))
 - Lars Karlsson, Mahmoud Eljammaly, Mirko Myllykoski: *D6.5 Evaluation of auto-tuning techniques*, public NLAFET deliverable, 2019 ([download](http://www.nlafet.eu/wp-content/uploads/2019/04/D6-5-eval-auto-tuning-final.pdf))
 - Bo Kågström et al.: *D7.8 Release of the NLAFET library*, public NLAFET deliverable, 2019 ([download](http://www.nlafet.eu/wp-content/uploads/2019/04/D7-8-Release-NLAFET-library-final.pdf))
 - Mirko Myllykoski, Lars Karlsson, Bo Kågström, Mahmoud Eljammaly, Srikara Pranesh, Mawussi Zounon: *D2.6 Prototype Software for Eigenvalue Problem Solvers*, public NLAFET deliverable, 2018 ([download](http://www.nlafet.eu/wp-content/uploads/2016/01/Deliverable2.6-180427-rev.pdf))
 - Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen, Lars Karlsson, Bo Kågström: *Task-Based Parallel Algorithms for Reordering of Matrices in Real Schur Forms*, NLAFET Working Note WN-11, 2017. Also as Report UMINF 17.11, Department of Computing Science, Umeå University, SE-901 87 Umeå, Sweden ([download](http://www8.cs.umu.se/research/uminf/index.cgi?year=2017&number=11))
 - Carl Christian Kjelgaard Mikkelsen, Mirko Myllykoski, Björn Adlerborn, Lars Karlsson, Bo Kågström: *D2.5 Eigenvalue Problem Solvers*, public NLAFET deliverable, 2017 ([download](http://www.nlafet.eu/wp-content/uploads/2016/01/D2.5-EVP-solvers-170427_v1.0-final.pdf))

## Documentation:

HTML and PDF documentation can be found from https://nlafet.github.io/StarNEig.

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

The PDF documentation is copied to `build_doc/starneig_manual.pdf` and the HTML
documentation is available at `build_doc/html` directory.

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
