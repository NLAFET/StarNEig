# Current status

The library currently supports only real arithmetic (real input and output
matrices but real and/or complex eigenvalues and eigenvectors). In addition,
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

## Known problems

 - The `STARPU_MINIMUM_AVAILABLE_MEM` and `STARPU_TARGET_AVAILABLE_MEM`
   environmental variables can be used to fix some GPU related memory allocation
   problems:
```
STARPU_MINIMUM_AVAILABLE_MEM=10 STARPU_TARGET_AVAILABLE_MEM=15 ...
```
 - If the GPU support is enabled, then the starneig_SEP_SM_Hessenberg()
   interface function cannot always handle problems that do not fit into GPU's
   memory. The cause of this problem is is not known.
 - The outputs of the starneig_GEP_SM_Schur() and starneig_GEP_DM_Schur()
   interface functions are not in the so-called standard format. It is possible
   that some diagonal entries in the right-hand side output matrix are negative.
   This will be fixed in the next version of the library.
 - The starneig_GEP_SM_Eigenvectors() interface function may scale the input
   matrices. This will be fixed in the next version of the library.

### Known compatibility problems

 - Some older OpenMPI versions (pre summer 2017, e.g. <= 2.1.1) have a bug that
   might lead to a segmentation fault during a parallel AED.
 - OpenBLAS version 0.3.1 has a bug that might lead to an incorrect result.
 - OpenBLAS versions 0.3.3-0.3.5 might lead to poor scalability.
 - Some MKL versions might lead to poor scalability. The problem appears to be
   related to Intel's OpenMP library. Setting the `KMP_AFFINITY` environmental
   variable to `disabled` fixes the problem in all known cases.
 - StarPU versions 1.2.4 - 1.2.8 and some StarPU 1.3 snapshots cause poor CUDA
   performance. The problem can be fixed by compiling StarPU with
   `--disable-cuda-memcpy-peer`. It is possible that newer versions of StarPU
   are also effected by this problem.
 - The library has an unsolved memory leak problem with OpenMPI. Only large
   problem sizes are effected. It is not known whether this problem is related
   to StarNEig, StarPU, OpenMPI or something else. A memory leak is sometimes
   accompanied by the following warning:
```
mpool.c:38   UCX  WARN  object 0x2652000 was not returned to mpool ucp_requests
```
   The problem is known to occur with PMIx 2.2.1, UCX 1.5.0, OpenMPI 3.1.3, and
   StarPU 1.2.8.
