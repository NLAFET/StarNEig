## Known problems

 - If the CUDA support is enabled, then the starneig_SEP_SM_Hessenberg()
   interface function cannot always handle problems that do not fit into GPU's
   memory. The cause of this problem is not known.
 - The outputs of the starneig_GEP_SM_Schur() and starneig_GEP_DM_Schur()
   interface functions are not always in the so-called standard format. It is
   possible that some diagonal entries in the right-hand side output matrix are
   negative.
 - The starneig_GEP_SM_Eigenvectors() interface function may scale the input
   matrices.

### Known compatibility problems

#### BLAS

 - With some OpenBLAS versions, it is necessary to set the
   `OPENBLAS_NUM_THREADS` environmental variable to value `1`
   (`export OPENBLAS_NUM_THREADS=1`).
 - Some MKL versions can cause poor scalability. The problem appears to be
   related to Intel's OpenMP library. Setting the `KMP_AFFINITY` environmental
   variable to value `disabled` fixes the problem
   (`export KMP_AFFINITY=disabled`).
 - OpenBLAS version 0.3.1 has a bug that can cause an incorrect result.
 - OpenBLAS versions 0.3.3-0.3.5 can cause poor scalability.

#### MPI

 - Some older OpenMPI versions (<= 2.1.1) have a bug that can cause a
   segmentation fault during a parallel AED.
 - The library has an unsolved memory leak problem with OpenMPI. Only large
   problem sizes are effected. It is not known whether this problem is related
   to StarNEig, StarPU, OpenMPI or something else. The problem is known to occur
   with PMIx 2.2.1, UCX 1.5.0, OpenMPI 3.1.3, and StarPU 1.2.8. The memory leak
   is sometimes accompanied by the following warning:
```
mpool.c:38   UCX  WARN  object 0x2652000 was not returned to mpool ucp_requests
```
 - The test program can trigger the following bug in UCX 1.6.1:
   https://github.com/openucx/ucx/issues/4525

#### StarPU

 - For optimal CUDA performance, StarPU version that is newer than 1.3.3 is
   recommended.
 - StarPU versions 1.2.4 - 1.2.8 and some StarPU 1.3 snapshots cav cause poor
   CUDA performance. The problem can be fixed by compiling StarPU with
   `--disable-cuda-memcpy-peer`. It is possible that newer versions of StarPU
   are also effected by this problem.
 - The `STARPU_MINIMUM_AVAILABLE_MEM` and `STARPU_TARGET_AVAILABLE_MEM`
   environmental variables can be used to fix some GPU-related memory allocation
   problems:
```
STARPU_MINIMUM_AVAILABLE_MEM=10 STARPU_TARGET_AVAILABLE_MEM=15 ...
```
