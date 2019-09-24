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

### Known problems

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
 - The `STARPU_MINIMUM_AVAILABLE_MEM` and `STARPU_TARGET_AVAILABLE_MEM`
   environmental variables can be used to fix some GPU related memory allocation
   problems:
```
STARPU_MINIMUM_AVAILABLE_MEM=10 STARPU_TARGET_AVAILABLE_MEM=15 ...
```
 - The library has an unsolved memory leak problem with OpenMPI. Only large
   problem sizes are effected. It is not known whether this problem is related
   to StarNEig, StarPU, OpenMPI or something else. A memory leak is sometimes
   accompanied by the following warning:
```
mpool.c:38   UCX  WARN  object 0x2652000 was not returned to mpool ucp_requests
```
   The problem is known to occur with PMIx 2.2.1, UCX 1.5.0, OpenMPI 3.1.3, and
   StarPU 1.2.8.
 - If the GPU support is enabled, then the starneig_SEP_SM_Hessenberg()
   interface function cannot always handle problems that do not fit into GPU's
   memory. The cause of this problem is is not known.
 - The outputs of the starneig_GEP_SM_Schur() and starneig_GEP_DM_Schur()
   interface functions are not in the so-called standard format. It is possible
   that some diagonal entries in the right-hand side output matrix are negative.
   This will be fixed in the next version of the library.
 - The starneig_GEP_SM_Eigenvectors() interface function may scale the input
   matrices. This will be fixed in the next version of the library.

## The standard eigenvalue problem

Given a square matrix \f$A\f$ of size \f$n \times n\f$, the *standard eigenvalue
problem* (SEP) consists of finding *eigenvalues* \f$\lambda_i \in \mathbb C\f$
and associated *eigenvectors* \f$0 \neq v_i \in \mathbb C^{n}\f$ such that
\f[
  A v_i = \lambda_i v_i, \text{ for } i = 1, 2, \dots, n.
\f]
The eigenvalues are the \f$n\f$ (potentially complex) roots of the polynomial
\f$\text{det}(A - \lambda I) = 0\f$ of degree \f$n\f$. There is often a full set
of \f$n\f$ linearly independent eigenvectors, but if there are *multiple*
eigenvalues (i.e., if \f$\lambda_{i} = \lambda_{j}\f$ for some \f$i \neq j\f$)
then there might not be a full set of independent eigenvectors.

### Reduction to Hessenberg form

The dense matrix \f$A\f$ is condensed to *Hessenberg form* by computing a
*Hessenberg decomposition*
\f[
  A = Q_{1} H Q_{1}^{H},
\f]
where \f$Q_{1}\f$ is unitary and \f$H\f$ is upper Hessenberg.
This is done in order to greatly accelerate the subsequent computation of a
Schur decomposition since when working on \f$H\f$ of size \f$n \times n\f$,
the amount of work in each iteration of the QR algorithm is reduced from
\f$\mathcal{O}(n^3)\f$ to \f$\mathcal{O}(n^2)\f$ flops.

### Reduction to Schur form

Starting from the Hessenberg matrix \f$H\f$ we compute a *Schur decomposition*
\f[
  H = Q_{2} S Q_{2}^H,
\f]
where \f$Q_{2}\f$ is unitary and \f$S\f$ is upper triangular. The eigenvalues of
\f$A\f$ can now be determined as they appear on the diagonal of \f$S\f$, i.e.,
\f$\lambda_i = s_{ii}\f$. For real matrices there is a similar decomposition
known as the *real Schur decomposition*
\f[
  H = Q_{2} S Q_{2}^T,
\f]
where \f$Q_{2}\f$ is orthogonal and \f$S\f$ is upper quasi-triangular with
\f$1\times 1\f$ and \f$2 \times 2\f$ blocks on the diagonal. The
\f$1 \times 1\f$ blocks correspond to the real eigenvalues and each
\f$2 \times 2\f$ block corresponds to a pair of complex conjugate eigenvalues.


### Eigenvalue reordering and invariant subspaces

Given a subset consisting of \f$m \leq n\f$ of the eigenvalues, we can
*reorder the eigenvalues* on the diagonal of the Schur form by constructing a
unitary matrix \f$Q_{3}\f$ such that
\f[
  S =
  Q_{3}
  \begin{bmatrix}
    \hat S_{11} & \hat S_{12} \\
    0 & \hat S_{22}
  \end{bmatrix}
  Q_{3}^{H}
\f]
and the eigenvalues of the \f$m \times m\f$ block \f$\hat S_{11}\f$ are the
selected eigenvalues. The first \f$m\f$ columns of \f$Q_{3}\f$ span an
*invariant subspace* associated with the selected eigenvalues.

### Computation of eigenvectors

Given a subset consisting of \f$m \leq n\f$ of the eigenvalues \f$\lambda_{i}\f$
for \f$i = 1, 2, \ldots, m\f$ and a Schur decomposition \f$A = Q S Q^{H}\f$, we
can compute for each \f$\lambda_{i}\f$ an *eigenvector* \f$v_{i} \neq 0\f$ such
that \f$A v_{i} = \lambda_{i} v_{i}\f$ by first computing an eigenvector
\f$w_{i}\f$ of \f$S\f$ and then transform it back to the original basis by
pre-multiplication with \f$Q\f$.

## The generalized eigenvalue problem

Given a square matrix pencil \f$A - \lambda B\f$, where \f$A, B \in
\mathbb{C}^{n \times n}\f$, the *generalized eigenvalue problem* (GEP) consists
of finding *generalized eigenvalues* \f$\lambda_i \in \mathbb C\f$ and
associated *generalized eigenvectors* \f$0 \neq v_i \in \mathbb C^{n}\f$ such
that
\f[
  A v_{i} = \lambda_{i} B v_{i}, \text{ for } i = 1, 2, \dots, n.
\f]
The eigenvalues are the \f$n\f$ (potentially complex) roots of the polynomial
\f$\text{det}(A - \lambda B) = 0\f$ of degree \f$n\f$. There is often a full set
of \f$n\f$ linearly independent generalized eigenvectors, but if there are
*multiple eigenvalues* (i.e., if \f$\lambda_{i} = \lambda_{j}\f$ for some \f$i
\neq j\f$) then there might not be a full set of independent eigenvectors.

At least in principle, a GEP can be transformed into a SEP provided that \f$B\f$
is invertible, since
\f[
  A v = \lambda B v \Leftrightarrow (B^{-1} A) v = \lambda v.
\f]
However, in finite precision arithmetic this practice is not recommended.

### Reduction to Hessenberg-triangular form

The dense matrix pair \f$(A, B)\f$ is condensed to *Hessenberg-triangular form*
by computing a *Hessenberg-triangular decomposition*
\f[
  A = Q_{1} H Z_{1}^{H},
  \quad
  B = Q_{1} Y Z_{1}^{H},
\f]
where \f$Q_{1}, Z_{1}\f$ are unitary, \f$H\f$ is upper Hessenberg, and \f$Y\f$
is upper triangular. This is done in order to greatly accelerate the subsequent
computation of a generalized Schur decomposition.

### Reduction to generalized Schur form

Starting from the Hessenberg-triangular pencil \f$H - \lambda Y\f$ we compute a
*generalized Schur decomposition*
\f[
  H = Q_{2} S Z_{2}^H,
  \quad
  Y = Q_{2} T Z_{2}^{H},
\f]
where \f$Q_{2}, Z_{2}\f$ are unitary and \f$S, T\f$ are upper triangular. The
eigenvalues of \f$A - \lambda B\f$ can now be determined from the diagonal
element pairs \f$(s_{ii}, t_{ii})\f$, i.e., \f$\lambda_i = s_{ii} / t_{ii}\f$
(if \f$t_{ii} \neq 0\f$). If \f$s_{ii} \neq 0\f$ and \f$t_{ii} = 0\f$, then
\f$\lambda_{i} = \infty\f$ is an *infinite eigenvalue* of the matrix pair
\f$(S,T)\f$. (If both \f$s_{ii} = 0\f$ and \f$t_{ii} = 0\f$ for some \f$i\f$,
then the pencil is *singular* and the eigenvalues are undetermined; all complex
numbers are eigenvalues). For real matrix pairs there is a similar decomposition
known as the *real generalized Schur decomposition*
\f[
  H = Q_{2} S Z_{2}^T,
  \quad
  Y = Q_{2} T Z_{2}^{T},
\f]
where \f$Q_{2}, Z_{2}\f$ are orthogonal, \f$S\f$ is upper quasi-triangular with
\f$1 \times 1\f$ and \f$2 \times 2\f$ blocks on the diagonal, and \f$T\f$ is
upper triangular. The \f$1 \times 1\f$ blocks on the diagonal of
\f$S - \lambda T\f$ correspond to the real generalized eigenvalues and each
\f$2 \times 2\f$ block corresponds to a pair of complex conjugate generalized
eigenvalues.


### Eigenvalue reordering and deflating subspaces

Given a subset consisting of \f$m \leq n\f$ of the generalized eigenvalues, we
can *reorder the generalized eigenvalues* on the diagonal of the generalized
Schur form by constructing unitary matrices \f$Q_{3}\f$ and \f$Z_{3}\f$ such
that
\f[
  S - \lambda T =
  Q_{3}
    \begin{bmatrix}
      \hat S_{11} - \lambda \hat T_{11} & \hat S_{12} - \lambda \hat T_{12} \\
      0 & \hat S_{22} - \lambda \hat T_{22}
    \end{bmatrix}
  Z_{3}^{H}
\f]
and the eigenvalues of the \f$m \times m\f$ block pencil
\f$\hat S_{11} - \lambda \hat T_{11}\f$ are the selected generalized
eigenvalues. The first \f$m\f$ columns of \f$Z_{3}\f$ spans a right
*deflating subspace* associated with the selected generalized eigenvalues.

### Computation of generalized eigenvectors

Given a subset consisting of \f$m \leq n\f$ of the eigenvalues \f$\lambda_{i}\f$
for \f$i = 1, 2, \ldots, m\f$ and a generalized Schur decomposition
\f$A - \lambda B = Q (S - \lambda T) Z^{H}\f$, we can compute for each
\f$\lambda_{i}\f$ a *generalized eigenvector* \f$v_{i} \neq 0\f$ such that
\f$A v_{i} = \lambda_{i} B v_{i}\f$ by first computing a generalized eigenvector
\f$w_{i}\f$ of \f$S - \lambda_{i} T\f$ and then transform it back to the
original basis by pre-multiplication with \f$Z\f$.
