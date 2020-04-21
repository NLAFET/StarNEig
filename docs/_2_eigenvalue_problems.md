# Eigenvalue problems

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
  B = Q_{1} R Z_{1}^{H},
\f]
where \f$Q_{1}, Z_{1}\f$ are unitary, \f$H\f$ is upper Hessenberg, and \f$R\f$
is upper triangular. This is done in order to greatly accelerate the subsequent
computation of a generalized Schur decomposition.

### Reduction to generalized Schur form

Starting from the Hessenberg-triangular pencil \f$H - \lambda R\f$ we compute a
*generalized Schur decomposition*
\f[
  H = Q_{2} S Z_{2}^H,
  \quad
  R = Q_{2} T Z_{2}^{H},
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
  R = Q_{2} T Z_{2}^{T},
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
