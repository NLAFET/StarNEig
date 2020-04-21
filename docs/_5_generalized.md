# Generalized eigenvalue problem

Given a matrix pair \f$(A, B)\f$, where \f$A, B \in \mathbb{R}^{n \times n}\f$,
the *generalized eigenvalue problem* (GEP) consists of finding *generalized
eigenvalues* \f$\lambda_i \in \mathbb C\f$ and associated *generalized
eigenvectors* \f$0 \neq v_i \in \mathbb C^{n}\f$ such that
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

The library provides 12 interface functions for the generalized case.

## Hessenberg-triangular reduction

Given a general matrix pair \f$(A,B)\f$, the
starneig_GEP_SM_HessenbergTriangular() and
starneig_GEP_DM_HessenbergTriangular() interface functions compute a
Hessenberg-triangular decomposition
\f[
    (A,B) = U_1 * (H,R) * U_2^T,
\f]
where \f$H\f$ is upper Hessenberg, \f$R\f$ is upper triangular, and
\f$U_1\f$ and \f$U_2\f$ are orthogonal. On exit, \f$A\f$ is overwritten by
\f$H\f$, \f$B\f$ is overwritten by \f$R\f$, and \f$Q\f$ and \f$Z\f$ (which are
orthogonal matrices on entry) are overwritten by
\f[
    Q \gets Q * U_1 \text{ and } Z \gets Z * U_2.
\f]

This is done in order to greatly accelerate the subsequent computation of a
generalized Schur decomposition.

## Generalized Schur reduction

Given a Hessenberg-triangular decomposition
\f[
    (A,B) = Q * (H,R) * Z^T
\f]
of a general matrix pair \f$(A,B)\f$, the starneig_GEP_SM_Schur() and
starneig_GEP_DM_Schur() interface functions function compute a generalized
Schur decomposition
\f[
    (A,B) = Q * ( U_1 * (S,T) * U_2^T ) * Z^T,
\f]
where \f$S\f$ is upper quasi-triangular with \f$1 \times 1\f$ and
\f$2 \times 2\f$ blocks on the diagonal, \f$T\f$ is a upper triangular
matrix, and \f$U_1\f$ and \f$U_2\f$ are orthogonal. The \f$1 \times 1\f$ blocks
on the diagonal of \f$(S,T)\f$ correspond to the real generalized eigenvalues
and each \f$2 \times 2\f$ block corresponds to a pair of complex conjugate
generalized eigenvalues.

On exit, \f$H\f$ is overwritten by \f$S\f$, \f$R\f$ is overwritten by
\f$T\f$, and \f$Q\f$ and \f$Z\f$ are overwritten by
\f[
    Q \gets Q * U_1 \text{ and } Z \gets Z * U_2.
\f]

The computed generalized eigenvalues are returned as a pair of values
\f$(\alpha,\beta)\f$ such that \f$\alpha/\beta\f$ gives the actual
generalized eigenvalue. The quantity \f$\alpha/\beta\f$ may overflow.

## Generalized eigenvalue reordering

Given a generalized Schur decomposition
\f[
    (A,B) = Q * (S,T) * Z^T
\f]
of a general matrix pair \f$(A,B)\f$ and a selection of generalized
eigenvalues, the starneig_GEP_SM_ReorderSchur() and
starneig_GEP_DM_ReorderSchur() interface functions attempt to compute an updated
generalized Schur decomposition
\f[
    (A,B) = Q * \left( U_1 * \left (
        \begin{bmatrix}
          \hat S_{11} & \hat S_{12} \\
          0 & \hat S_{22}
        \end{bmatrix},
        \begin{bmatrix}
          \hat T_{11} & \hat T_{12} \\
          0 & \hat T_{22}
        \end{bmatrix}
    \right) * U_2^T \right) * Z^T.
\f]
where the selected generalized eigenvalues appear in
\f$(\hat S_{11},\hat T_{11})\f$.

On exit, \f$S\f$ is overwritten by \f$\hat{S}\f$, \f$T\f$ is overwritten by
\f$\hat{T}\f$, and \f$Q\f$ and \f$Z\f$ are overwritten by
\f[
    Q \gets Q * U_1 \text{ and } Z \gets Z * U_2.
\f]

Reordering may in rare cases fail. In such cases the output is guaranteed to
be a Schur decomposition and all (if any) selected generalized eigenvalues
that are correctly placed are marked in the selection array on exit.

Reordering may perturb the generalized eigenvalues and the generalized
eigenvalues after reordering are returned. The computed generalized
eigenvalues are returned as a pair of values \f$(\alpha,\beta)\f$ such
that \f$\alpha/\beta\f$ gives the actual generalized eigenvalue. The quantity
\f$\alpha/\beta\f$ may overflow.

## Combined reduction to generalized Schur form and eigenvalue reordering

Given a general matrix pair \f$(A,B)\f$, the starneig_GEP_SM_Reduce() and
starneig_GEP_DM_Reduce() interface functions compute a (reordered) generalized
Schur decomposition
\f[
    (A,B) = U_1 * (S,T) * U_2^T.
\f]
Optionally, the interface functions attempt to reorder selected generalized
eigenvalues to the top left corner of the generalized Schur decomposition.

On exit, \f$A\f$ is overwritten by \f$S\f$, \f$B\f$ is overwritten by
\f$T\f$, and \f$Q\f$ and \f$Z\f$ (which are orthogonal matrices on entry) are
overwritten by
\f[
    Q \gets Q * U_1 \text{ and } Z \gets Z * U_2.
\f]

The computed generalized eigenvalues are returned as a pair of values
\f$(\alpha,\beta)\f$ such that \f$\alpha/\beta\f$ gives the actual
generalized eigenvalue. The quantity \f$\alpha/\beta\f$ may overflow.

Reordering may in rare cases fail. In such cases the output is guaranteed to
be a Schur-triangular decomposition and all (if any) selected generalized
eigenvalues that are correctly placed are marked in the selection array on
exit.

## Generalized eigenvectors

Given a subset consisting of \f$m \leq n\f$ of the eigenvalues \f$\lambda_{i}\f$
for \f$i = 1, 2, \ldots, m\f$ and a generalized Schur decomposition
\f$(A, B) = Q (S, T) Z^{H}\f$, we can compute for each \f$\lambda_{i}\f$ a
*generalized eigenvector* \f$v_{i} \neq 0\f$ such that
\f$A v_{i} = \lambda_{i} B v_{i}\f$ by first computing a generalized eigenvector
\f$w_{i}\f$ of \f$S - \lambda_{i} T\f$ and then transform it back to the
original basis by pre-multiplication with \f$Z\f$.

Given a generalized Schur decomposition
\f[
    (A,B) = Q * (S,T) * Z^T
\f]
of a general matrix pair \f$(A,B)\f$ and a selection of generalized
eigenvalues, the starneig_GEP_SM_Eigenvectors() and
starneig_GEP_DM_Eigenvectors() interface functions compute and return a
generalized eigenvector for each of the selected generalized eigenvalues.

The generalized eigenvectors are stored as columns in the output matrix \f$X\f$
in the same order as their corresponding generalized eigenvalues appear in the
selection array. A real generalized eigenvector is stored as a single column.
The real and imaginary parts of a complex generalized eigenvector are stored as
consecutive columns.

For a selected pair of complex conjugate generalized eigenvalues, a
generalized eigenvector is computed only for the generalized eigenvalue with
positive imaginary part. Thus, every selected generalized eigenvalue
contributes one column to the output matrix and thus the number of selected
generalized eigenvalues is equal to the number of columns of \f$X\f$.

## Eigenvalue selection helper

Given a Schur-triangular matrix pair \f$(S,T)\f$ and a predicate function,
the starneig_GEP_SM_Select() and starneig_GEP_DM_Select() interface functions
conveniently generate a correct selection array and count the
number of selected generalized eigenvalues. The count is useful when
allocating storage for the generalized eigenvector matrix computed by
starneig_GEP_DM_Eigenvectors().

@code{.c}
// a predicate function that selects all finite generalized eigenvalues that
// have a real part that is larger than a given value
static int predicate(double real, double imag, double beta, void *arg)
{
    double value = * (double *) arg;

    if (beta != 0.0 && value < real/beta)
        return 1;
    return 0;
}

void func(...)
{
    ...

    double value = 0.5;
    int num_selected, *selected = malloc(n*sizeof(int));
    starneig_GEP_SM_Select(
        n, S, ldS, T, ldT, &predicate, &value, selected, &num_selected);

    ...
}
@endcode

See modules @ref starneig_sm_gep and @ref starneig_dm_gep for further
information. See also examples @ref gep_sm_full_chain.c,
@ref gep_dm_full_chain.c and @ref gep_sm_eigenvectors.c.

@example gep_sm_full_chain.c
@example gep_dm_full_chain.c
@example gep_sm_eigenvectors.c
