# Standard eigenvalue problem

Given a matrix \f$A \in \mathbb R^{n \times n}\f$, the *standard eigenvalue
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

The library provides 12 interface functions for the standard case.

## Hessenberg reduction

Given a general matrix \f$A\f$, the starneig_SEP_SM_Hessenberg() and
starneig_SEP_DM_Hessenberg() interface functions compute a Hessenberg
decomposition
\f[
    A = U * H * U^T,
\f]
where \f$H\f$ is upper Hessenberg and \f$U\f$ is orthogonal. On exit, \f$A\f$ is
overwritten by \f$H\f$ and \f$Q\f$ (which is an orthogonal matrix on entry) is
overwritten by
\f[
    Q \gets Q * U.
\f]

This is done in order to greatly accelerate the subsequent computation of a
Schur decomposition since when working on \f$H\f$ of size \f$n \times n\f$,
the amount of work in each iteration of the QR algorithm is reduced from
\f$\mathcal{O}(n^3)\f$ to \f$\mathcal{O}(n^2)\f$ flops.

## Schur reduction

Given a Hessenberg decomposition
\f[
    A = Q * H * Q^T,
\f]
of a general matrix \f$A\f$, the starneig_SEP_SM_Schur() and
starneig_SEP_DM_Schur() interface functions compute a Schur decomposition
\f[
    A = Q * ( U * S * U^T ) * Q^T
\f]
where \f$S\f$ is upper quasi-triangular with \f$1 \times 1\f$ and
\f$2 \times 2\f$ blocks on the diagonal (Schur matrix) and \f$U\f$ is
orthogonal. The \f$1 \times 1\f$ blocks correspond to the real eigenvalues and
each \f$2 \times 2\f$ block corresponds to a pair of complex conjugate
eigenvalues. On exit, \f$H\f$ is overwritten by \f$S\f$ and \f$Q\f$ is
overwritten by
\f[
    Q \gets Q * U.
\f]

## Eigenvalue reordering

Given a Schur decomposition
\f[
    A = Q * S * Q^T
\f]
of a general matrix \f$A\f$ and a selection of eigenvalues, the
starneig_SEP_SM_ReorderSchur() and starneig_SEP_DM_ReorderSchur() interface
functions attempt to compute an updated Schur decomposition
\f[
    A = Q * \left( U *
        \begin{bmatrix}
          \hat S_{11} & \hat S_{12} \\
          0 & \hat S_{22}
        \end{bmatrix}
    * U^T \right) * Q^T,
\f]
where the selected eigenvalues appear in \f$\hat S_{11}\f$. On exit, \f$S\f$ is overwritten by \f$\hat{S}\f$ and \f$Q\f$ is overwritten
by
\f[
    Q \gets Q * U.
\f]

Reordering may in rare cases fail. In such cases the output is guaranteed to
be a Schur decomposition and all (if any) selected eigenvalues that are
correctly placed are marked in the selection array on exit. Reordering may
perturb the eigenvalues and the eigenvalues after reordering are returned.

## Combined reduction to Schur form and eigenvalue reordering

Given a general matrix \f$A\f$, the starneig_SEP_SM_Reduce() and
starneig_SEP_DM_Reduce() interface functions compute a (reordered) Schur
decomposition
\f[
    A = U * S * U^T.
\f]
Optionally, the interface functions attempt to reorder selected eigenvalues to
the top left corner of the Schur matrix \f$S\f$. On exit, \f$A\f$ is overwritten
by \f$S\f$ and \f$Q\f$ (which is an orthogonal matrix on entry) is overwritten
by
\f[
    Q \gets Q * U.
\f]

Reordering may in rare cases fail. In such cases the output is guaranteed to
be a Schur decomposition and all (if any) selected eigenvalues that are
correctly placed are marked in the selection array on exit. Reordering may
perturb the eigenvalues and the eigenvalues after reordering are returned.

## Eigenvectors

Given a subset consisting of \f$m \leq n\f$ of the eigenvalues \f$\lambda_{i}\f$
for \f$i = 1, 2, \ldots, m\f$ and a Schur decomposition \f$A = Q S Q^{H}\f$, we
can compute for each \f$\lambda_{i}\f$ an *eigenvector* \f$v_{i} \neq 0\f$ such
that \f$A v_{i} = \lambda_{i} v_{i}\f$ by first computing an eigenvector
\f$w_{i}\f$ of \f$S\f$ and then transform it back to the original basis by
pre-multiplication with \f$Q\f$.

Given a Schur decomposition
\f[
    A = Q * S * Q^T
\f]
of a general matrix \f$A\f$ and a selection of eigenvalues,
the starneig_SEP_SM_Eigenvectors() and starneig_SEP_DM_Eigenvectors() interface
functions compute and return an eigenvector for each of the selected
eigenvalues.

The eigenvectors are stored as columns in the output matrix \f$X\f$ in the same
order as their corresponding eigenvalues appear in the selection array. A real
eigenvector is stored as a single column. The real and imaginary parts of a
complex eigenvector are stored as consecutive columns.

For a selected pair of complex conjugate eigenvalues, an eigenvector is
computed only for the eigenvalue with positive imaginary part. Thus, every
selected eigenvalue contributes one column to the output matrix and thus the
number of selected eigenvalues is equal to the number of columns of \f$X\f$.

## Eigenvalue selection helper

Given a Schur matrix and a predicate function, the starneig_SEP_SM_Select() and
starneig_SEP_DM_Select() interface functions conveniently generate a correct
selection array and count the number of selected eigenvalues. The count is
useful when allocating storage for the eigenvector matrix computed by the
starneig_SEP_SM_Eigenvectors() and starneig_SEP_DM_Eigenvectors() interface
functions.

@code{.c}
// a predicate function that selects all eigenvalues that have a real
// part that is larger than a given value
static int predicate(double real, double imag, void *arg)
{
    double value = * (double *) arg;

    if (value < real)
        return 1;
    return 0;
}

void func(...)
{
    double *S; int ldS;

    ...

    double value = 0.5;
    int num_selected, *selected = malloc(n*sizeof(int));
    starneig_SEP_SM_Select(
        n, S, ldS, &predicate, &value, selected, &num_selected);

    ...
}
@endcode

See modules @ref starneig_sm_sep and @ref starneig_dm_sep for further
information. See also examples @ref sep_sm_full_chain.c,
@ref sep_dm_full_chain.c and @ref sep_sm_eigenvectors.c.

@example sep_sm_full_chain.c
@example sep_dm_full_chain.c
@example sep_sm_eigenvectors.c
