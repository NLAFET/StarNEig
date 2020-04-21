# Standard eigenvalue problem

The library provides 12 interface functions for the standard case:

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
orthogonal. On exit, \f$H\f$ is overwritten by \f$S\f$ and \f$Q\f$ is
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
functions attempt to reorder the selected eigenvalues to the top left corner of
an updated Schur matrix \f$\hat{S}\f$ by an orthogonal similarity transformation
\f[
    A = Q * ( U * \hat{S} * U^T ) * Q^T.
\f]
On exit, \f$S\f$ is overwritten by \f$\hat{S}\f$ and \f$Q\f$ is overwritten
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
    A = U * S * U^T,
\f]
where \f$S\f$ is upper quasi-triangular with \f$1 \times 1\f$ and
\f$2 \times 2\f$ blocks on the diagonal (Schur matrix) and \f$U\f$ is
orthogonal. Optionally, the interface functions attempt to reorder selected
eigenvalues to the top left corner of the Schur matrix \f$S\f$.

On exit, \f$A\f$ is overwritten by \f$S\f$ and \f$Q\f$ (which is an
orthogonal matrix on entry) is overwritten by
\f[
    Q \gets Q * U.
\f]

Reordering may in rare cases fail. In such cases the output is guaranteed to
be a Schur decomposition and all (if any) selected eigenvalues that are
correctly placed are marked in the selection array on exit. Reordering may
perturb the eigenvalues and the eigenvalues after reordering are returned.

## Eigenvectors

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
