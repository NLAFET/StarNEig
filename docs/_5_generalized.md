# Generalized eigenvalue problem

The library provides 12 interface functions for the generalized case:

## Hessenberg-triangular reduction

Given a general matrix \f$(A,B)\f$, the starneig_GEP_SM_HessenbergTriangular()
and starneig_GEP_DM_HessenbergTriangular() interface functions compute a
Hessenberg-triangular decomposition
\f[
    (A,B) = U_1 * (H,T) * U_2^T,
\f]
where \f$H\f$ is upper Hessenberg, \f$T\f$ is upper triangular, and
\f$U_1\f$ and \f$U_2\f$ are orthogonal. On exit, \f$A\f$ is overwritten by
\f$H\f$, \f$B\f$ is overwritten by \f$T\f$, and \f$Q\f$ and \f$Z\f$ (which are
orthogonal matrices on entry) are overwritten by
\f[
    Q \gets Q * U_1 \text{ and } Z \gets Z * U_2.
\f]

## Generalized Schur reduction

Given a Hessenberg-triangular decomposition
\f[
    (A,B) = Q * (H,T) * Z^T
\f]
of a general matrix pencil \f$(A,B)\f$, the starneig_GEP_SM_Schur() and
starneig_GEP_DM_Schur() interface functions function compute a generalized
Schur decomposition
\f[
    (A,B) = Q * ( U_1 * (S,\hat{T}) * U_2^T ) * Z^T,
\f]
where \f$S\f$ is upper quasi-triangular with \f$1 \times 1\f$ and
\f$2 \times 2\f$ blocks on the diagonal, \f$\hat{T}\f$ is a upper triangular
matrix, and \f$U_1\f$ and \f$U_2\f$ are orthogonal.

On exit, \f$H\f$ is overwritten by \f$S\f$, \f$T\f$ is overwritten by
\f$\hat{T}\f$, and \f$Q\f$ and \f$Z\f$ are overwritten by
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
of a general matrix pencil \f$(A,B)\f$ and a selection of generalized
eigenvalues, the starneig_GEP_SM_ReorderSchur() and
starneig_GEP_DM_ReorderSchur() interface functions attempt to reorder the
selected generalized eigenvalues to the top left corner of an updated
generalized Schur decomposition by an orthogonal similarity transformation
\f[
    (A,B) = Q * ( U_1 * (\hat{S},\hat{T}) * U_2^T ) * Z^T.
\f]

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

Given a general matrix pencil \f$(A,B)\f$, the starneig_GEP_SM_Reduce() and
starneig_GEP_DM_Reduce() interface functions compute a (reordered) generalized
Schur decomposition
\f[
    (A,B) = U_1 * (S,T) * U_2^T,
\f]
where \f$S\f$ is upper quasi-triangular with \f$1 \times 1\f$ and
\f$2 \times 2\f$ blocks on the diagonal, \f$T\f$ is a upper triangular
matrix, and \f$U_1\f$ and \f$U_2\f$ are orthogonal. Optionally, the interface
functions attempt to reorder selected generalized eigenvalues to the top left
corner of the generalized Schur decomposition.

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

Given a generalized Schur decomposition
\f[
    (A,B) = Q * (S,T) * Z^T
\f]
of a general matrix pencil \f$(A,B)\f$ and a selection of generalized
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

Given a Schur-triangular matrix pencil \f$(S,T)\f$ and a predicate function,
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
