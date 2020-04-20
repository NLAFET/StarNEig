///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @internal LICENSE
///
/// Copyright (c) 2019-2020, Umeå Universitet
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
/// 1. Redistributions of source code must retain the above copyright notice,
///    this list of conditions and the following disclaimer.
///
/// 2. Redistributions in binary form must reproduce the above copyright notice,
///    this list of conditions and the following disclaimer in the documentation
///    and/or other materials provided with the distribution.
///
/// 3. Neither the name of the copyright holder nor the names of its
///    contributors may be used to endorse or promote products derived from this
///    software without specific prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/sep_dm.h>
#include <starneig/gep_dm.h>
#include <starneig/blacs_matrix.h>
#include <starneig/blacs_helpers.h>
#include "common.h"
#include "../common/common.h"
#include "../common/node_internal.h"

static int are_compatible(
    starneig_distr_matrix_t A, starneig_distr_matrix_t B,
    starneig_distr_matrix_t Q, starneig_distr_matrix_t Z)
{
    starneig_distr_t distr = starneig_distr_matrix_get_distr(A);
    int bm = starneig_distr_matrix_get_row_blksz(A);
    int bn = starneig_distr_matrix_get_row_blksz(A);

    if (!starneig_distr_is_blacs_compatible(distr))
        return 0;

    if (bm != bn)
        return 0;

    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

#define CHECK_MATRIX(X) \
    if (X != NULL) { \
        starneig_distr_t _distr = starneig_distr_matrix_get_distr(X); \
        int _bm = starneig_distr_matrix_get_row_blksz(X); \
        int _bn = starneig_distr_matrix_get_row_blksz(X); \
        if (_bm != bm || _bn != bn || \
        !starneig_distr_is_compatible_with(_distr, context)) { \
            starneig_blacs_gridexit(context); \
            return 0; \
        } \
    }

    CHECK_MATRIX(B)
    CHECK_MATRIX(Q)
    CHECK_MATRIX(Z)

#undef CHECK_MATRIX

    starneig_blacs_gridexit(context);

    return 1;
}

static int are_suitable(
    starneig_distr_matrix_t A, starneig_distr_matrix_t B,
    starneig_distr_matrix_t Q, starneig_distr_matrix_t Z)
{
#define CHECK_MATRIX(X) \
    if (X != NULL) { \
        int bm = starneig_distr_matrix_get_row_blksz(X); \
        int bn = starneig_distr_matrix_get_row_blksz(X); \
        if (128 < bm || 128 < bn) \
            return 0; \
    }

    CHECK_MATRIX(A)
    CHECK_MATRIX(B)
    CHECK_MATRIX(Q)
    CHECK_MATRIX(Z)

#undef CHECK_MATRIX

    return 1;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Hessenberg(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t Q)
{
    extern void pdgehrd_(int const *, int const *, int const *, double *,
        int const *, int const *, starneig_blacs_descr_t const *, double *,
        double *, int const *, int *);

    extern void pdormhr_(char const *, char const *, int const *, int const *,
        int const *, int const *, double *, int const *, int const *,
        starneig_blacs_descr_t const *, double *, double *, int const *,
        int const *, starneig_blacs_descr_t const *, double *, int const *,
        int *);

    extern void pdlaset_(char const *, int const *, int const *,
        double const *, double const *, double *, int const *, int const *,
        starneig_blacs_descr_t const *);

    if (A == NULL)  return -1;
    if (Q == NULL)  return -2;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    //
    // redistribute if necessary
    //

    int n = starneig_distr_matrix_get_rows(A);

    starneig_distr_t distr;
    starneig_distr_matrix_t _A, _Q;
    if (are_compatible(A, NULL, Q, NULL) && are_suitable(A, NULL, Q, NULL)) {
        distr = starneig_distr_matrix_get_distr(A);
        _A = A;
        _Q = Q;
    }
    else {
        distr = starneig_distr_init();
        _A = starneig_distr_matrix_create(
            n, n, 96, 96, STARNEIG_REAL_DOUBLE, distr);
        _Q = starneig_distr_matrix_create(
            n, n, 96, 96, STARNEIG_REAL_DOUBLE, distr);
        starneig_verbose("Copying A to a temporary buffer.");
        starneig_distr_matrix_copy(A, _A);
        starneig_verbose("Copying Q to a temporary buffer.");
        starneig_distr_matrix_copy(Q, _Q);
    }

    //
    // convert to BLACS format
    //

    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    starneig_blacs_descr_t descr_a, descr_q;
    double *local_a, *local_q;

    starneig_distr_matrix_to_blacs_descr(
        _A, context, &descr_a, (void **)&local_a);
    starneig_distr_matrix_to_blacs_descr(
        _Q, context, &descr_q, (void **)&local_q);

    starneig_wrappers_prepare();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_PARALLEL);

    int info, ia = 1, ja = 1, ilo = 1, ihi = n;

    //
    // allocate workspace
    //

    double *tau = NULL, *work = NULL;

    int lwork = 0;

    {
        int _lwork = -1;
        double dlwork;

        pdgehrd_(&n, &ilo, &ihi, local_a, &ia, &ja, &descr_a, tau, &dlwork,
            &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        pdormhr_("Right", "No transpose", &n, &n, &ilo, &ihi, local_a,
            &ia, &ja, &descr_a, tau, local_q, &ia, &ja, &descr_q, &dlwork,
            &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    tau = malloc(n*sizeof(double));
    work = malloc(lwork*sizeof(double));

    //
    // reduce
    //

    starneig_verbose("Reducing.");

    pdgehrd_(&n, &ilo, &ihi, local_a, &ia, &ja, &descr_a, tau, work,
        &lwork, &info);
    if (info != 0)
        goto cleanup;

    pdormhr_("Right", "No transpose", &n, &n, &ilo, &ihi, local_a,
        &ia, &ja, &descr_a, tau, local_q, &ia, &ja, &descr_q, work,
        &lwork, &info);
    if (info != 0)
        goto cleanup;

    {
        int nm2 = n-2, one = 1, three = 3;
        double dzero = 0.0;
        pdlaset_("Lower", &nm2, &nm2, &dzero, &dzero, local_a, &three, &one,
            &descr_a);
    }

cleanup:

    free(tau);
    free(work);
    starneig_blacs_gridexit(context);

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);
    starneig_wrappers_finish();

    //
    // redistribute if necessary
    //

    if (A != _A) {
        starneig_verbose("Copying A from a temporary buffer.");
        starneig_distr_matrix_copy(_A, A);
        starneig_distr_matrix_destroy(_A);
    }
    if (Q != _Q) {
        starneig_verbose("Copying Q from a temporary buffer.");
        starneig_distr_matrix_copy(_Q, Q);
        starneig_distr_matrix_destroy(_Q);
    }

    return info;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_DM_HessenbergTriangular(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t B,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z)
{
    extern void pdgeqrf_(int const *, int const *, double *, int const *,
        int const *, starneig_blacs_descr_t const *, double *, double *,
        int const *, int *);

    extern void pdormqr_(char const *, char const *, int const *, int const *,
        int const *, double const *, int const *, int const *,
        starneig_blacs_descr_t const *, double const *, double *, int const *,
        int const *, starneig_blacs_descr_t const *, double *, int const *,
        int *);

    extern void pdgeadd_(
        char const *, int const *, int const *, double const *, double const *,
        int const *, int const *, starneig_blacs_descr_t const *,
        double const *, double *, int const *, int const *,
        starneig_blacs_descr_t const *);

    extern void pdlaset_(char const *, int const *, int const *,
        double const *, double const *, double *, int const *, int const *,
        starneig_blacs_descr_t const *);

    extern void pdgghrd_(
        char const *, char const *, int const *, int const *, int const *,
        double *, starneig_blacs_descr_t const *, double *,
        starneig_blacs_descr_t const *, double *,
        starneig_blacs_descr_t const *, double *,
        starneig_blacs_descr_t const *, double *, int const *, int *);

    if (A == NULL)  return -1;
    if (B == NULL)  return -2;
    if (Q == NULL)  return -3;
    if (Z == NULL)  return -4;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    //
    // redistribute if necessary
    //

    int n = starneig_distr_matrix_get_rows(A);

    starneig_distr_t distr;
    starneig_distr_matrix_t _A, _B, _Q, _Z;
    if (are_compatible(A, B, Q, Z) && are_suitable(A, B, Q, Z)) {
        distr = starneig_distr_matrix_get_distr(A);
        _A = A;
        _B = B;
        _Q = Q;
        _Z = Z;
    }
    else {
        distr = starneig_distr_init();
        _A = starneig_distr_matrix_create(
            n, n, 96, 96, STARNEIG_REAL_DOUBLE, distr);
        _B = starneig_distr_matrix_create(
            n, n, 96, 96, STARNEIG_REAL_DOUBLE, distr);
        _Q = starneig_distr_matrix_create(
            n, n, 96, 96, STARNEIG_REAL_DOUBLE, distr);
        _Z = starneig_distr_matrix_create(
            n, n, 96, 96, STARNEIG_REAL_DOUBLE, distr);
        starneig_verbose("Copying A to a temporary buffer.");
        starneig_distr_matrix_copy(A, _A);
        starneig_verbose("Copying B to a temporary buffer.");
        starneig_distr_matrix_copy(B, _B);
        starneig_verbose("Copying Q to a temporary buffer.");
        starneig_distr_matrix_copy(Q, _Q);
        starneig_verbose("Copying Z to a temporary buffer.");
        starneig_distr_matrix_copy(Z, _Z);
    }

    //
    // convert to BLACS format
    //

    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    starneig_blacs_descr_t descr_a, descr_b, descr_q, descr_z;
    double *local_a, *local_b, *local_q, *local_z;

    starneig_distr_matrix_to_blacs_descr(
        _A, context, &descr_a, (void **)&local_a);
    starneig_distr_matrix_to_blacs_descr(
        _B, context, &descr_b, (void **)&local_b);
    starneig_distr_matrix_to_blacs_descr(
        _Q, context, &descr_q, (void **)&local_q);
    starneig_distr_matrix_to_blacs_descr(
        _Z, context, &descr_z, (void **)&local_z);

    int info, ia = 1, ja = 1, ilo = 1, ihi = n;

    starneig_blacs_descr_t descr_w;
    double *local_w;
    starneig_create_blacs_matrix(
        n, n,
        starneig_distr_matrix_get_row_blksz(_A),
        starneig_distr_matrix_get_col_blksz(_A),
        STARNEIG_REAL_DOUBLE, context,
        &descr_w, (void **)&local_w);

    starneig_wrappers_prepare();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_PARALLEL);

    //
    // allocate workspace
    //

    double *tau = NULL, *work = NULL;

    int lwork = 0;
    {
        int _lwork = -1;
        double dlwork;

        pdgeqrf_(&n, &n, local_b, &ia, &ja, &descr_b, tau,
            &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        pdormqr_("L", "T", &n, &n, &n, local_b, &ia, &ja, &descr_b, tau,
            local_a, &ia, &ja, &descr_a, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        pdormqr_("L", "T", &n, &n, &n, local_b, &ia, &ja, &descr_b, tau,
            local_q, &ia, &ja, &descr_q, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        pdgghrd_("V", "V", &n, &ilo, &ihi, local_a, &descr_a, local_b, &descr_b,
            local_q, &descr_q, local_z, &descr_z, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    tau = malloc(n*sizeof(double));
    work = malloc(lwork*sizeof(double));

    //
    // reduce
    //

    starneig_verbose("Reducing.");

    // form B = ~Q * R
    pdgeqrf_(&n, &n, local_b, &ia, &ja, &descr_b, tau,
        work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // A <- ~Q^T * A
    pdormqr_("L", "T", &n, &n, &n, local_b, &ia, &ja, &descr_b, tau,
        local_a, &ia, &ja, &descr_a, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // W <- Q^T
    {
        double zero = 0.0, one = 1.0;
        pdgeadd_("T", &n, &n, &one, local_q, &ia, &ja, &descr_q, &zero, local_w,
            &ia, &ja, &descr_w);
    }

    // W <- ~Q^T * W
    pdormqr_("L", "T", &n, &n, &n, local_b, &ia, &ja, &descr_b, tau,
        local_w, &ia, &ja, &descr_w, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // clean B (B <- R)
    {
        int nm1 = n-1, one = 1, two = 2;
        double dzero = 0.0;
        pdlaset_(
            "Lower", &nm1, &nm1, &dzero, &dzero, local_b, &two, &one, &descr_b);
    }

    // reduce W (A,B) Z^T to Hessenberg-triangular form
    pdgghrd_("V", "V", &n, &ilo, &ihi, local_a, &descr_a, local_b, &descr_b,
        local_w, &descr_w, local_z, &descr_z, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // Q <- W^T
    {
        double zero = 0.0, one = 1.0;
        pdgeadd_("T", &n, &n, &one, local_w, &ia, &ja, &descr_w, &zero, local_q,
            &ia, &ja, &descr_q);
    }

cleanup:

    starneig_destroy_blacs_matrix(&descr_w, (void **)&local_w);

    free(tau);
    free(work);
    starneig_blacs_gridexit(context);

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);
    starneig_wrappers_finish();

    //
    // redistribute if necessary
    //

    if (A != _A) {
        starneig_verbose("Copying A from a temporary buffer.");
        starneig_distr_matrix_copy(_A, A);
        starneig_distr_matrix_destroy(_A);
    }
    if (B != _B) {
        starneig_verbose("Copying B from a temporary buffer.");
        starneig_distr_matrix_copy(_B, B);
        starneig_distr_matrix_destroy(_B);
    }
    if (Q != _Q) {
        starneig_verbose("Copying Q from a temporary buffer.");
        starneig_distr_matrix_copy(_Q, Q);
        starneig_distr_matrix_destroy(_Q);
    }
    if (Z != _Z) {
        starneig_verbose("Copying Z from a temporary buffer.");
        starneig_distr_matrix_copy(_Z, Z);
        starneig_distr_matrix_destroy(_Z);
    }

    return info;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Reduce(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t Q,
    double real[], double imag[],
    int (*predicate)(double real, double imag, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (A == NULL)  return -1;
    if (Q == NULL)  return -2;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int ret = STARNEIG_SUCCESS;
    int *_selected = NULL;

    ret = starneig_SEP_DM_Hessenberg(A, Q);
    if (ret)
        goto cleanup;

    ret = starneig_SEP_DM_Schur(A, Q, real, imag);
    if (ret)
        goto cleanup;

    if (predicate) {
        if (selected == NULL)
            selected = _selected =
                malloc(starneig_distr_matrix_get_rows(A)*sizeof(int));

        ret = starneig_SEP_DM_Select(A, predicate, arg, selected, num_selected);
        if (ret)
            goto cleanup;

        ret = starneig_SEP_DM_ReorderSchur(selected, A, Q, real, imag);
        if (ret)
            goto cleanup;
    }

cleanup:
    free(_selected);
    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_DM_Reduce(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t B,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[],
    int (*predicate)(double real, double imag, double beta, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (A == NULL)  return -1;
    if (B == NULL)  return -2;
    if (Q == NULL)  return -3;
    if (Z == NULL)  return -4;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int ret = STARNEIG_SUCCESS;
    int *_selected = NULL;

    ret = starneig_GEP_DM_HessenbergTriangular(A, B, Q, Z);
    if (ret)
        goto cleanup;

    ret = starneig_GEP_DM_Schur(A, B, Q, Z, real, imag, beta);
    if (ret)
        goto cleanup;

    if (predicate) {
        if (selected == NULL)
            selected = _selected =
                malloc(starneig_distr_matrix_get_rows(A)*sizeof(int));

        ret = starneig_GEP_DM_Select(
            A, B, predicate, arg, selected, num_selected);
        if (ret)
            goto cleanup;

        ret = starneig_GEP_DM_ReorderSchur(
            selected, A, B, Q, Z, real, imag, beta);
        if (ret)
            goto cleanup;
    }

cleanup:
    free(_selected);
    return ret;
}
