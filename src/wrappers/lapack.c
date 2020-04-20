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
#include "common.h"
#include "../common/common.h"
#include "../common/node_internal.h"
#include <starneig/gep_sm.h>
#include <cblas.h>
#include <stdlib.h>

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_HessenbergTriangular(
    int n,
    double A[], int ldA,
    double B[], int ldB,
    double Q[], int ldQ,
    double Z[], int ldZ)
{
    extern void dgeqrf_(int const *, int const *, double *, int const *,
        double *, double *, int const *, int *);

    extern void dormqr_(char const *, char const *, int const *, int const *,
        int const *, double const *, int const *, double const *, double *,
        int const *, double*, const int *, int *);

    extern void dgghd3_(char const *, char const *, int const *, int const *,
        int const *, double *, int const *, double *, int const *, double *,
        int const *, double *, int const *, double *, int const *, int *);

    if (n < 1)          return -1;
    if (A == NULL)      return -2;
    if (ldA < n)        return -3;
    if (B == NULL)      return -4;
    if (ldB < n)        return -5;
    if (Q == NULL)      return -6;
    if (ldQ < n)        return -7;
    if (Z == NULL)      return -8;
    if (ldZ < n)        return -9;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_wrappers_prepare();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_PARALLEL);

    double *tau = NULL, *work = NULL;
    int info, ilo = 1, ihi = n;

    //
    // allocate workspace
    //

    int lwork = 0;

    {
        int _lwork = -1;
        double dlwork;

        dgeqrf_(&n, &n, B, &ldB, tau, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        dormqr_("L", "T", &n, &n, &n,
            B, &ldB, tau, A, &ldA, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        dormqr_("R", "N", &n, &n, &n,
            B, &ldB, tau, Q, &ldQ, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        dgghd3_("V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, Q, &ldQ, Z, &ldZ, &dlwork, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        lwork = MAX(lwork, dlwork);
    }

    tau = malloc(n*sizeof(double));
    work = malloc(lwork*sizeof(double));

    //
    // reduce
    //

    // form B = ~Q * R
    dgeqrf_(&n, &n, B, &ldB, tau, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // A <- ~Q^T * A
    dormqr_("L", "T", &n, &n, &n, B, &ldB, tau, A, &ldA, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // Q <- Q * ~Q
    dormqr_("R", "N", &n, &n, &n, B, &ldB, tau, Q, &ldQ, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

    // clean B (B <- R)
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            B[(size_t)i*ldB+j] = 0.0;

    // reduce (A,B) to Hessenberg-triangular form
    dgghd3_("V", "V", &n, &ilo, &ihi,
        A, &ldA, B, &ldB, Q, &ldQ, Z, &ldZ, work, &lwork, &info);
    if (info != 0)
        goto cleanup;

cleanup:

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);
    starneig_wrappers_finish();

    free(tau);
    free(work);

    return info;
}
