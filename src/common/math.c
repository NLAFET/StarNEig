///
/// @file
///
/// @brief This file contains math functions that are shared among all
/// components of the library.
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
#include "math.h"
#include "common.h"
#include "sanity.h"
#include <stddef.h>
#include <string.h>
#include <cblas.h>

int starneig_largers_factor(int a, int b)
{
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

void starneig_init_local_q(int n, size_t ldA, double *A)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*ldA+j] = 0.0;

    for (int i = 0; i < n; i++)
        A[i*ldA+i] = 1.0;
}

void starneig_small_left_gemm_update(int rbegin, int rend, int cbegin, int cend,
    size_t ldQ, size_t ldA, size_t ldT, double const *Q, double *A, double *T) {

    STARNEIG_SANITY_CHECK_INF(rbegin, rend, cbegin, cend, ldA, A, "A (in)");
    STARNEIG_SANITY_CHECK_INF(0, rend-rbegin, 0, rend-rbegin, ldQ, Q, "Q");

    int m = rend-rbegin;
    int n = cend-cbegin;
    int k = rend-rbegin;

    if (m == 0 || n == 0)
        return;

    starneig_copy_matrix(
        k, n, ldA, ldT, sizeof(double), A+cbegin*ldA+rbegin, T);

    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, Q, ldQ, T, ldT,
        0.0, A+cbegin*ldA+rbegin, ldA);

    STARNEIG_SANITY_CHECK_INF(rbegin, rend, cbegin, cend, ldA, A, "A (out)");
}

void starneig_small_right_gemm_update(
    int rbegin, int rend, int cbegin, int cend,
    size_t ldQ, size_t ldA, size_t ldT, double const *Q, double *A, double *T) {

    STARNEIG_SANITY_CHECK_INF(rbegin, rend, cbegin, cend, ldA, A, "A (in)");
    STARNEIG_SANITY_CHECK_INF(0, cend-cbegin, 0, cend-cbegin, ldQ, Q, "Q");

    int m = rend-rbegin;
    int n = cend-cbegin;
    int k = cend-cbegin;

    if (m == 0 || n == 0)
        return;

    starneig_copy_matrix(
        m, k, ldA, ldT, sizeof(double), A+cbegin*ldA+rbegin, T);

    cblas_dgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, T, ldT, Q, ldQ,
        0.0, A+cbegin*ldA+rbegin, ldA);

    STARNEIG_SANITY_CHECK_INF(rbegin, rend, cbegin, cend, ldA, A, "A (out)");
}

void starneig_small_gemm_updates(
    int begin, int end, int n, size_t ldlQ, size_t ldlZ, size_t ldQ, size_t ldZ,
    size_t ldA, size_t ldB, size_t ldhT, size_t ldvT, double const *lQ,
    double const *lZ, double *Q, double *Z, double *A, double *B,
    double *hT, double *vT)
{
    // apply the local transformation matrices lQ and lZ to Q and Z
    if (Q != NULL)
        starneig_small_right_gemm_update(
            0, n, begin, end, ldlQ, ldQ, ldvT, lQ, Q, vT);
    if (Z != NULL && Z != Q)
        starneig_small_right_gemm_update(
            0, n, begin, end, ldlZ, ldZ, ldvT, lZ, Z, vT);

    // apply the local transformation matrices lQ and lZ to A
    if (A != NULL) {
        starneig_small_right_gemm_update(
            0, begin, begin, end, ldlZ, ldA, ldvT, lZ, A, vT);
        starneig_small_left_gemm_update(
            begin, end, end, n, ldlQ, ldA, ldhT, lQ, A, hT);
    }

    // apply the local transformation matrices lQ and lZ to B
    if (B != NULL) {
        starneig_small_right_gemm_update(
            0, begin, begin, end, ldlZ, ldB, ldvT, lZ, B, vT);
        starneig_small_left_gemm_update(
            begin, end, end, n, ldlQ, ldB, ldhT, lQ, B, hT);
    }
}

void starneig_compute_complex_eigenvalue(
    int ldA, int ldB, double const *A, double const *B,
    double *real1, double *imag1, double *real2, double *imag2,
    double *beta1, double *beta2)
{
    if (B != NULL) {
        extern void dlag2_(double const *, int const *, double const *,
            int const *, double const *, double*, double *, double *, double *,
            double *);

        extern double dlamch_(char const *);
        const double safmin = dlamch_("S");

        double _real1, _real2, _beta1, _beta2, wi;
        dlag2_(
            A, &ldA, B, &ldB, &safmin, &_beta1, &_beta2, &_real1, &_real2, &wi);

        if (beta1) {
            *real1 = _real1;
            *real2 = _real2;
            *imag1 = wi;
            *imag2 = -wi;
            *beta1 = _beta1;
            *beta2 = _beta2;
        }
        else {
            *real1 = _real1/_beta1;
            *real2 = _real2/_beta2;
            *imag1 = wi/_beta1;
            *imag2 = -wi/_beta2;
        }
    }
    else {
        extern void dlanv2_(
            double *, double *, double *, double *, double *,
            double *, double *, double *, double *, double *);

        double a[] = { A[0], A[1], A[ldA], A[ldA+1] };
        double cs, ss;

        dlanv2_(
            &a[0], &a[2], &a[1], &a[3], real1, imag1, real2, imag2, &cs, &ss);

        if (beta1) {
            *beta1 = 1.0;
            *beta2 = 1.0;
        }
    }
}
