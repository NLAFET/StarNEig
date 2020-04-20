///
/// @file This file contains auxiliary subroutines that are used to check the
/// outcome of the computation.
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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "checks.h"
#include "init.h"
#include "local_pencil.h"
#include "crawler.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

///
/// @brief An argument structure for eigenvalue_crawler crawler function.
///
struct eigenvalue_arg {
    double *real;
    double *imag;
    double *beta;
};

///
/// @brief A crawler that computes all eigenvalues of a Schur matrix.
///
static int eigenvalue_crawler(
    int offset, int size, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *real = ((struct eigenvalue_arg *)arg)->real;
    double *imag = ((struct eigenvalue_arg *)arg)->imag;
    double *beta = ((struct eigenvalue_arg *)arg)->beta;

    double const *A = ptrs[0];
    size_t ldA = lds[0];

    double const *B = NULL;
    size_t ldB = 0;
    if (1 < count) {
        B = ptrs[1];
        ldB = lds[1];
    }

    int i = 0;
    int _size = offset+size < n ? size-1 : size;
    while (i < _size) {
        if (i+1 < size && A[i*ldA+i+1] != 0.0) {
            compute_complex_eigenvalue(ldA, ldB,
                &A[i*ldA+i], B != NULL ? &B[i*ldB+i]: NULL,
                &real[offset+i],   &imag[offset+i],
                &real[offset+i+1], &imag[offset+i+1],
                &beta[offset+i],   &beta[offset+i+1]);
            i += 2;
        }
        else {
            real[offset+i] = A[i*ldA+i];
            imag[offset+i] = 0.0;
            beta[offset+i] = (B != NULL) ? B[i*ldB+i] : 1.0;
            i++;
        }
    }

    return i;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int int_compare(void const *a, void const *b)
{
    return (*(int*)a - *(int*)b);
}

double int_mean(int n, int const *ptr)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += ptr[i];
    return 1.0f * sum / n;
}

double int_var(int n, int const *ptr)
{
    double mean = int_mean(n, ptr);
    double var = 0.0;
    for (int i = 0; i < n; i++)
        var += squ(mean-ptr[i]);
    return var / n;
}

double int_cv(int n, int const *ptr)
{
    double var= int_var(n, ptr);
    if (0 < var)
        return sqrt(var) / int_mean(n, ptr);
    return 0.0;
}

int double_compare(void const *a, void const *b)
{
    return (*(double*)a - *(double*)b) < 0.0 ? -1 : 1;
}

double double_median(int n, double const *ptr)
{
    double *tmp = malloc(n*sizeof(double));
    memcpy(tmp, ptr, n*sizeof(double));
    qsort(tmp, n, sizeof(double), &double_compare);

    int median;
    if (n % 2 == 0)
        median = (tmp[n/2-1] + tmp[n/2]) / 2.0;
    else
        median = tmp[n/2];

    free(tmp);
    return median;
}

double double_mean(int n, double const *ptr)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += ptr[i];
    return sum / n;
}

double double_var(int n, double const *ptr)
{
    double mean = double_mean(n, ptr);
    double var = 0.0;
    for (int i = 0; i < n; i++)
        var += squ(mean-ptr[i]);
    return var / n;
}

double double_cv(int n, double const *ptr)
{
    double var = double_var(n, ptr);
    if (0 < var)
        return sqrt(var) / double_mean(n, ptr);
    return 0.0;
}

double compute_qazt_c_norm(
    const matrix_t mat_q, const matrix_t mat_a,
    const matrix_t mat_z, const matrix_t mat_c)
{
    // B <- Q A Z^T - C
    matrix_t tmp = NULL, mat_b = copy_matrix_descr(mat_c);
    mul_C_AB("N", "N", 1.0, mat_q, mat_a, 0.0, &tmp);
    mul_C_AB("N", "T", 1.0, tmp, mat_z, -1.0, &mat_b);
    free_matrix_descr(tmp);

    double res = ((long long)1<<52) * norm_C(mat_b)/norm_C(mat_c);
    free_matrix_descr(mat_b);

    return res;
}

double compute_qqt_norm(const matrix_t mat_q)
{
    int n = GENERIC_MATRIX_N(mat_q);

    matrix_t tmp = copy_matrix_descr(mat_q);
    init_identity(tmp);
    mul_C_AB("N", "T", 1.0, mat_q, mat_q, -1.0, &tmp);

    double res = ((long long)1<<52) * norm_C(tmp)/sqrt(n);
    free_matrix_descr(tmp);

    return res;
}

void compute_complex_eigenvalue(
    int ldA, int ldB, double const *A, double const *B,
    double *real1, double *imag1, double *real2, double *imag2,
    double *beta1, double *beta2)
{
    if (B != NULL) {
        extern void dlag2_(double const *, int const *, double const *,
            int const *, double const *, double const *, double *,
            double *, double *, double *);

        extern double dlamch_(char const *);
        const double safmin = dlamch_("S");

        double wi;
        dlag2_(A, &ldA, B, &ldB, &safmin, beta1, beta2, real1, real2, &wi);
        *imag1 = wi;
        *imag2 = -wi;
    }
    else {
        extern void dlanv2_(
            double *, double *, double *, double *, double *,
            double *, double *, double *, double *, double *);

        double a[] = { A[0], A[1], A[ldA], A[ldA+1] };
        double cs, ss;

        dlanv2_(
            &a[0], &a[2], &a[1], &a[3], real1, imag1, real2, imag2, &cs, &ss);

        if (beta1)
            *beta1 = 1.0;
        if (beta2)
            *beta2 = 1.0;
    }
}

void extract_eigenvalues(
    matrix_t A, matrix_t B, double *real, double *imag, double *beta)
{
    struct eigenvalue_arg arg = {
        .real = real,
        .imag = imag,
        .beta = beta
    };

    crawl_matrices(CRAWLER_R, CRAWLER_DIAG_WINDOW,
        &eigenvalue_crawler, &arg, 0, A, B, NULL);

#ifdef STARNEIG_ENABLE_MPI
    if (A->type == STARNEIG_MATRIX || A->type == BLACS_MATRIX) {
        int n = GENERIC_MATRIX_N(A);
        MPI_Bcast(real, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(imag, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(beta, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
#endif
}
