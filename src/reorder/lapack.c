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
#include "lapack.h"
#include <stddef.h>

///
/// @brief Applies LAPACK's dtrsen subroutine to a diagonal window in an upper
/// quasi-triangular matrix A and accumulates the corresponding transformations
/// to a matrix Q.
///
/// @param[in] begin - first row that belongs to the window
/// @param[in] end - last row that belongs to the window + 1
/// @param[in] ldQ - leading dimension of the matrix Q
/// @param[in] ldA - leading dimension of the matrix A
/// @param[out] m - dimension of the specified invariant subspace
/// @param[in,out] select - eigenvalue selection array
/// @param[out] Q - pointer to the matrix Q
/// @param[in,out] A - pointer to the matrix A
/// @param[out] tmp - temporary buffer (3*(end-begin) elements)
///
/// @return info field from LAPACK's dtrsen subroutine
///
int starneig_dtrsen(int begin, int end, int ldQ, int ldA,
    int *m, int *select, double *Q, double *A, double *tmp)
{

    // LAPACK DTRSEN subroutine
    extern void dtrsen_(char const *, char const *, int const *, int const *,
        double *, int const *, double *, int const *,
        double *, double *, int *, double *, double *, double *,
        int const *, int*, int const *, int*);

    int n = end-begin, one = 1, iwork, info;
    double sep, s;

    double *wr = tmp;
    double *wi = tmp + n;
    double *work = tmp + 2*n;

    // reorder the window

    dtrsen_("N", Q != NULL ? "V" : "N", select+begin, &n,
        A+(size_t)begin*ldA+begin, &ldA, Q, &ldQ,
        wr, wi, m, &s, &sep, work, &n, &iwork, &one, &info);

    // update the eigenvalue selection array to indicate that dtrsen has
    // reordered the window
    for (int i = 0; i < *m; i++)
        select[begin+i] = 1;
    for (int i = *m; i < n; i++)
        select[begin+i] = 0;

    return info;
}

///
/// @brief Applies LAPACK's dtgsen subroutine to a diagonal window in a matrix
/// pair (A,B) where A is an upper quasi-triangular matrix and B is an upper
/// triangular matrix. The corresponding left and right transformation are
/// accumulated to matrices Q and Z.
///
/// @param[in] begin - first row that belongs to the window
/// @param[in] end - last row that belongs to the window + 1
/// @param[in] ldQ - leading dimension of the matrix Q
/// @param[in] ldZ - leading dimension of the matrix Z
/// @param[in] ldA - leading dimension of the matrix A
/// @param[in] ldB - leading dimension of the matrix B
/// @param[out] m - dimension of the specified invariant subspace
/// @param[in,out] select - eigenvalue selection array
/// @param[out] Q - pointer to the matrix Q
/// @param[out] Z - pointer to the matrix Z
/// @param[in,out] A - pointer to the matrix A
/// @param[in,out] B - pointer to the matrix B
/// @param[out] tmp - temporary buffer (7*(end-begin)+16 elements)
///
/// @return info field from LAPACK's dtrsen subroutine
///
int starneig_dtgsen(
    int begin, int end, int ldQ, int ldZ, int ldA, int ldB, int *m,
    int *select, double *Q, double *Z, double *A, double *B, double *tmp)
{

    // LAPACK DTGSEN subroutine
    extern void dtgsen_(int const *, int const *, int const *,
        int const *, int const *, double *, int const *, double *,
        int const *, double *, double *, double *, double *, int const *,
        double *, int const *, int *, double *, double *, double *, double *,
        int const *, int *, int const *, int *);

    int n = end - begin;
    int lwork = 4*n+16, zero = 0, one = 1, iwork, info;

    double *alphar = tmp;
    double *alphai = tmp + n;
    double *beta = tmp + 2*n;
    double *work = tmp + 3*n;

    // reorder the window
    dtgsen_(&zero, Q != NULL ? &one : &zero, Z != NULL ? &one : &zero,
        select+begin, &n,
        A+(size_t)begin*ldA+begin, &ldA,
        B+(size_t)begin*ldB+begin, &ldB,
        alphar, alphai, beta, Q, &ldQ, Z, &ldZ,
        m, NULL, NULL, NULL, work, &lwork, &iwork, &one, &info);

    // update the eigenvalue selection array to indicate that dtgsen has
    // reordered the window
    for (int i = 0; i < *m; i++)
        select[begin+i] = 1;
    for (int i = *m; i < n; i++)
        select[begin+i] = 0;

    return info;
}
