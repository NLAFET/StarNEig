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

#ifndef STARNEIG_COMMON_MATH_H
#define STARNEIG_COMMON_MATH_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// @brief A wrapper function for dlamch LAPACK function
///
/// @param[in] param
///         Parameter for the dlamch functions
///
/// @return dlamch function's return value.
///
static inline double dlamch(char const *param)
{
    // LAPACK: DLAMCH determines double precision machine parameters.
    extern double dlamch_(char const *);
    return dlamch_(param);
}

///
/// @brief Computes square of a floating-point number.
///
/// @param[in] x
///         The argument.
///
/// @return square of the argument
///
static inline double squ(double x)
{
    return x*x;
}

///
/// @brief Computes the greatest common divisor of two numbers.
///
/// @param[in] a
///         First number.
///
/// @param[in] b
///         Second number.
///
/// @return Greatest common divisor.
///
int starneig_largers_factor(int a, int b);

///
/// @brief Initializes a matrix to identity.
///
/// @param[in]  n
///         Matrix dimension.
///
/// @param[in] ldA
///         First dimension of the matrix.
///
/// @param[out] A
///         Pointer to the matrix.
///
void starneig_init_local_q(int n, size_t ldA, double *A);

///
/// @brief Performs a localized left-hand side update on a matrix A using using
/// transformation matrix Q.
///
/// @param[in] rbegin
///         First row that belongs to the update windows.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last row that belongs to the update window + 1.
///
/// @param[in] ldQ
///         Leading dimension of the matrix Q.
///
/// @param[in] ldA
///         Leading dimension of the matrix A.
///
/// @param[in] ldT
///         Scratch buffer leading dimension.
///
/// @param[in] Q
///         Pointer to the matrix Q.
///
/// @param[in,out] A
///         Pointer to the matrix A.
///
/// @param[out] T
///         Pointer to a scratch buffer.
///
void starneig_small_left_gemm_update(int rbegin, int rend, int cbegin, int cend,
    size_t ldQ, size_t ldA, size_t ldT, double const *Q, double *A, double *T);

///
/// @brief Performs a localized right-hand side update on a matrix A using using
/// transformation matrix Q.
///
/// @param[in] rbegin
///         First row that belongs to the update windows.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last row that belongs to the update window + 1.
///
/// @param[in] ldQ
///         Leading dimension of the matrix Q.
///
/// @param[in] ldA
///         Leading dimension of the matrix A.
///
/// @param[in] ldT
///         Scratch buffer leading dimension.
///
/// @param[in] Q
///         Pointer to the matrix Q.
///
/// @param[in,out] A
///         Pointer to the matrix A.
///
/// @param[out] T
///         Pointer to a scratch buffer.
///
void starneig_small_right_gemm_update(
    int rbegin, int rend, int cbegin, int cend,
    size_t ldQ, size_t ldA, size_t ldT, double const *Q, double *A, double *T);

///
/// @brief Performs off-diagonal updates that are associated with a given
/// diagonal window.
///
/// @param[in] begin
///         First row/column that belongs to the diagonal windows.
///
/// @param[in] end
///         Last row/column that belongs to the diagonal window + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldlQ
///         The leading dimension of the left-hand side update matrix.
///
/// @param[in] ldlZ
///         The leading dimension of the right-hand side update matrix.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldA
///         The leading dimension of the matrix Q.
///
/// @param[in] ldB
///         The leading dimension of the matrix Q.
///
/// @param[in] ldhT
///         The leading dimension of the horizontal workspace (>= end-begin).
///
/// @param[in] ldvT
///         The leading dimension of the vertical workspace (>= n)
///
/// @param[in] lQ
///         The left-hand side transformation matrix (end-begin rows/columns).
///
/// @param[in] lZ
///         The right-hand side transformation matrix (end-begin rows/columns).
///
/// @param[in,out] Q
///         Pointer to the matrix Q.
///
/// @param[in,out] Z
///         Pointer to the matrix Z.
///
/// @param[in,out] A
///         Pointer to the matrix A.
///
/// @param[in,out] B
///         Pointer to the matrix B.
///
/// @param[in,out] hT
///         The horizontal workspace (n columns).
///
/// @param[in,out] vT
///         The vertical workspace (end-begin columns).
///
void starneig_small_gemm_updates(
    int begin, int end, int n, size_t ldlQ, size_t ldlZ, size_t ldQ, size_t ldZ,
    size_t ldA, size_t ldB, size_t ldhT, size_t ldvT, double const *lQ,
    double const *lZ, double *Q, double *Z, double *A, double *B,
    double *hT, double *vT);

///
/// @brief Computes the generalized eigenvalues of a 2-by-2 matrix pencil (A,B).
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         Matrix A.
///
/// @param[in] B
///         Matrix B. If NULL, then it is assumed that B = I.
///
/// @param[out] real1
///         The real part of the largest eigenvalue.
///
/// @param[out] imag1
///         The imaginary part of the largest eigenvalue.
///
/// @param[out] real2
///         The real part of the smallest eigenvalue.
///
/// @param[out] imag2
///         The imaginary part of the smallest eigenvalue.
///
/// @param[out] beta1
///         Scaling factor for the largest eigenvalues. If NULL, then the
///         scaling is applied to real1 and imag1.
///
/// @param[out] beta2
///         Scaling factor for the smallest eigenvalues. If NULL, then the
///         scaling is applied to real2 and imag2.
///
void starneig_compute_complex_eigenvalue(
    int ldA, int ldB, double const *A, double const *B,
    double *real1, double *imag1, double *real2, double *imag2,
    double *beta1, double *beta2);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
