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

#ifndef STARNEIG_TESTS_COMMON_CHECKS_H
#define STARNEIG_TESTS_COMMON_CHECKS_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include "pencil.h"

///
/// @brief Compares two integers.
///
/// @param[in] a
///         The first integer.
///
/// @param[in] b
///         The second integer.
///
/// @return A positive number if a > b, negative number otherwise.
///
int int_compare(void const *a, void const *b);

///
/// @brief Computes a mean value over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The mean value.
///
double int_mean(int n, int const *ptr);

///
/// @brief Computes variance over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The variance.
///
double int_var(int n, int const *ptr);

///
/// @brief Computes coefficient of variance over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The coefficient of variance.
///
double int_cv(int n, int const *ptr);

///
/// @brief Compares two floating-point numbers.
///
/// @param[in] a
///         The first floating-point number.
///
/// @param[in] b
///         The second floating-point number.
///
/// @return A positive number if a > b, negative number otherwise.
///
int double_compare(void const *a, void const *b);

///
/// @brief Computes median over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The median value.
///
double double_median(int n, double const *ptr);

///
/// @brief Computes a mean value over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The mean value.
///
double double_mean(int n, double const *ptr);

///
/// @brief Computes variance over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The variance.
///
double double_var(int n, double const *ptr);


///
/// @brief Computes coefficient of variance over array values.
///
/// @param[in] n
///         The array length.
///
/// @param[in] ptr
///         A pointer to the array.
///
/// @return The coefficient of variance.
///
double double_cv(int n, double const *ptr);

///
/// @brief Computes a residual ||Q A Z^T - C||_F / u * ||C||_F.
///
/// @param[in] mat_q
///         Matrix Q.
///
/// @param[in] mat_a
///         Matrix A.
///
/// @param[in] mat_z
///         Matrix Z.
///
/// @param[in] mat_c
///         Matrix C.
///
/// @return The residual as a multiple of the double-precision unit roundoff
/// error.
///
double compute_qazt_c_norm(
    const matrix_t mat_q, const matrix_t mat_a,
    const matrix_t mat_z, const matrix_t mat_c);

///
/// @brief Computes a residual ||Q Q^T - T||_F / u * ||I||_F
///
/// @param[in] mat_q
///         Matrix Q.
///
/// @return The residual as a multiple of the double-precision unit roundoff
/// error.
///
double compute_qqt_norm(const matrix_t mat_q);

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
///         Scaling factor for the largest eigenvalues.
///
/// @param[out] beta2
///         Scaling factor for the smallest eigenvalues.
///
void compute_complex_eigenvalue(
    int ldA, int ldB, double const *A, double const *B,
    double *real1, double *imag1, double *real2, double *imag2,
    double *beta1, double *beta2);

///
/// @brief Extracts the eigenvalues of the matrix pencil (A,B).
///
/// @param[in] A
///         Matrix A.
///
/// @param[in] B
///         Matrix B. If NULL, then it is assumed that B = I.
///
/// @param[out] real
///         The real parts of the eigenvalues.
///
/// @param[out] imag
///         The imaginary parts of the largest eigenvalues.
///
/// @param[out] beta
///         Scaling factors for the largest eigenvalues.
///
void extract_eigenvalues(
    const matrix_t A, const matrix_t B, double *real, double *imag,
    double *beta);

#endif
