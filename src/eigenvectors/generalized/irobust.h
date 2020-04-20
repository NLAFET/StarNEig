///
/// @file
///
/// @brief Header file
///
/// @author Carl Christian K. Mikkelsen (spock@cs.umu.se), Umeå University
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

#ifndef STARNEIG_EIGVEC_GEN_IROBUST_H_
#define STARNEIG_EIGVEC_GEN_IROBUST_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Computes the scaling necessary to prevent overflow in a scalar
/// division y = b/t
///
/// @param[in] b real number bounded by Omega
/// @param[in] t nonzero real number bounded by Omega
///
/// @return integer k s.t. (alpha*b)/t is bounded by Omega where alpha=2^k.
///
int starneig_eigvec_gen_int_protect_division(double b, double t);

///
/// @brief Computes the scaling necessary to prevent overflow in a linear
/// update Y:=Y-T*X
///
/// @param[in] t upper bound of the infinity norm of the matrix T, t <= Omega
/// @param[in] x upper bound of the infinity norm of the matrix X, x <= Omega
/// @param[in] y upper bound of the infinity norm of the matrix Y, y <= Omega
///
/// @return integer k such that the calculation of Y:=(alpha*Y) - T*(alpha*X)
/// cannot exceed Omega where alpha = 2^k.
///
int starneig_eigvec_gen_int_protect_update(double t, double x, double y);

///
/// @brief Robust computation of X:=alpha*X. If necessary, the columns are
/// scaled to prevent components from exceeding Omega.
///
/// @param[in] alpha real scalar
/// @param[in] m number of rows of matrix X
/// @param[in] n number of columns of matrix X
/// @param[in,out] x array containing matrix X
/// @param[in] ldx leading dimension of array x
/// @param[in, out] xscal array.
///         On entry, xscal[j] is the original scaling factor of the jth column.
///         On exit, xscal[j] is the updated scaling factor of the jth column.
/// @param[in, out] xnorm array.
///         On entry, xnorm[j] bounds the infinity norm of the jth column of X.
///         On exit, xnorm[j] bounds the infinity norm of the jth column of X.
///
void starneig_eigvec_gen_int_robust_scaling(
    double alpha, int m, int n, double *x, size_t ldx, int *xscal,
    double *xnorm);

///
/// @brief Robust linear update Y:= alpha*A*X + beta*Y using power of 2 scaling
/// factors
///
/// @param[in] m number of rows of matrix Y
/// @param[in] n number of columns of matrix Y
/// @param[in] k number of columns of matrix A/rows of matrix X
///
/// @param[in] alpha real scalar
/// @param[in] a array containing matrix A
/// @param[in] lda leading dimension of array a
/// @param[in] anorm infinity norm of matrix A
///
/// @param[in] x array containing matrix X
/// @param[in] ldx leading dimension of array x
/// @param[in] xscal array of scaling factors for columns of matrix X
/// @param[in] xnorm array of infinity norms of columns of matrix X
///
/// @param[in] beta real scalar
/// @param[in,out] y array containing matrix Y
/// @param[in] ldy leading dimension of array Y
/// @param[in,out] yscal array.
///         On entry, yscal[j] is the original scaling factor the jth column.
///         On exit, yscal[j] is the updated scaling factor for the jth column.
/// @param[in] ynorm array,
///         On entry, ynorm[j] bounds the infinity norm of the jth column.
///         On exit, unchanged and therefore unrelated to the new matrix Y.
///
void starneig_eigvec_gen_int_robust_update(
    int m, int n, int k, double alpha, double *a, size_t lda, double anorm,
	double *x, size_t ldx, int *xscal, double *xnorm, double beta, double *y,
    size_t ldy, int *yscal, double *ynorm);

#endif // STARNEIG_EIGVEC_GEN_IROBUST_H_
