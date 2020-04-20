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

#ifndef STARNEIG_EIGVEC_GEN_GEIG_H_
#define STARNEIG_EIGVEC_GEN_GEIG_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Auxiliary routines which finds all information related to tilings
///
/// @param[in] m the dimension of the problem
/// @param[in] mb number of rows per block of Y, target value
/// @param[in] nb number of columns per block of Y, target value
/// @param[in] s array containing matrix S
/// @param[in] lds leading dimension of array s
/// @param[out] ptr1 pointer to left looking array
/// @param[out] ptr2 pointer to map of selected eigenvalues
/// @param[out] ptr3 pointer to practical row tiling
/// @param[out] ptr4 pointer to induced column tiling
/// @param[out] ptr5 pointer to practical column tiling
///
void starneig_eigvec_gen_find_tilings(
    int m, int mb, int nb, double *s, size_t lds, int *select, int **ptr1,
    int **ptr2, int **ptr3, int **ptr4, int **ptr5, int *num1, int *num2);

///
/// @brief Mini-block column norms of a matrix
///
/// @param[in] m number of rows of matrix X
/// @param[in] n number of columns of matrix Y
/// @param[in] alphai array of real numbers which dictate column structure
/// @param[in] x array containing the matrix X
/// @param[in] ldx leading dimension of the array x
/// @param[out] xnorm array of infinity norms of the mini-block columns of X
///
void starneig_eigvec_gen_mini_block_column_norms(
    int m, int n, double *alphai, double *x, size_t ldx, double *xnorm);

///
/// @brief Computes selected generalised eigenvalues from pencil (S,T) in real
/// Schur form
///
/// The jth eigenvalue is lambda[j] = (alphar[j] + i*alphai[j])/beta[j].
/// Complex conjugate eigenvalues are stored next to each other.
/// If lambda[j] and lambda[j+1] are a pair of complex conjugate eigenvalues,
/// then beta[j+1]=beta[j], alphai[j+1]=-alphai[j] and alphar[j+1]=alphar[j].
/// If alphai[j]=0 then lambda[j] is real. In this case is beta[j]=0 possible.
/// This corresponds to an infinite eigenvalue.
///
/// @param[in] m  dimension of matrices S, T
/// @param[in] s  array containing matrix S
/// @param[in] lds  leading dimension of array s
/// @param[in] t  array containing matrix T
/// @param[in] ldt  leading dimension of array t
/// @param[in] select  LAPACK style selection array of length m
/// @param[out]  alphar array of length m
/// @param[out]  alphai array of length m
/// @param[out]  beta array of length m
///
int starneig_eigvec_gen_generalised_eigenvalues(
    int m, double *s, size_t lds, double *t, size_t ldt, int *select,
	double *alphar, double *alphai, double *beta);

///
/// @brief Performs the multishift linear update Y:=Y-(S*X*D-T*X*B)
///
/// @param[in] m  number of rows of S, T, Y.
/// @param[in] n  number of shifts and number of columns of Y.
/// @param[in] k  number of columns of S and T, number of rows of X.
/// @param[in] s  array containing the matrix S.
/// @param[in] lds  leading dimension of s.
/// @param[in] t  array containing the matrix T.
/// @param[in] ldt  leading dimension of t.
/// @param[in] alphar  array of length at least n.
/// @param[in] alphai  array of length at least n.
/// @param[in] beta  array of length at least n.
/// @param[in] x  array containing the matrix X.
/// @param[in] ldx  leading dimension of array x.
/// @param[in,out] y  array containing matrix Y.
///         On entry, the original value of Y.
///         On exit, overwritten by the updated value of Y.
/// @param[in] ldy leading dimension of array y.
///
int starneig_eigvec_gen_multi_shift_update(
    int m, int n, int k, double *s, size_t lds, double *t, size_t ldt,
	double *alphar, double *alphai, double *beta, double *x, size_t ldx,
	double *y, size_t ldy);

// Infinity norm relative residual for each mini-block column
double starneig_eigvec_gen_relative_residual(
    int m, int n, double *s, size_t lds, double *t, size_t ldt,
	double *alphar, double *alphai, double *beta, double *x, size_t ldx,
	double *f, size_t ldf, double *rres);

#endif // STARNEIG_EIGVEC_GEN_GEIG_H_
