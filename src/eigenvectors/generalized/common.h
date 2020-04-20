///
/// @file
///
/// @brief Constants, LAPACK interfaces, macros
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

#ifndef STARNEIG_EIGVEC_STD_COMMON_H_
#define STARNEIG_EIGVEC_STD_COMMON_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "../../common/common.h"

//
// Formatting constants
//

static char *const shorte="%12.4e";
static char *const longi="%12d";
static char *const shorti="%4d";

//
//  Constants needed for calls to BLAS and LAPACK
//

static int const int_zero = 0;
static int const int_one = 1;
static int const int_two = 2;
static int const int_minus_one = -1;

static double const double_zero = 0;
static double const double_one = 1.0;
static double const double_minus_one = -1.0;
static double const smin = 1e-300;

// Wrappers for LAPACK routines which read leading dimensions
// These wrappers cast the leading dimension as int and then call LAPACK.

// Solver for small shifted linear systems
void starneig_eigvec_gen_dlaln2(
    int ltrans, int na, int nw, double smin, double ca, double *a, size_t lda,
    double d1, double d2, double *b, size_t ldb, double wr, double wi,
	double *x, size_t ldx, double *scale, double *xnorm, int *info);

// Copy matrices
void starneig_eigvec_gen_dlacpy(
    char *uplo, int m, int n, double *a, size_t lda, double *b, size_t ldb);

// Compute norms
double starneig_eigvec_gen_dlange(
    char *norm, int m, int n, double *a, size_t lda, double *work);

// Matrix matrix multiplication
void starneig_eigvec_gen_dgemm(
    char *transa, char *transb, int m, int n, int k, double alpha, double *a,
    size_t lda, double* b, size_t ldb, double beta, double* c, size_t ldc);

// Generalised eigenvalues of 2-by-2 matrices
void starneig_eigvec_gen_dlag2(
    double *a, size_t lda, double *b, size_t ldb, double safemin,
    double *scale1, double *scale2, double *wr1, double *wr2, double *wi);

// Generalized eigenvectors
void starneig_eigvec_gen_dtgevc(
    char *side, char *howmany, int *select, int m, double *s, size_t lds,
	double *t, size_t ldt, double *x, size_t ldx, double *y, size_t ldy,
	int n, int *used, double *work, int *info);

//
// BLAS and LAPACK subroutines and functions
//

// Routines which do NOT read a leading dimension

// Linear update of vector
extern void daxpy_(
    int const *, double const *, double const *, int const *, double *,
    int const *);

// Scale vector
extern void dscal_(int const *, double const *, double *, int const *);

// Routines which read a leading dimension

// Solve small shifted equation
extern void dlaln2_(
    int const *, int const *, int const *, double const *, double const *,
	double const *, int const *, double const *, double const *, double const *,
    int const *, double const *, double const *, double const *, int const *,
	double const *, double const *, int const *);

// Copies all or part of a 2-d array to another
extern void dlacpy_(
    char const *, int const *, int const *, double const *, int const *,
    double *, int const *);

// 1-norm, inf-norm, Frobenius-norm, largest absolute value
extern double dlange_(
    char const *, int const *, int const *, double const *, int const *,
    double *);

// Dense matrix times dense matrix
extern void dgemm_(
    char const *, char const *, int const *, int const *, int const *,
    double const *, double const *, int const *, double const *, int const *,
    double const *, double *, int const *);

// Reduction to upper Hessenberg and triangular form
extern void dgghrd_(
    char const *, char const *, int const *, int const *, int const *,
    double const *, int const *, double const *, int const *, double const *,
    int const *, double const *, int const *, int const *);

// Reduction to generalised real Schur form using QZ algorithm
extern void dhgeqz_(
    char const *, char const *, char const *, int const *, int const *,
    int const *, double const *, int const *, double const *, int const *,
	double const *, double const *, double const *, double const *, int const *,
	double const *, int const *, double const *, int const *, int const *);

// Generalised eigenvalues of 2-by-2 matrices
extern void dlag2_(
    double const *, int const *, double const *, int const *, double const *,
	double const *, double const *, double const *, double const *,
    double const *);

// Generalised eigenvectors from real Schur forms
extern void dtgevc_(
    char const *, char const *, int const *, int const *, double const *,
    int const *, double const *, int const *, double const *, int const *,
	double const *, int const *, int const *, int const *, double const *,
	int const *);

#endif // STARNEIG_EIGVEC_STD_COMMON_H_
