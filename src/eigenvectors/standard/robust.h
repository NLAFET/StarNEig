///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
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

#ifndef STARNEIG_EIGENVECTORS_STD_ROBUST_H
#define STARNEIG_EIGENVECTORS_STD_ROBUST_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "typedefs.h"


///
/// @brief Initializes all scaling factors with the neutral element.
///
/// @param[in] n
///         Number of entries in alpha. n >= 0.
///
/// @param[out] alpha
///         Vector of length n. On exit, all entries equal the neutral element.
///
void starneig_eigvec_std_init_scaling_factor(int n, scaling_t *alpha);


///
/// @brief Determines the smallest scaling factor column-wise.
///
/// @param[in] num_tiles
///         Number of tile segments an eigenvector is split into.
///
/// @param[in] num_selected
///         Number of columns in the eigenvector matrix.
///
/// @param[in] scales
///         A num_tiles-by-num_cols matrix of scaling factors.
///
/// @param[out] smin
///         An array of length num_cols. On exit, the i-th entry holds the
///         smallest scaling factor for the i-th eigenvector.
///
void starneig_eigvec_std_find_smallest_scaling(int num_tiles, int num_selected,
    const scaling_t *restrict scales, scaling_t *restrict smin);


///
/// @brief Scales a vector with a scalar.
///
/// @param[in] n
///         Number of entries in x. n >= 0.
///
/// @param[in,out] x
///         Vector of length n. On exit, x := beta * x.
///
/// @param[in] beta
///         Pointer to a scalar.
///
void starneig_eigvec_std_scale(int n, double *restrict const x, const scaling_t *beta);


///
/// @brief Combines two scalars.
///
/// @param[in,out] global
///         Pointer to a scalar. On exit the combined scalar of global and phi.
///
/// @param[in] phi
///         A scalar scaling factor.
///
void starneig_eigvec_std_update_global_scaling(scaling_t *global, scaling_t phi);


///
/// @brief Multiplies a scalar with a scaling factor.
///
/// @param[in, out] norm
///         Pointer to a scalar. On exit, norm scaled with phi.
///
/// @param[in] phi
///         A scalar scaling factor.
///
void starneig_eigvec_std_update_norm(double *norm, scaling_t phi);


///
/// @brief Compute ratio between alpha_min and alpha for upscaling.
///
/// @param[in] alpha_min
///         The smallest scalar.
///
/// @param[in] alpha
///         A scalar.
///
/// @return alpha_min / alpha
///
double starneig_eigvec_std_compute_upscaling(scaling_t alpha_min, scaling_t alpha);


///
/// @brief Converts a scaling to a double-precision scaling factor.
///
/// @param[in] alpha
///         A scalar.
///
/// @return The scalar alpha converted to double-precision.
///
double starneig_eigvec_std_convert_scaling(scaling_t alpha);


///
/// @brief Compute ratio required for upscaling.
///
/// @param[in] alpha_min
///         The smallest scalar.
///
/// @param[in] alpha
///         A scalar.
///
/// @param[in] beta
///         A scalar.
///
/// @return (alpha_min / alpha) * beta
///
double starneig_eigvec_std_compute_combined_upscaling(
    scaling_t alpha_min, scaling_t alpha, scaling_t beta);


///
/// @brief Computes scaling such that the update y := y + t x cannot overflow.
///
/// If the return type is of type double, this routine
/// returns a scaling alpha such that y := (alpha * y) + t * (alpha * x)
/// cannot overflow.
///
/// If the return type is of type int, this routine
/// returns a scaling alpha such that y := (2^alpha * y) + t * (2^alpha * x)
/// cannot overflow.
///
/// Without checks, this routine assumes 0 <= t, x, y <= Omega.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] t
///         Scalar. 0 <= t <= Omega.
///
/// @param[in] x
///         Scalar. 0 <= x <= Omega.
///
/// @param[in] y
///         Scalar. 0 <= y <= Omega.
///
/// @return The scaling factor alpha.
///
scaling_t starneig_eigvec_std_protect_update(double t, double x, double y);


///
/// @brief Computes scaling such that the update Y(:,i) := Y(:,i) + T X(:,i)
/// cannot overflow.
///
/// This routine wraps multiple calls to protect_update().
///
/// Without checks, this routine assumes that all norms satisfy
/// 0 <= norm <= Omega.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] Xnorms
///         Array of length num_rhs. The i-th entry contains an upper bound
///         for X(:,i). 0 <= Xnorms[i] <= Omega.
///
/// @param[in] num_rhs
///         Number of right-hand sides. num_rhs >= 0.
///
/// @param[in] tnorm
///         Scalar, upper bounds of T. 0 <= tnorm <= Omega.
///
/// @param[in] Ynorms
///        Array of length num_rhs. The i-th entry contains an upper bound
///        for Y(:,i). 0 <= Ynorms[i] <= Omega.
///
/// @param[in] lambda_type
///         Array of length num_rhs. The i-th entry is 0 if X(:,i) and Y(:,i)
///         is a real-valued column. The i-th and (i+1)-th entry are 1 if
///         X(:,i:i+1) and Y(:,i:i+1) are the real and imaginary part of
///         a complex column.
///
/// @param[out] scales
///         Array of length num_rhs. The i-th entry holds a scaling factor
///         to survive Y(:,i) + T X(:,i).
///
/// @return Flag that indicates if rescaling is necessary (status == RESCALE)
///         or not (status == NO_RESCALE).
///
int starneig_eigvec_std_protect_multi_rhs_update(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    const int *restrict const lambda_type,
    scaling_t *restrict const scales);


///
/// @brief Solves (t - lambda) * ? = x robustly.
///
/// If the type of scale is double, the routine solves (scale * x) / (t - lambda)
/// whereas, if the type of scale is int, the routine solves
/// (2^scale * x) / (t - lambda) such that no overflow occurs.
///
/// @param[in] smin
///         Desired lower bound on (t - lambda).
///
/// @param[in] t
///         Real scalar t.
///
/// @param[in] lambda
///         Real scalar lambda.
///
/// @param[in, out] x
///         On entry, the scale rhs. On exit, the real solution x in
///         (scale * x) / (t - lambda) or in (2^scale * x) / (t - lambda).
///
/// @param[out] scale
///         Scalar scaling factor of x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if (t - lambda)
///         was perturbed to make it greater than smin.
///
int starneig_eigvec_std_solve_1x1_real_system(
    double smin, double t, double lambda, double *x, scaling_t *scale);


///
/// @brief Solves (t - lambda_re - lambda_im) * ? = x_re + i * x_im robustly.
///
/// If the type of scale is double, the routine solves (scale * x) / (t - lambda)
/// whereas, if the type of scale is int, the routine solves
/// (2^scale * x) / (t - lambda) such that no overflow occurs. The complex
/// division is executed in real arithmetic.
///
/// @param[in] smin
///         Desired lower bound on (t - lambda_re - lambda_im).
///
/// @param[in] t
///         Real scalar t.
///
/// @param[in] lambda_re
///         Real part of the scalar complex eigenvalue.
///
/// @param[in] lambda_im
///         Imaginary part of the scalar complex eigenvalue.
///
/// @param[in, out] x_re
///         On entry, the real part of the right-hand side. On exit, the real
///         part of the solution.
///
/// @param[in, out] x_im
///         On entry, the imaginary part of the right-hand side. On exit, the
///         imaginary part of the solution.
///
/// @param[out] scale
///         Joint scalar scaling factor for the real and imaginary part of
///         the solution x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if (t - lambda)
///         was perturbed to make it greater than smin.
///
int starneig_eigvec_std_solve_1x1_cmplx_system(double smin, double t, double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t *scale);


///
/// @brief Solves a real-valued 2-by-2 system robustly.
///
/// Solves the real-valued system
///        [ t11-lambda  t12        ] * [ x1 ] = [ b1 ]
///        [ t21         t22-lambda ]   [ x2 ]   [ b2 ]
/// such that if cannot overflow.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] smin
///         Desired lower bound on the singular values of (T - lambda * I).
///
/// @param[in] T
///         Real 2-by-2 matrix T.
///
/// @param[in] ldT
///         The leading dimension of T. ldT >= 2.
///
/// @param[in] lambda
///         Real eigenvalue.
///
/// @param[in, out] b
///         Real vector of length 2. On entry, the right-hand side.
///         On exit, the solution.
///
/// @param[out] scale
///         Scalar scaling factor of the solution x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if the singular
///         values of (T - lambda * I) were smaller than smin and perturbed.
///
int starneig_eigvec_std_solve_2x2_real_system(
    double smin,
    const double *restrict const T, int ldT,
    double lambda,
    double *restrict const b, scaling_t *restrict const scale);


///
/// @brief Solves a complex-valued 2-by-2 system robustly.
///
/// Let lambda := lambda_re + i * lambda_im. Solves the complex-valued system
///       [ t11-lambda_re   t12        ] * [ x1 ] = [ b_re1 ] + i * [b_im1]
///       [ t21             t22-lambda ]   [ x2 ]   [ b_re2 ]       [b_im2]
/// such that if cannot overflow. The solution x1 and x2 is complex-valued.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] smin
///         Desired lower bound on the singular values of (T - lambda * I).
///
/// @param[in] T
///         Real 2-by-2 matrix T.
///
/// @param[in] ldT
///         The leading dimension of T. ldT >= 2.
///
/// @param[in] lambda_re
///         Real part of the eigenvalue.
///
/// @param[in] lambda_im
///         Imaginary part of the eigenvalue.
///
/// @param[in, out] b_re
///         Vector of length 2. On entry, the real part of the right-hand side.
///         On exit, the real part of the solution.
///
/// @param[in, out] b
///         Vector of length 2. On entry, the imaginary part of the right-hand
///         side. On exit, the imaginary part of the solution.
///
/// @param[out] scale
///         Joint scalar scaling factor of the solution x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if the singular
///         values of (T - lambda * I) were smaller than smin and perturbed.
///
int starneig_eigvec_std_solve_2x2_cmplx_system(
    double smin,
    const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    scaling_t *restrict const scale);

#endif
