//
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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "robust.h"
#include "../../common/common.h"

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <math.h>


int MIN_EXP = DBL_MIN_EXP - 1; // -1022
int MAX_EXP = DBL_MAX_EXP - 1; //  1023


static const double g_omega = 1.e+300;     ///< overflow threshold
static const double g_omega_inv = 1.e-300; ///< inverse of the overflow threshold

#define NO_RESCALE 0
#define RESCALE 1
#define REAL 0
#define CMPLX 1


///////////////////////////////////////////////////////////////////////////////
// initialize scaling factors
////////////////////////////////////////////////////////////////////////////////
void starneig_eigvec_std_init_scaling_factor(int n, scaling_t *alpha)
{
#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    for (int i = 0; i < n; i++)
        alpha[i] = 0;
#else
    for (int i = 0; i < n; i++)
        alpha[i] = 1.0;
#endif
}


///////////////////////////////////////////////////////////////////////////////
// find the smallest scaling factor
////////////////////////////////////////////////////////////////////////////////
void starneig_eigvec_std_find_smallest_scaling(int num_tiles, int num_selected,
    const scaling_t *restrict scales, scaling_t *restrict smin)
{
#define scales(col, tilerow) scales[(col) + (tilerow) * (size_t)num_selected]

    starneig_eigvec_std_init_scaling_factor(num_selected, smin);

    // Find the minimum scaling factor for each column.
    for (int j = 0; j < num_selected; j++) {
        for (int tli = 0; tli < num_tiles; tli++) {
#ifdef STARNEIG_ENABLE_INTEGER_SCALING
            smin[j] = MIN(smin[j], scales(j, tli));
#else
            smin[j] = MIN(smin[j], scales(j, tli));
#endif
        }
    }

#undef scales
}



///////////////////////////////////////////////////////////////////////////////
// manipulation of scaling factors
///////////////////////////////////////////////////////////////////////////////
void starneig_eigvec_std_scale(int n, double *restrict const x, const scaling_t *beta)
{
#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    double alpha = ldexp(1.0, beta[0]);
#else
    double alpha = beta[0];
#endif

    // Scale vector, if necessary.
    if (alpha != 1.0) {
        for (int i = 0; i < n; i++) {
            x[i] = alpha * x[i];
        }
    }
}


void starneig_eigvec_std_update_global_scaling(scaling_t *global, scaling_t phi)
{
#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    *global = phi + (*global);
#else
    *global = phi * (*global);
#endif
}


void starneig_eigvec_std_update_norm(double *norm, scaling_t phi)
{
#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    *norm = ldexp(1.0, phi) * (*norm);
#else
    *norm = phi * (*norm);
#endif
}


double starneig_eigvec_std_compute_upscaling(scaling_t alpha_min, scaling_t alpha)
{
    double scaling;

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    // Common scaling is 2^alpha_min / 2^alpha.
    scaling_t exp = alpha_min - alpha;
    scaling = ldexp(1.0, exp);
#else
    scaling = alpha_min / alpha;
#endif

    return scaling;
}


double starneig_eigvec_std_convert_scaling(scaling_t alpha)
{
#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    double scaling = ldexp(1.0, alpha);
#else
    double scaling = alpha;
#endif

    return scaling;
}


double starneig_eigvec_std_compute_combined_upscaling(
    scaling_t alpha_min, scaling_t alpha, scaling_t beta)
{
    double scaling;

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    // Common scaling is (2^alpha_min / 2^alpha) * 2^beta.
    scaling_t exp = alpha_min - alpha + beta;
    scaling = ldexp(1.0, exp);
#else
    scaling = (alpha_min / alpha) * beta;
#endif

    return scaling;
}



////////////////////////////////////////////////////////////////////////////////
// protect real division
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Compute scaling such that the division b / t cannot overflow
 * where b, t are real-valued.
 *
 * If the return type is double-prevision, this routine returns a scaling alpha
 * such that x = (alpha * b) / t cannot overflow.
 *
 * If the return type is int, this routine returns a scaling alpha such that
 * x = (2^alpha * b) / t cannot overflow.
 *
 * Assume |b|, |t| are bounded by Omega.
 *
 * Credits: Carl Christian Kjelgaard Mikkelsen.
 */
static double protect_real_division(double b, double t)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Find scaling alpha such that x = (alpha * b) / t cannot overflow.
    if (fabs(t) < g_omega_inv) {
        if (fabs(b) > fabs(t) * g_omega) {
            // Please observe that scales will be strictly less than 1.
            scale = (fabs(t) * g_omega) / fabs(b);
        }
    }
    else { // fabs(t) >= g_omega_inv
        // Exploit short circuiting, i.e., the left side is evaluated first.
        // If 1.0 > abs(t) holds, then it is safe to compute
        // fabs(t) * g_omega.
        if (1.0 > fabs(t) && fabs(b) > fabs(t) * g_omega) {
            scale = 1.0 / fabs(b);
        }
    }

    return scale;
}



////////////////////////////////////////////////////////////////////////////////
// protect sum
////////////////////////////////////////////////////////////////////////////////

// Returns scaling such that sum := (alpha * x) + (alpha * y) cannot overflow.
static double protect_sum(double x, double y)
{
    double scale = 1.0;

    // Protect against overflow if x and y have the same sign.
    if ((x > 0 && y > 0) || (x < 0 && y < 0))
        if (fabs(x) > g_omega - fabs(y))
            scale = 0.5;

    return scale;
}



////////////////////////////////////////////////////////////////////////////////
// protect multiplication (internal)
////////////////////////////////////////////////////////////////////////////////

// Returns scaling alpha such that y := t * (alpha * x) cannot overflow.
static double protect_mul(double tnorm, double xnorm)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Process simplified decision tree of protect_update().
    if (fabs(xnorm) <= 1.0) {
        if (fabs(tnorm) * fabs(xnorm) > g_omega) {
            scale = 0.5;
        }
    }
    else { // xnorm > 1.0
        if (fabs(tnorm) > g_omega / fabs(xnorm)) {
            scale = 0.5 / fabs(xnorm);
        }
    }

    return scale;
}


////////////////////////////////////////////////////////////////////////////////
// protect update
////////////////////////////////////////////////////////////////////////////////

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
scaling_t /* == int*/ starneig_eigvec_std_protect_update(
    double tnorm, double xnorm, double ynorm)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Process decision tree.
    if (xnorm <= 1.0) {
        if (tnorm * xnorm > g_omega - ynorm) {
            scale = 0.5;
        }
    }
    else { // xnorm > 1.0
        if (tnorm > (g_omega - ynorm) / xnorm) {
            scale = 0.5 / xnorm;
        }
    }

    return ilogb(scale);
}

#else

// Returns scaling alpha such that y := (alpha * y) - t * (alpha * x) cannot
// overflow.
scaling_t /* == double*/ starneig_eigvec_std_protect_update(
    double tnorm, double xnorm, double ynorm)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Process decision tree.
    if (xnorm <= 1.0) {
        if (tnorm * xnorm > g_omega - ynorm) {
            scale = 0.5;
        }
    }
    else { // xnorm > 1.0
        if (tnorm > (g_omega - ynorm) / xnorm) {
            scale = 0.5 / xnorm;
        }
    }

    return scale;
}

#endif

////////////////////////////////////////////////////////////////////////////////
// protect update scalar
////////////////////////////////////////////////////////////////////////////////

static double protect_update_scalar(double t, double x, double y)
{
    double scale = 1.0;

    // Protect p = x * y.
    double alpha1 = protect_mul(x, t);
    double p = t * (alpha1 * x);
    if (abs(ilogb(y) - ilogb(p)) > 52) {
        // The factors are far apart. Either y or p is the final result.
        if (ilogb(p) > ilogb(y))
            scale = alpha1;
    }
    else {
        // Scale y consistently.
        y = y / alpha1;
        double alpha2 = protect_sum(y, -p);
        scale = alpha1 * alpha2;
    }

    return scale;
}



////////////////////////////////////////////////////////////////////////////////
// protect multi-rhs update
////////////////////////////////////////////////////////////////////////////////

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
int starneig_eigvec_std_protect_multi_rhs_update(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    const int *restrict const lambda_type,
    scaling_t /* == int*/ *restrict const scales)
{
    // Status flag to indicate if rescaling is necessary.
    int status = NO_RESCALE;

    for (int k = num_rhs - 1; k >= 0; k--) {
        // Compute scaling factor for the k-th eigenvector.
        scales[k] = starneig_eigvec_std_protect_update(tnorm, Xnorms[k], Ynorms[k]);

        if (lambda_type[k] == CMPLX) {
            // We have only one scaling factor per complex conjugate pair.
            scales[k - 1] = scales[k];

            // Skip the next entry.
            k--;
        }

        if (scales[k] != 0)
            status = RESCALE;
    }

    return status;
}

#else

int starneig_eigvec_std_protect_multi_rhs_update(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    const int *restrict const lambda_type,
    scaling_t /* == double*/ *restrict const scales)
{
    // Status flag to indicate if rescaling is necessary.
    int status = NO_RESCALE;

    for (int k = num_rhs - 1; k >= 0; k--) {
        // Compute scaling factor for the k-th eigenvector.
        scales[k] = starneig_eigvec_std_protect_update(tnorm, Xnorms[k], Ynorms[k]);

        if (lambda_type[k] == CMPLX) {
            // We have only one scaling factor per complex conjugate pair.
            scales[k - 1] = scales[k];

            // Skip the next entry.
            k--;
        }

        if (scales[k] != 1.0)
            status = RESCALE;
    }

    return status;
}

#endif


////////////////////////////////////////////////////////////////////////////////
// solve 1x1 real system
////////////////////////////////////////////////////////////////////////////////

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
int starneig_eigvec_std_solve_1x1_real_system(
    double smin, double t, double lambda, double *x,
    scaling_t /* == int*/ *scale)
{
    int info = 0;

    // Compute csr := t + (-lambda) robustly. Note that the scaling contributes
    // as reciprocal to the global scaling.
    double s = protect_sum(t, -lambda);
    double csr = (s * t) - (s * lambda);

    if (fabs(csr) < smin) {
        csr = smin;
        info = 1;
    }

    // Compute a scaling to survive the real-valued division.
    double alpha = protect_real_division(x[0], csr);

    // Execute the division safely.
    x[0] = (alpha * x[0]) / csr;

    // Return scaling factor.
    scale[0] = ilogb(alpha / s);

    return info;
}

#else

/// Solves the real 1x1 system (t - lambda) x = b robustly.
/// x = x / (t - lambda)
int starneig_eigvec_std_solve_1x1_real_system(
    double smin, double t, double lambda, double *x,
    scaling_t /* == double*/ *scale)
{
    int info = 0;

    // Compute csr := t + (-lambda) robustly. Note that the scaling contributes
    // as reciprocal to the global scaling.
    double s = protect_sum(t, -lambda);
    double csr = (s * t) - (s * lambda);

    if (fabs(csr) < smin) {
        csr = smin;
        info = 1;
    }

    // Compute a scaling to survive the real-valued division.
    double alpha = protect_real_division(x[0], csr);

    // Execute the division safely.
    x[0] = (alpha * x[0]) / csr;

    // Return scaling factor.
    scale[0] = alpha / s;

    return info;
}

#endif


////////////////////////////////////////////////////////////////////////////////
// complex division in real arithmetic
////////////////////////////////////////////////////////////////////////////////

static void dladiv2(double a, double b, double c, double d, double r, double t,
    double *ret, double *scale)
{
    volatile double res;
    double alpha = 1.0;

    if (r != 0.0) {
        // Since r is in [0, 1], the multiplication is safe to execute.
        volatile double br = b * r;

        if (br != 0.0) {
            // res = (a + br) * t
            double s = protect_sum(a, br);
            res = (s * a) + (s * br);
            alpha = s * alpha;

            // WARNING: If optimization flags activate associative math, the
            // brackets in the computation of res is ignored. This problem has
            // been observed with -Ofast (GCC) and -O3 (Intel). The computation
            // overflows and produces NaNs in the solution.
            // The crude fix is as follows:
            // volatile double sres = s * res;
            // res = sres * t;
            s = protect_mul(fabs(t), fabs(res));
            res = (s * res) * t;
            alpha = s * alpha;
        }
        else {
            // res = a * t + (b * t) * r
            // Left term.
            double s1 = protect_mul(fabs(t), fabs(a));
            volatile double tmp1 = (s1 * a) * t;

            // Right term.
            double s2 = protect_mul(fabs(t), fabs(b));
            volatile double tmp2 = (s2 * b) * t;
            // The multiplication with r is safe.
            tmp2 = tmp2 * r;

            // Scale summands consistently.
            double smin = MIN(s1, s2);
            tmp1 = tmp1 * (s1 / smin);
            tmp2 = tmp2 * (s2 / smin);
            alpha = smin * alpha;

            // Add both terms.
            double s = protect_sum(tmp1, tmp2);
            res = (s * tmp1) + (s * tmp2);
            alpha = s * alpha;
        }
    }
    else {
        // res = (a + d * (b / c)) * t
        // tmp = b / c
        double s1 = protect_real_division(b, c);
        alpha = s1 * alpha;
        volatile double tmp = (s1 * b) / c;

        // tmp = d * tmp
        double s2 = protect_mul(fabs(d), fabs(tmp));
        alpha = s2 * alpha;
        tmp = d * (s2 * tmp);

        // Apply scaling to left term 'a' in the sum so that both summands
        // are consistently scaled.
        a = (s1 * s2) * a;

        // tmp = a + tmp
        double s = protect_sum(a, tmp);
        alpha = s * alpha;
        tmp = (s * a) + (s * tmp);

        // res = tmp * t
        s = protect_mul(fabs(tmp), fabs(t));
        alpha = s * alpha;
        res = (s * tmp) * t;
    }

    // Return augmented vector (alpha, res).
    *scale = alpha;
    *ret = res;
}



static void dladiv1(double a, double b, double c, double d,
    double *p, double *q, double *scale)
{
    //           a + ib
    // p + i q = -------
    //           c + id

    // Since |d| < |c|, this division is safe to execute.
    volatile double r = d / c;

    // t = 1 / (c + d * r)
    // Since r is in [0, 1], the multiply is safe.
    volatile double dr = d * r;

    double s1 = protect_sum(c, dr);
    volatile double sum = (s1 * c) + (s1 * dr);

    double s2 = protect_real_division(1.0, sum);
    volatile double t = 1.0 / (s2 * sum);
    volatile double alpha = 1.0 / (s1 * s2);

    // Introduce local scaling factors for dladiv2.
    double beta1 = 1.0, beta2 = 1.0;

    // Compute (beta1, p).
    dladiv2(a, b, c, d, r, t, p, &beta1);

    // Compute (beta2, q).
    dladiv2(b, -a, c, d, r, t, q, &beta2);

    // Scale real and imaginary part consistently.
    double beta = 1.0;
    if ((beta1 > 1.0 && beta2 < 1.0) || (beta1 < 1.0 && beta2 > 1.0)) {
        starneig_error(
            "The scalings cannot be consolidated without overflow or " \
            "underflow.\n");
        // A complex eigenvector has a real part that under/overflowed and an
        // imaginary part that over/underflowed. LAPACK cannot capture this
        // case either (they, too, have only one scaling factor per eigenvector)
    }
    else {
        // Find the more extreme scaling factor.
        beta = MIN(beta1, beta2);

        // Apply scaling.
        *p = (*p) * (beta / beta1);
        *q = (*q) * (beta / beta2);
    }

    // Record global scaling factor.
    *scale = alpha * beta;
}


static void dladiv(double a, double b, double c, double d,
    double *x_re, double *x_im, double *scale)
{
    //                 a + ib
    // x_re + i x_im = -------
    //                 c + id
    if (fabs(d) < fabs(c)) {
        dladiv1(a, b, c, d, x_re, x_im, scale);
    }
    else {
        dladiv1(b, a, d, c, x_re, x_im, scale);
        *x_im = -(*x_im);
    }
}


#ifdef STARNEIG_ENABLE_INTEGER_SCALING

int starneig_eigvec_std_solve_1x1_cmplx_system(
    double smin, double t, double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t /* == int*/ *scale)
{
    int info = 0;

    // Solve (t - (lambda_re + i * lambda_im)) (p + i * q) = x_re + i * x_im.

    // Compute csr := (t + (-lambda_re)) robustly.
    double s = protect_sum(t, -lambda_re);
    double csr = (s * t) - (s * lambda_re);

    // Scale consistently csi := s * (-lambda_im).
    double csi = s * (-lambda_im);

    // Note that the scaling is applied to the rhs (x_re + i * x_im) after
    // the complex division.

    if (fabs(csr) + fabs(csi) < smin) {
        csr = smin;
        csi = 0.0;
        info = 1;
    }

    // The scaling check for X = B / C in LAPACK is covered in protect_division.

    // Local scaling factor generated in the process of the complex division.
    double alpha = 1.0;

    // Compute the complex division in real arithmetic.
    //                 a + ib
    // x_re + i x_im = -------
    //                 c + id
    double a = *x_re;
    double b = *x_im;
    double c = csr;
    double d = csi;
    dladiv(a, b, c, d, x_re, x_im, &alpha);

    // Combine scaling factors and convert to int scaling factor.
    *scale = ilogb((1.0 / s) * (alpha));

    return info;
}

#else

int starneig_eigvec_std_solve_1x1_cmplx_system(
    double smin, double t, double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t /* == double*/ *scale)
{
    int info = 0;

    // Solve (t - (lambda_re + i * lambda_im)) (p + i * q) = x_re + i * x_im.

    // Compute csr := (t + (-lambda_re)) robustly.
    double s = protect_sum(t, -lambda_re);
    double csr = (s * t) - (s * lambda_re);

    // Scale consistently csi := s * (-lambda_im).
    double csi = s * (-lambda_im);

    // Note that the scaling is applied to the rhs (x_re + i * x_im) after
    // the complex division.

    if (fabs(csr) + fabs(csi) < smin) {
        csr = smin;
        csi = 0.0;
        info = 1;
    }

    // The scaling check for X = B / C in LAPACK is covered in protect_division.

    // Local scaling factor generated in the process of the complex division.
    double alpha = 1.0;

    // Compute the complex division in real arithmetic.
    //                 a + ib
    // x_re + i x_im = -------
    //                 c + id
    double a = *x_re;
    double b = *x_im;
    double c = csr;
    double d = csi;
    dladiv(a, b, c, d, x_re, x_im, &alpha);

    // Combine scaling factors.
    *scale = (1.0 / s) * (alpha);

    return info;
}

#endif



////////////////////////////////////////////////////////////////////////////////
// solve 2x2 real system
////////////////////////////////////////////////////////////////////////////////


// Credits: Carl Christian Kjelgaard Mikkelsen
static double backsolve_real_2x2_system(double *T, int ldT, double *b)
{
#define T(i,j) T[(i) + (j) * (size_t)ldT]

    // Global scaling factor.
    double alpha = 1.0;

    double xnorm = MAX(fabs(b[0]), fabs(b[1]));

    double s = protect_real_division(b[1], T(1,1));
    if (s != 1.0) {
        // Apply scaling to right-hand side.
        b[0] = s * b[0];
        b[1] = s * b[1];

        // Update global scaling.
        alpha = s * alpha;

        // Update the infinity norm of the solution.
        xnorm = s * xnorm;
    }

    // Execute the division.
    b[1] = b[1] / T(1,1);

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    s = ldexp(1.0, starneig_eigvec_std_protect_update(fabs(T(0,1)), fabs(b[1]), xnorm));
#else
    s = starneig_eigvec_std_protect_update(fabs(T(0,1)), fabs(b[1]), xnorm);
#endif

    if (s != 1.0) {
        // Apply scaling to right-hand side.
        b[0] = s * b[0];
        b[1] = s * b[1];

        // Update global scaling.
        alpha = s * alpha;
    }

    // Execute the linear update.
    b[0] = b[0] - b[1] * T(0,1);

    // Recompute norm.
    xnorm = MAX(fabs(b[0]), fabs(b[1]));

    s = protect_real_division(b[0], T(0,0));
    if (s != 1.0) {
        // Apply scaling to right-hand side.
        b[0] = s * b[0];
        b[1] = s * b[1];

        // Update global scaling.
        alpha = s * alpha;

        // Update the infinity norm of the solution.
        xnorm = s * xnorm;
    }

    // Execute the division.
    b[0] = b[0] / T(0,0);

    return alpha;

#undef T
}

// Swap row 0 and row 1.
static void swap_rows(int n, double *C)
{
#define C(i,j) C[(i) + (j) * 2]

    // Swap row 0 and row 1.
    for (int j = 0; j < n; j++) {
        double swap = C(0,j);
        C(0,j) = C(1,j);
        C(1,j) = swap;
    }

#undef C
}


static void find_real_pivot(double *C, int *pivot_row, int *pivot_col)
{
#define C(i,j) C[(i) + (j) * 2]

    // Find the coordinates of the pivot element.
    int row = 0;
    int col = 0;
    double cmax = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double lmax = fabs(C(i,j));
            if (lmax > cmax) {
                row = i;
                col = j;
                cmax = lmax;
            }
        }
    }

    *pivot_row = row;
    *pivot_col = col;

#undef C
}


// Complete pivoting.
static int solve_2x2_real_system_internal(
    double smin, const double *restrict const T, int ldT,
    double lambda,
    double *restrict const b, double *restrict const scale)
{
#define T(i,j) T[(i) + (j) * (size_t)ldT]
#define C(i,j) C[(i) + (j) * 2]

    // Solve
    // (T - lambda I)x = b.

    int info = 0;

    // C = [(T - lambda * I) | b]
    double C[2 * 3];

    // Compute t + (-lambda) robustly. Recall the diagonals in the the 2-by-2
    // T block are equal, so that we have to protect only one subtraction.
    double s = protect_sum(T(0,0), -lambda);
    double csr = (s * T(0,0)) - (s * lambda);

    // Apply scaling to T. Note that scaling of b is not safe. Therefore s is
    // incorporated into the global scaling at the very end of this routine.
    // C := [s * (T - lambda I) | b].
    C(0,0) = csr;         C(0,1) = s * T(0,1);  C(0,2) = b[0];
    C(1,0) = s * T(1,0);  C(1,1) = csr;         C(1,2) = b[1];

    ////////////////////////////////////////////////////////////////////////////
    // Transform A to echelon form with complete pivoting.
    ////////////////////////////////////////////////////////////////////////////

    // Find pivot element in entire matrix.
    int pivot_row = 0, pivot_col = 0;
    find_real_pivot(C, &pivot_row, &pivot_col);

    // Permute pivot to the top-left corner.
    if (pivot_row == 1) {
        // Swap row 0 and row 1.
        swap_rows(3, C);
    }
    if (pivot_col == 1) {
        // Swap column 0 and column 1.
        for (int i = 0; i < 2; i++) {
            double swap = C(i,0);
            C(i,0) = C(i,1);
            C(i,1) = swap;
        }
    }

    // If the largest entry is 0.0, perturb.
    if (C(0,0) == 0.0) {
        C(0,0) = smin;
        info = 1;
    }

    // Compute multiplier, the reciprocal of the pivot.
    double ur11r = 1.0 / C(0,0);

    // Multiply first row with reciprocal of C(0,0).
    {
    C(0,0) = 1.0;
    C(0,1) = C(0,1) * ur11r; // Safe multiplication.

    // Treat rhs.
    double beta = protect_mul(C(0,2), ur11r);
    *scale = beta;
    C(0,2) = C(0,2) * beta;
    C(1,2) = C(1,2) * beta;
    C(0,2) = C(0,2) * ur11r;
    }

    // Second row - CR(1,0) * first_row.
    {
    C(1,1) = C(1,1) - C(1,0) * C(0,1); // Safe update.

    // Perturb C(1,1), if too small.
    if (fabs(C(1,1)) < smin) {
        C(1,1) = smin;
        info = 1;
    }

    // Treat rhs.
    double beta = protect_update_scalar(C(1,0), C(0,2), C(1,2));
    *scale = (*scale) * beta;
    C(0,2) = C(0,2) * beta;
    C(1,2) = C(1,2) * beta;
    C(1,2) = C(1,2) - C(1,0) * C(0,2);

    // (1,0) has been annihilated.
    C(1,0) = 0.0;
    }

    // The system is now in upper triangular form.

    ////////////////////////////////////////////////////////////////////////////
    // Backward substitution.
    ////////////////////////////////////////////////////////////////////////////

    double alpha = backsolve_real_2x2_system(&C(0,0), 2, &C(0,2));
    *scale = (*scale) * alpha;

    // Copy the solution back.
    if (pivot_col == 1) {
        b[0] = C(1,2);
        b[1] = C(0,2);
    }
    else {
        b[0] = C(0,2);
        b[1] = C(1,2);
    }

    return info;

#undef T
#undef C
}


#ifdef STARNEIG_ENABLE_INTEGER_SCALING

int starneig_eigvec_std_solve_2x2_real_system(
    double smin, const double *restrict const T, int ldT,
    double lambda,
    double *restrict const b, int *restrict const scale)
{
    int info = 0;

    // Local scaling factor.
    double phi = 1.0;

    info = solve_2x2_real_system_internal(smin, T, ldT, lambda, b, &phi);

    // Convert double-precision scaling factor to int scaling factor.
    *scale = ilogb(phi);

    return info;
}

#else

int starneig_eigvec_std_solve_2x2_real_system(
    double smin, const double *restrict const T, int ldT,
    double lambda,
    double *restrict const b, double *restrict const scale)
{
    return solve_2x2_real_system_internal(smin, T, ldT, lambda, b, scale);
}

#endif


////////////////////////////////////////////////////////////////////////////////
// solve 2x2 complex system
////////////////////////////////////////////////////////////////////////////////


// Credits: Carl Christian Kjelgaard Mikkelsen
static void find_pivot(double *CR, double *CI, int *pivot_row, int *pivot_col)
{
#define CR(i,j) CR[(i) + (j) * 2]
#define CI(i,j) CI[(i) + (j) * 2]
#define cr(i,j) cr[(i) + (j) * 2]
#define ci(i,j) ci[(i) + (j) * 2]

    double cr[2 * 2];
    double ci[2 * 2];

    // Copy CR, CI.
    cr(0,0) = CR(0,0); cr(0,1) = CR(0,1);
    cr(1,0) = CR(1,0); cr(1,1) = CR(1,1);
    ci(0,0) = CI(0,0); ci(0,1) = CI(0,1);
    ci(1,0) = CI(1,0); ci(1,1) = CI(1,1);

    // Scalings done here are applied only locally.

    // Find smallest scaling factor.
    double smin = 1.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double s = protect_sum(fabs(cr(i,j)), fabs(ci(i,j)));
            if (s < smin)
                smin = s;
        }
    }

    // Scale all entries, if necessary.
    if (smin != 1.0) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                cr(i,j) = smin * cr(i,j);
                ci(i,j) = smin * ci(i,j);
            }
        }
    }

    // Now it is safe to find the coordinates of the pivot element.
    int row = 0;
    int col = 0;
    double cmax = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double lmax = fabs(cr(i,j)) + fabs(ci(i,j));
            if (lmax > cmax) {
                row = i;
                col = j;
                cmax = lmax;
            }
        }
    }

    *pivot_row = row;
    *pivot_col = col;

#undef CR
#undef CI
#undef cr
#undef ci
}


static int solve_2x2_cmplx_system_internal(
    double smin, const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    double *restrict const scale)
{
#define CR(i,j) CR[(i) + (j) * 2]
#define CI(i,j) CI[(i) + (j) * 2]
#define T(i,j) T[(i) + (j) * (size_t)ldT]

    // Solve
    // (T - lambda I) x = b.

    int info = 0;

    // CR = [(T - lambda_re * I) | b_re], CI = [(-lambda_im * I) | b_im].
    double CR[2 * 3];
    double CI[2 * 3];

    // Compute t + (-lambda_re) robustly. Recall the diagonals in the 2-by-2
    // T block are equal, so that we have to protect only one subtraction.
    double s = protect_sum(T(0,0), -lambda_re);
    double csr = (s * T(0,0)) - (s * lambda_re);

    // Apply scaling to T. Note that scaling of b is not safe. Therefore s is
    // incorporated into the global scaling at the very end of this routine.
    // CR + i * CI := s * (T - lambda I).
    CR(0,0) = csr;          CR(0,1) = s * T(0,1);   CR(0,2) = b_re[0];
    CR(1,0) = s * T(1,0);   CR(1,1) = csr;          CR(1,2) = b_re[1];

    CI(0,0) = s * (-lambda_im);CI(0,1) = 0.0;             CI(0,2) = b_im[0];
    CI(1,0) = 0.0;             CI(1,1) = s * (-lambda_im);CI(1,2) = b_im[1];

    ////////////////////////////////////////////////////////////////////////////
    // Transform A to echelon form with complete pivoting.
    ////////////////////////////////////////////////////////////////////////////

    // Find pivot element in entire matrix.
    int pivot_row = 0, pivot_col = 0;
    find_pivot(CR, CI, &pivot_row, &pivot_col);

    // Permute pivot to the top-left corner.
    if (pivot_row == 1) {
        // Swap row 0 and row 1.
        swap_rows(3, CR);
        swap_rows(3, CI);
    }
    if (pivot_col == 1) {
        // Swap column 0 and column 1.
        for (int i = 0; i < 2; i++) {
            double swap = CR(i,0);
            CR(i,0) = CR(i,1);
            CR(i,1) = swap;
            swap = CI(i,0);
            CI(i,0) = CI(i,1);
            CI(i,1) = swap;
        }
    }

    // Recall that (T-lambda I) has form [ a b; c a ]. With pivoting, there
    // are three cases.
    // 1) a is pivot. a is complex-valued, i.e. CI(0,0) != 0 or CI(1,1) != 0.
    // 2) b is pivot. After column pivoting, real values are on the diagonal.
    // 3) c is pivot. After row pivoting, real values are on the diagonal.

    if (CI(0,0) != 0.0 || CI(1,1) != 0.0) {
        // The pivot element is complex. As a consequence, the off-diagonals
        // are real.

        // Compute multipliers.
        volatile double temp;
        volatile double ur11r, ui11r;
        // Compute reciprocal of CR(0,0) + i CI(0,0) as
        //         1              CR(0,0) - i CI(0,0)
        // ------------------- = ---------------------
        // CR(0,0) + i CI(0,0)   CR(0,0)^2 + CI(0,0)^2
        if (fabs(CR(0,0)) > fabs(CI(0,0))) {
            //  CR(0,0) - i CI(0,0)         1 - i CI(0,0)/CR(0,0)
            // --------------------- = ----------------------------
            // CR(0,0)^2 + CI(0,0)^2    CR(0,0) + CI(0,0)^2/CR(0,0)
            //
            //      1 - i CI(0,0)/CR(0,0)
            // = -----------------------------------
            //   CR(0,0) * (1 + (CI(0,0)/CR(0,0))^2)
            temp = CI(0,0) / CR(0,0);
            // temp is in [0, 1). Then (1.0 + temp * temp) is in [1, 2).
            // The multiplication cannot overflow (safe mantissa growth).
            // As CR(0,0) is representable, the division is safe, too.
            ur11r = 1.0 / (CR(0,0) * (1.0 + temp * temp) );
            // Safe multiplication because temp is in [0, 1).
            ui11r = -temp * ur11r;
        }
        else { // 0 <= fabs(CR(0,0)) <= fabs(CI(0,0))
            if (CI(0,0) == 0.0) {
                // (T - lambda * I) is a zero matrix. Perturb.
                CI(0,0) = smin;
                info = 1;
            }

            // The safety of all instructions follows as in the if case.
            temp = CR(0,0) / CI(0,0);
            ui11r = -1.0 / ( CI(0,0)*( 1.0 + temp * temp) );
            ur11r = -temp * ui11r;
        }

        // Multiply first row with reciprocal of CR(0,0) + i CI(0,0).
        {
        CI(0,1) = CR(0,1) * ui11r;
        CR(0,1) = CR(0,1) * ur11r;

        // Treat rhs.
        // Prevent data race.
        temp = CR(0,2);

        // CR(0,2) = CR(0,2) * ur11r - CI(0,2) * ui11r;
        // CI(0,2) = temp * ui11r + CI(0,2) * ur11r;
        // Investigate multiplications and apply most constraining scaling.
        double beta1 = protect_mul(CR(0,2), ur11r);
        double beta2 = protect_mul(CI(0,2), ui11r);
        double beta3 = protect_mul(temp, ui11r);
        double beta4 = protect_mul(CI(0,2), ur11r);
        double beta = MIN(MIN(beta1, beta2), MIN(beta3, beta4));
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        temp = temp * beta;
        volatile double tmp1 = CR(0,2) * ur11r;
        volatile double tmp2 = - CI(0,2) * ui11r;
        volatile double tmp3 = temp * ui11r;
        volatile double tmp4 = CI(0,2) * ur11r;
        beta = MIN(protect_sum(tmp1, tmp2), protect_sum(tmp3, tmp4));
        *scale = (*scale) * beta;

        CR(0,2) = beta * tmp1 + beta * tmp2;
        CI(0,2) = beta * tmp3 + beta * tmp4;

        // (0,0) has been normalized.
        CR(0,0) = 1.0;
        CI(0,0) = 0.0;
        }

        // Second row - CR(1,0) * first row.
        {
        // Treat rhs.
        // Use more extreme scaling factor.
        double beta1 = protect_update_scalar(CR(1,0), CR(0,2), CR(1,2));
        double beta2 = protect_update_scalar(CR(1,0), CI(0,2), CI(1,2));
        double beta = MIN(beta1, beta2);
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        *scale = (*scale) * beta;

        CR(1,2) = CR(1,2) - CR(1,0) * CR(0,2);
        CI(1,2) = CI(1,2) - CR(1,0) * CI(0,2);

        // Treat (1,1).
        //   CR11 + i CI11 - CR10 * (CR01 + i CI01)
        // = [CR11 + i CI11] * [1 - CR10 / (CR11 + i CI11) * (CR01 + i CI01)]
        // As (1,0) and (0,1) have opposite signs, cancellation cannot occur.

        // Temporarily use (1,0) to compute CR10 / (CR11 + i CI11).
        // Reuse reciprocal since (1,1) and (0,0) are identical.
        CI(1,0) = CR(1,0) * ui11r;
        CR(1,0) = CR(1,0) * ur11r;

        // Compute 1 - CR10 / (CR11 + i CI11) * (CR01 + i CI01).
        const volatile double tr = 1.0 - CR(1,0) * CR(0,1) + CI(1,0) * CI(0,1);
        const volatile double ti = - CR(1,0) * CI(0,1) - CI(1,0) * CR(0,1);

        // Compute final multiplication with [CR11 + i CI11].
        // Precent data race.
        temp = CR(1,1);
        CR(1,1) = CR(1,1) * tr - CI(1,1) * ti;
        CI(1,1) = CI(1,1) * tr + temp * ti;

        // (1,0) has been annihilated.
        CR(1,0) = 0.0;
        CI(1,0) = 0.0;
        }
    }
    else {
        // The pivot element is real. The off-diagonals are complex.

        // If (T - lambda * I) is a zero matrix, perturb the diagonal (not just
        // CR(0,0)) to maintain the Schur canonical form.
        if (CR(0,0) == 0.0) {
            CR(0,0) = smin;
            CR(1,1) = smin;
            info = 1;
        }

        // Multiply first row with 1/CR(0,0). This multiplication is safe for
        // [ CR(0,0)    CR(0,1)+i*CI(0,1) ] when Omega is at least 1.
        {
        CR(0,1) = (1.0 / CR(0,0)) * CR(0,1);
        CI(0,1) = (1.0 / CR(0,0)) * CI(0,1);

        // Threat rhs.
        //CR(0,2) = (1.0 / CR(0,0)) * CR(0,2);
        //CI(0,2) = (1.0 / CR(0,0)) * CI(0,2);
        // Investigate multiplications and apply most constraining scaling.
        double beta1 = protect_mul(fabs(1.0 / CR(0,0)), fabs(CR(0,2)));
        double beta2 = protect_mul(fabs(1.0 / CR(0,0)), fabs(CI(0,2)));
        double beta = MIN(beta1, beta2);
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        CR(0,2) = (1.0 / CR(0,0)) * CR(0,2);
        CI(0,2) = (1.0 / CR(0,0)) * CI(0,2);

        // Entry (0,0) has been normalized.
        CR(0,0) = 1.0;
        CI(0,0) = 0.0;
        }

        // Eliminate C(1,0): second row - C(1,0) * first row.
        {
        // C11 is real-valued.
        CR(1,1) = CR(1,1) - CR(1,0) * CR(0,1) + CI(1,0) * CI(0,1);
        CI(1,1) = - CI(1,0) * CR(0,1) - CR(1,0) * CI(0,1);

        // Treat rhs.
        //CR(1,2) = CR(1,2) - CR(1,0) * CR(0,2) + CI(1,0) * CI(0,2);
        //CI(1,2) = CI(1,2) - CI(1,0) * CR(0,2) - CR(1,0) * CI(0,2);
        // Investigate multiplications and apply most constraining scaling.
        double beta1 = protect_mul(CR(1,0), CI(0,2));
        double beta2 = protect_mul(CI(1,0), CI(0,2));
        double beta3 = protect_mul(CI(1,0), CR(0,2));
        double beta4 = protect_mul(CR(1,0), CI(0,2));
        double beta = MIN(MIN(beta1, beta2), MIN(beta3, beta4));
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        double max1 = MAX(fabs(CR(1,2)), fabs(CR(1,0) * CR(0,2)));
        double max2 = MAX(fabs(CI(1,0) * CI(0,2)), fabs(CI(1,2)));
        double max3 = MAX(fabs(CI(1,0) * CR(0,2)), fabs(CR(1,0) * CI(0,2)));
        max1 = MAX(max1, max2);
        max1 = MAX(max1, max3);
        beta = protect_sum(max1, 2 * max1);
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        CR(1,2) = CR(1,2) - CR(1,0) * CR(0,2) + CI(1,0) * CI(0,2);
        CI(1,2) = CI(1,2) - CI(1,0) * CR(0,2) - CR(1,0) * CI(0,2);
        }

    }

    // The system is now in upper triangular form.

    ////////////////////////////////////////////////////////////////////////////
    // Backward substitution.
    ////////////////////////////////////////////////////////////////////////////

    double xr1, xi1, xr2, xi2;
    double beta = 1.0;

    // xr2 + i xi2 := ( CR(1,2) + i CI(1,2) ) / ( CR(1,1) + i CI(1,1) )
    // If denominator is too small, perturb.
    if (fabs(CR(1,1)) + fabs(CI(1,1)) < smin) {
        CR(1,1) = smin;
        CI(1,1) = 0.0;
        info = 1;
    }
    dladiv(CR(1,2), CI(1,2), CR(1,1), CI(1,1), &xr2, &xi2, &beta);

    *scale = (*scale) * beta;
    xr1 = CR(0,2);
    xi1 = CI(0,2);
    xr1 = xr1 - CR(0,1) * xr2 + CI(0,1) * xi2;
    xi1 = xi1 - CI(0,1) * xr2 - CR(0,1) * xi2;

    if (pivot_col == 1) {
        b_re[0] = xr2; b_im[0] = xi2;
        b_re[1] = xr1; b_im[1] = xi1;
    }
    else {
        b_re[0] = xr1; b_im[0] = xi1;
        b_re[1] = xr2; b_im[1] = xi2;
    }

    return info;

#undef CR
#undef CI
#undef T
}


#ifdef STARNEIG_ENABLE_INTEGER_SCALING
int starneig_eigvec_std_solve_2x2_cmplx_system(
    double smin,
    const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    int *restrict const scale)
{
    // Local scaling factor.
    double phi = 1.0;

    int info = solve_2x2_cmplx_system_internal(
        smin, T, ldT, lambda_re, lambda_im, b_re, b_im, &phi);

    // Convert double-precision scaling factor to int scaling factor.
    *scale = ilogb(phi);

    return info;
}

#else

int starneig_eigvec_std_solve_2x2_cmplx_system(
    double smin,
    const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    double *restrict const scale)
{
    return solve_2x2_cmplx_system_internal(
        smin, T, ldT, lambda_re, lambda_im, b_re, b_im, scale);
}

#endif


#undef NO_RESCALE
#undef RESCALE
#undef REAL
#undef CMPLX
