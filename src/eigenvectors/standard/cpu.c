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

#include <starneig_config.h>
#include <starneig/configuration.h>

#include "typedefs.h"
#include "cpu.h"
#include "robust.h"

#include "../../common/common.h"
#include <starpu.h>
#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>


static double vec_real_infnorm(int n, const double *x)
{
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double abs = fabs(x[i]);
        if (abs > norm)
            norm = abs;
    }

    return norm;
}

static double vec_cmplx_infnorm(int n, const double *x_re, const double *x_im)
{
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        // Compute len = sqrt(x_re[i]*x_re[i]+x_im[i]*x_im[i]) robustly.
        double maxabs = MAX(fabs(x_re[i]), fabs(x_im[i]));
        double len = maxabs*sqrt(  (x_re[i]/maxabs)*(x_re[i]/maxabs)
                                 + (x_im[i]/maxabs)*(x_im[i]/maxabs));

        if (len > norm)
            norm = len;
    }

    return norm;
}

static double mat_infnorm(int m, int n, const double *A, int ldA)
{
#define A(i,j) A[(i) + (j) * (size_t)ldA]

    double *rowsums = calloc(m, sizeof(double));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; i++)
            rowsums[i] += fabs(A(i,j));

    double norm = rowsums[0];
    for (int i = 1; i < m; i++)
        if (rowsums[i] > norm)
            norm = rowsums[i];

    free(rowsums);

    return norm;

#undef A
}


static void find_max(
    int num_rows, int num_selected, int n,
    const double *restrict const X, int ldX,
    const int *restrict const lambda_type, const int *restrict const selected,
    double *restrict emax)
{
    int si = num_selected-1;
    for (int ki = n-1; ki >= 0; ki--) {
        if (!selected[ki]) {
            // Proceed with the next eigenvalue.
            if (lambda_type[ki] == 1) { // CMPLX
                // A complex conjugate pair of eigenvalues is not selected, so
                // skip the next diagonal entry.
                ki--;
            }
        }
        else { // ki-th eigenvalue is selected
            if (lambda_type[ki] == 0) { // REAL
                // Locate eigenvector to ki-th eigenvalue.
               const double *X_re = X+si*(size_t)ldX;

               // Find max entry.
               emax[si] = vec_real_infnorm(num_rows, X_re);
               si--;
            }
            else { // CMPLX
                // Locate real and imaginary part of complex eigenvector for
                // complex conjugate pair of eigenvalues (ki, ki-1).
                const double *X_re = X+(si-1)*(size_t)ldX;
                const double *X_im = X+si*(size_t)ldX;

                // Find max entry.
                emax[si] = 0.0;
                for (int i = 0; i < num_rows; i++)
                    if (fabs(X_re[i])+fabs(X_im[i]) > emax[si])
                        emax[si] = fabs(X_re[i])+fabs(X_im[i]);

                // Duplicate max entry for real and imaginary column.
                emax[si-1] = emax[si];

                ki--;
                si -= 2;
            }
        }
    }
}




void starneig_eigvec_std_unify_scaling(int num_tiles, int *first_row, int *first_col,
    scaling_t *restrict scales,
    double *restrict X, int ldX,
    const int *restrict lambda_type, const int *restrict selected)
{
    // TODO: Is it possible to add a #pragma parallel for here?

    const int num_selected = first_col[num_tiles];

    //
    // Compute the most constraining scaling factor.
    //
    scaling_t *smin = (scaling_t *) malloc(num_selected*sizeof(scaling_t));
    starneig_eigvec_std_init_scaling_factor(num_selected, smin);

    starneig_eigvec_std_find_smallest_scaling(num_tiles, num_selected, scales, smin);

#ifndef STARNEIG_ENABLE_INTEGER_SCALING

    //
    // Check if the range of double-precision scaling factors was sufficient
    // and replace flushed entries with 1/Omega to avoid NaNs.
    //

    for (int i = 0; i < num_selected; i++) {
        if (smin[i] == 0.0) {
            if (lambda_type[i] == 1) { // CMPLX
                starneig_warning("Scaling factor of complex eigenvector "
                                 "X(:,%d:%d) was flushed. Rerun with integer "
                                 "scaling factors.", i, i+1);
                smin[i] = DBL_MIN;
                smin[i+1] = DBL_MIN;
                i++;
            }
            else { // REAL
                starneig_warning("Scaling factor of real eigenvector X(:,%d) "
                                 "was flushed. Rerun with integer "
                                 "scaling factors.", i);
                smin[i] = DBL_MIN;

            }
        }
    }

    for (int i = 0; i < (size_t)num_tiles*num_selected;i++)
        if (scales[i] == 0.0)
            scales[i] = DBL_MIN;

#endif

    double *tmp = (double *) malloc(num_selected * sizeof(double));
    double *emax = (double *) malloc(num_selected * sizeof(double));
    memset(emax, 0.0, num_selected * sizeof(double));

#define scales(col, tilerow) scales[(col) + (tilerow) * (size_t)num_selected]

    for (int blkj = 0; blkj < num_tiles; blkj++) {
        for (int blki = 0; blki <= blkj; blki++) {
            const int num_rows = first_row[blki+1]-first_row[blki];
            const int num_sel = first_col[blkj+1]-first_col[blkj];
            const int num_cols = first_row[blkj+1]-first_row[blkj];

            double *tile = X+(size_t)first_col[blkj]*ldX+first_row[blki];
            const int *lambda_type_tile = lambda_type+first_row[blkj];
            const int *selected_tile = selected+first_row[blkj];
            double *tmp_tile = tmp+first_col[blkj];
            memset(tmp_tile, 0.0, num_sel*sizeof(double));

            find_max(num_rows, num_sel, num_cols,
                tile, ldX, lambda_type_tile, selected_tile, tmp_tile);

            // Reduce to maximum normalization factor.
            for (int j = first_col[blkj]; j < first_col[blkj+1]; j++) {
                // Compute normalization factor simulating consistent scaling.
               double s = starneig_eigvec_std_compute_upscaling(smin[j], scales(j, blki));
               emax[j] = MAX(s * tmp[j], emax[j]);
            }
        }

        // Apply scaling.
        for (int blki = 0; blki <= blkj; blki++) {
            const int num_rows = first_row[blki+1]-first_row[blki];
            const int num_cols = first_col[blkj+1]-first_col[blkj];

            // Scale column.
            for (int j = 0; j < num_cols; j++) {
                // The current column index.
                size_t col = (size_t)first_col[blkj]+j;

                // The current column.
                double *x = X+col*ldX+first_row[blki];
                double s =
                    starneig_eigvec_std_compute_upscaling(smin[col], scales(col, blki));

                // Avoid oo.
                if (isinf(s))
                    s = DBL_MIN;

                for (int i = 0; i < num_rows; i++)
                    x[i] = (s*x[i])/emax[col];
            }
        }
    }

    free(tmp);
    free(emax);
#undef scales
}

void starneig_eigvec_std_cpu_bound(void *buffers[], void *cl_args)
{
    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[0]);
    int m = STARPU_MATRIX_GET_NX(buffers[0]);
    int n = STARPU_MATRIX_GET_NY(buffers[0]);

    double *tnorm = (double *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    double ub = mat_infnorm(m, n, T, ldT);
    *tnorm = ub;
}



static void backsolve(
    int n, const double *restrict T, int ldT, double tnorm,
    double *restrict X, int ldX, scaling_t *restrict const scales,
    double *restrict Xnorms, const int *restrict lambda_type,
    const int *restrict selected, int num_selected,
    double smlnum, int *restrict infos)
{
#define T(i,j) T[(i) + (j) * (size_t)ldT]

    // Solve T * X = X * \Lambda.

    // The i-th selected eigenvalue.
    int si = num_selected-1;

    // Loop over eigenvalues from bottom to top.
    for (int ki = n-1; ki >= 0; ki--) {
        if (!selected[ki]) {
            // Proceed with the next eigenvalue.
            if (lambda_type[ki] == 1) { // CMPLX
                // A complex conjugate pair of eigenvalues is not selected,
                // so skip the next diagonal entry.
                ki--;
            }
        }
        else { // ki-th eigenvalue is selected

            // Error flag.
            int info = 0;

            if (lambda_type[ki] == 0) {
                // Compute a real right eigenvector.
                double lambda = T(ki,ki);

                // Locate eigenvector to ki-th eigenvalue.
                double *X_re = X+si*(size_t)ldX;

                // Locate corresponding scaling factor.
                scaling_t *beta = scales+si;

                // Critical threshold to detect unsafe divisions.
                const double smin = MAX(DBL_EPSILON/2*fabs(lambda), smlnum);

                // Form right-hand side.
                X_re[ki] = 1.0;
                for (int i = 0; i < ki; i++)
                    X_re[i] = -T(i, ki);
                for (int i = ki+1; i < n; i++)
                    X_re[i] = 0.0;

                // Compute norm of entire vector.
                double norm = vec_real_infnorm(ki+1, X_re);

                // Solve the upper quasi-triangular system.
                // (T(0:ki-1,0:ki-1) - lambda I) \ X.
                for (int j = ki-1; j >= 0; j--) {
                    // if next block is 1-by-1 diagonal block:
                    if (lambda_type[j] == 0) { // REAL
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_1x1_real_system(
                            smin, T(j,j), lambda, X_re+j, &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale remaining parts of the vector.
                        starneig_eigvec_std_scale(j, X_re, &phi);
                        starneig_eigvec_std_scale(ki-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the linear update.
                        phi =
                            starneig_eigvec_std_protect_update(tnorm, fabs(X_re[j]), norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply the scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(ki+1, X_re, &phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++)
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];

                        // Recompute norm excluding the entries j:n.
                        norm = vec_real_infnorm(j, X_re);
                    }
                    // if next block is 2-by-2 diagonal block:
                    else {
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_2x2_real_system(smin,
                            &T(j-1,j-1), ldT, lambda, &X_re[j-1], &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale remaining parts of vector.
                        starneig_eigvec_std_scale(j-1, X_re, &phi);
                        starneig_eigvec_std_scale(ki-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the first linear update.
                        phi = starneig_eigvec_std_protect_update(
                            tnorm, fabs(X_re[j-1]), norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply the scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(ki+1, X_re, &phi);

                        // Now it is safe to execute the first linear update.
                        for (int i = 0; i < j-1; i++)
                            X_re[i] = X_re[i]-T(i,j-1)*X_re[j-1];

                        // Recompute norm excluding the entries j:n.
                        norm = vec_real_infnorm(j, X_re);

                        // Protect against overflow in the second linear update.
                        phi =
                            starneig_eigvec_std_protect_update(tnorm, fabs(X_re[j]), norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply the scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(ki+1, X_re, &phi);

                        // Now it is safe to execute the second linear update.
                        for (int i = 0; i < j-1; i++)
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];

                        // Recompute norm excluding the entries j-1:n.
                        norm = vec_real_infnorm(j-1, X_re);

                        // We processed a 2-by-2 block, so skip the next entry.
                        j--;
                    }
                }

                // The ki-th real eigenvector has been computed. Recompute norm.
                Xnorms[si] = vec_real_infnorm(ki+1, X_re);

                // Record error flag.
                infos[si] = info;

                // This eigenvector spans 1 column. Update selected counter.
                si--;
            }
            else { // lambda_type[ki] == CMPLX
                // Locate real and imaginary part of complex eigenvector for
                // complex conjugate pair of eigenvalues (ki, ki-1).
                double *X_re = X+(si-1)*(size_t)ldX;
                double *X_im = X+si*(size_t)ldX;

                // Locate corresponding scaling factor, one per complex pair.
                scaling_t *beta = scales+si;

                // Compute eigenvalue as lambda = lambda_re + i * lambda_im.
                // A 2-by-2 block in canonical Schur form has the shape
                // [ T(ki-1,ki-1) T(ki-1,ki) ] = [ a b ]  or [ a -b ]
                // [ T(ki, ki-1)  T(ki, ki)  ]   [-c a ]     [ c  a ].
                double lambda_re = T(ki,ki);
                double lambda_im =
                    sqrt(fabs(T(ki,ki-1)))*sqrt(fabs(T(ki-1,ki)));

                // Critical threshold to detect unsafe divisions.
                const double smin = MAX(
                    DBL_EPSILON/2*(fabs(lambda_re)+fabs(lambda_im)), smlnum);

                // Form right-hand side.
                if (fabs(T(ki-1,ki)) >= fabs(T(ki,ki-1))) {
                    X_re[ki-1] = 1.0;
                    X_im[ki] = lambda_im/T(ki-1,ki);
                }
                else {
                    X_re[ki-1] = -lambda_im/T(ki,ki-1);
                    X_im[ki] = 1.0;
                }
                X_re[ki] = 0.0;
                X_im[ki-1] = 0.0;
                for (int i = 0; i < ki-1; i++) {
                    X_re[i] = -X_re[ki-1]*T(i,ki-1);
                    X_im[i] = -X_im[ki]*T(i,ki);
                }
                for (int i = ki+1; i < n; i++) {
                    X_re[i] = 0.0;
                    X_im[i] = 0.0;
                }

                // Compute norm of the entire vector.
                double norm = vec_cmplx_infnorm(ki+1, X_re, X_im);

                // Solve the upper quasi-triangular system
                // (T(0:ki-2,0:ki-2) - (lambda * I) \ (X_re + i * X_im).

                // Loop over triangular matrix above the eigenvalue. Note ki-2!
                for (int j = ki-2; j >= 0; j--) {
                    // If next block is 1-by-1 diagonal bock:
                    if (lambda_type[j] == 0) { // REAL
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_1x1_cmplx_system(
                            smin, T(j,j), lambda_re, lambda_im,
                            X_re+j, X_im+j, &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale the remaining parts of the vector.
                        starneig_eigvec_std_scale(j, X_re, &phi);
                        starneig_eigvec_std_scale(ki-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_scale(j, X_im, &phi);
                        starneig_eigvec_std_scale(ki-(j+1), X_im+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the linear update.
                        double absmax = MAX(fabs(X_re[j]), fabs(X_im[j]));
                        phi = starneig_eigvec_std_protect_update(tnorm, absmax, norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(ki+1, X_re, &phi);
                        starneig_eigvec_std_scale(ki+1, X_im, &phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++) {
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];
                            X_im[i] = X_im[i]-T(i,j)*X_im[j];
                        }

                        // Recompute norm excluding the entries j:n.
                        norm = vec_cmplx_infnorm(j, X_re, X_im);
                    }
                    // If next block is 2-by-2 diagonal block:
                    else {
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_2x2_cmplx_system(
                            smin, &T(j-1,j-1), ldT, lambda_re, lambda_im,
                            X_re+j-1, X_im+j-1, &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale the remaining parts of the vector.
                        starneig_eigvec_std_scale(j-1, X_re, &phi);
                        starneig_eigvec_std_scale(ki-j, X_re+(j+1), &phi);
                        starneig_eigvec_std_scale(j-1, X_im, &phi);
                        starneig_eigvec_std_scale(ki-j, X_im+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the first linear update.
                        double absmax = MAX(fabs(X_re[j-1]), fabs(X_im[j-1]));
                        phi = starneig_eigvec_std_protect_update(tnorm, absmax, norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(ki+1, X_re, &phi);
                        starneig_eigvec_std_scale(ki+1, X_im, &phi);

                        // Now it is safe to execute the first linear update.
                        for (int i = 0; i < j-1; i++) {
                            X_re[i] = X_re[i]-T(i,j-1)*X_re[j-1];
                            X_im[i] = X_im[i]-T(i,j-1)*X_im[j-1];
                        }

                        // Recompute norm excluding the entries j+1:n.
                        norm = vec_cmplx_infnorm(j+1, X_re, X_im);

                        // Protect against overflow in the second linear update.
                        absmax = MAX(fabs(X_re[j]), fabs(X_im[j]));
                        phi = starneig_eigvec_std_protect_update(tnorm, absmax, norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(ki+1, X_re, &phi);
                        starneig_eigvec_std_scale(ki+1, X_im, &phi);

                        // Now it is safe to execute the second linear update.
                        for (int i = 0; i < j-1; i++) {
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];
                            X_im[i] = X_im[i]-T(i,j)*X_im[j];
                        }

                        // Recompute norm excluding the entries j:n.
                        norm = vec_cmplx_infnorm(j-1, X_re, X_im);

                        // We processed a 2-by-2 block, so skip the next entry.
                        j--;
                    }
                }

                // Note that the second column of a complex conjugate pair is
                // never allocated or computed. Obtaining it is left to the
                // user. If the positions si - 1, si mark a 2-by-2 block, then
                // the eigenvector corresponding to lambda = alpha + i beta
                // is X(:, si - 1) + i * X(:, si).
                // The complex conjugate eigenvector corresponding to
                // lambda = alpha - i beta can be derived as
                // conj(X) := X(:, si -1) - i * X(:, si).

                // The real part and the imaginary part are scaled alike.
                scales[si-1] = scales[si];

                // The ki-th complex eigenvector has been computed. Recompute
                // norm of the complex eigenvector.
                Xnorms[si] = vec_cmplx_infnorm(ki+1, X_re, X_im);
                Xnorms[si-1] = Xnorms[si];

                // Record error flag.
                infos[si] = info;
                infos[si-1] = info;

                // We processed a complex conjugate pair of eigenvalues,
                // so skip the next entry.
                ki--;

                // This eigenvector spans 2 columns. Update selected counter.
                si -= 2;
            }
        }
    }

    // Note that the eigenvectors are not normalized.

#undef T


}




void starneig_eigvec_std_cpu_backsolve(void *buffers[], void *cl_args)
{
    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int n = STARPU_MATRIX_GET_NX(buffers[0]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[0]);

    double *ubT = (double *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    double tnorm = *ubT;

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int num_selected = STARPU_MATRIX_GET_NY(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    scaling_t *scales = (scaling_t *) STARPU_VECTOR_GET_PTR(buffers[3]);

    double *Xnorms = (double *) STARPU_VECTOR_GET_PTR(buffers[4]);

    int *lambda_type = (int *) STARPU_VECTOR_GET_PTR(buffers[5]);

    int *selected = (int *) STARPU_VECTOR_GET_PTR(buffers[6]);

    int *infos = (int *) STARPU_VECTOR_GET_PTR(buffers[7]);

    double smlnum;
    starpu_codelet_unpack_args(cl_args, &smlnum);

    backsolve(n, T, ldT, tnorm, X, ldX, scales, Xnorms, lambda_type,
        selected, num_selected, smlnum, infos);
}


void starneig_eigvec_std_cpu_solve(void *buffers[], void *cl_args)
{
    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int n = STARPU_MATRIX_GET_NX(buffers[0]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[0]);

    double *ubT = (double *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    double tnorm = *ubT;

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int num_selected = STARPU_MATRIX_GET_NY(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    scaling_t *scales = (scaling_t *) STARPU_VECTOR_GET_PTR(buffers[3]);

    double *Xnorms = (double *) STARPU_VECTOR_GET_PTR(buffers[4]);

    double *lambda = (double *) STARPU_VECTOR_GET_PTR(buffers[5]);
    int num_rhs = STARPU_VECTOR_GET_NX(buffers[5]);

    int *lambda_type = (int *) STARPU_VECTOR_GET_PTR(buffers[6]);

    int *selected = (int *) STARPU_VECTOR_GET_PTR(buffers[7]);

    int *diag_type = (int *) STARPU_VECTOR_GET_PTR(buffers[8]);

    int *infos = (int *) STARPU_VECTOR_GET_PTR(buffers[9]);

    double smlnum;
    starpu_codelet_unpack_args(cl_args, &smlnum);

#define T(i,j) T[(i) + (j) * (size_t)ldT]
    // The i-th selected eigenvalue.
    int si = num_selected-1;

    // Loop over eigenvalues.
    for (int k = num_rhs-1; k >= 0; k--) {
        if (!selected[k]) {
            // Proceed with the next eigenvalue.
            if (lambda_type[k] == 1) { // CMPLX
                // A complex conjugate pair of eigenvalues is not selected,
                // so skip the next diagonal entry.
                k--;
            }
        }
        else { // k-th eigenvalue is selected

            // Error flag.
            int info = 0;

            if (lambda_type[k] == 0) { // REAL
                // Locate eigenvector to k-th eigenvalue.
                double *X_re = X+si*(size_t)ldX;

                // Locate corresponding scaling factor.
                scaling_t *beta = scales+si;

                // Compute norm of entire vector.
                double norm = vec_real_infnorm(n, X_re);

                // Critical threshold to detect unsafe divisions.
                const double smin = MAX(DBL_EPSILON/2*fabs(lambda[k]), smlnum);


                for (int j = n-1; j >= 0; j--) {
                    // if next block is 1-by-1 diagonal block:
                    if (diag_type[j] == 0) { // REAL
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_1x1_real_system(
                            smin, T(j,j), lambda[k], X_re+j, &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale remaining parts of vector.
                        starneig_eigvec_std_scale(j, X_re, &phi);
                        starneig_eigvec_std_scale(n-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the linear update.
                        phi =
                            starneig_eigvec_std_protect_update(tnorm, fabs(X_re[j]), norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply the scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(n, X_re, &phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++)
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];

                        // Recompute norm excluding the entries j:n.
                        norm = vec_real_infnorm(j, X_re);
                    }
                    // if next block is 2-by-2 block:
                    else {
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_2x2_real_system(smin,
                            &T(j-1,j-1), ldT, lambda[k], &X_re[j-1], &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale remaining parts of vector.
                        starneig_eigvec_std_scale(j-1, X_re, &phi);
                        starneig_eigvec_std_scale(n-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect first linear update against overflow.
                        phi = starneig_eigvec_std_protect_update(
                            tnorm, fabs(X_re[j-1]), norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply the scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(n, X_re, &phi);

                        // Now it is safe to execute the linear udpate.
                        for (int i = 0; i < j-1; i++)
                            X_re[i] = X_re[i]-T(i,j-1)*X_re[j-1];

                        // Recompute norm excluding the entries j:n.
                        norm = vec_real_infnorm(j, X_re);

                        // Protect second linear update against overflow.
                        phi =
                            starneig_eigvec_std_protect_update(tnorm, fabs(X_re[j]), norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply the scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(n, X_re, &phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j - 1; i++)
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];

                        // Recompute norm excluding the entries j-1:n.
                        norm = vec_real_infnorm(j-1, X_re);

                        // We processed a 2-by-2 block, so skip the next
                        // diagonal entry.
                        j--;
                    }
                }

                // The k-th real eigenvector has been computed.
                // Recompute norm.
                Xnorms[si] = vec_real_infnorm(n, X_re);

                // Record error flag.
                infos[si] = info;

                // This eigenvalue spans 1 column. Update selected counter.
                si--;
            }
            else { // lambda_type[k] == CMPLX
                // Locate real and imaginary part of complex eigenvector
                // for complex conjugate pair of eigenvalues (k, k-1).
                double *X_re = X+(si-1)*(size_t)ldX;
                double *X_im = X+si*(size_t)ldX;

                // The eigenvalue is lambda = lambda_re+i*lambda_im.
                double lambda_re = lambda[k-1];
                double lambda_im = fabs(lambda[k]);

                // Locate corresponding scaling factor.
                scaling_t *beta = scales+si;

                // Compute norm of entire vector.
                double norm = vec_cmplx_infnorm(n, X_re, X_im);

                // Critical threshold to detect unsafe divisions.
                const double smin = MAX(
                    DBL_EPSILON/2*(fabs(lambda_re)+fabs(lambda_im)), smlnum);

                for (int j = n-1; j >= 0; j--) {
                    // if the next block is 1-by-1 diagonal block:
                    if (diag_type[j] == 0) { // REAL
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_1x1_cmplx_system(smin, T(j,j),
                            lambda_re, lambda_im, X_re+j, X_im+j, &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale the remaining parts of the 2 columns.
                        starneig_eigvec_std_scale(j, X_re, &phi);
                        starneig_eigvec_std_scale(n-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_scale(j, X_im, &phi);
                        starneig_eigvec_std_scale(n-(j+1), X_im+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the linear update.
                        double absmax = MAX(fabs(X_re[j]), fabs(X_im[j]));
                        phi = starneig_eigvec_std_protect_update(tnorm, absmax, norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(n, X_re, &phi);
                        starneig_eigvec_std_scale(n, X_im, &phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++) {
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];
                            X_im[i] = X_im[i]-T(i,j)*X_im[j];
                        }

                        // Recompute norm excluding the entries j:n.
                        norm = vec_cmplx_infnorm(j, X_re, X_im);
                    }
                    // if next block is 2-by-2 diagonal block:
                    else {
                        scaling_t phi;
                        starneig_eigvec_std_init_scaling_factor(1, &phi);
                        info |= starneig_eigvec_std_solve_2x2_cmplx_system(smin,
                            &T(j-1,j-1), ldT, lambda_re, lambda_im,
                            X_re+j-1, X_im+j-1, &phi);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Scale remaining parts of vector.
                        starneig_eigvec_std_scale(j-1, X_re, &phi);
                        starneig_eigvec_std_scale(n-(j+1), X_re+(j+1), &phi);
                        starneig_eigvec_std_scale(j-1, X_im, &phi);
                        starneig_eigvec_std_scale(n-(j+1), X_im+(j+1), &phi);
                        starneig_eigvec_std_update_norm(&norm, phi);

                        // Protect against overflow in the first linear update.
                        double absmax = MAX(fabs(X_re[j-1]), fabs(X_im[j-1]));
                        phi = starneig_eigvec_std_protect_update(tnorm, absmax, norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(n, X_re, &phi);
                        starneig_eigvec_std_scale(n, X_im, &phi);

                        // Now it is safe to execute the first linear update.
                        for (int i = 0; i < j-1; i++) {
                            X_re[i] = X_re[i]-T(i,j-1)*X_re[j-1];
                            X_im[i] = X_im[i]-T(i,j-1)*X_im[j-1];
                        }

                        // Recompute norm excluding the entries j+1:n.
                        norm = vec_cmplx_infnorm(j+1, X_re, X_im);

                        // Protect against overflow in the second linear update.
                        absmax = MAX(fabs(X_re[j]), fabs(X_im[j]));
                        phi = starneig_eigvec_std_protect_update(tnorm, absmax, norm);
                        starneig_eigvec_std_update_global_scaling(beta, phi);

                        // Apply scaling to the whole eigenvector.
                        starneig_eigvec_std_scale(n, X_re, &phi);
                        starneig_eigvec_std_scale(n, X_im, &phi);

                        // Now it is safe to execute the second linear update.
                        for (int i = 0; i < j-1; i++) {
                            X_re[i] = X_re[i]-T(i,j)*X_re[j];
                            X_im[i] = X_im[i]-T(i,j)*X_im[j];
                        }

                        // Recompute norm excluding the entries j-1:n.
                        norm = vec_cmplx_infnorm(j-1, X_re, X_im);

                        // We processed a 2-by-2 block, so skip the next
                        // diagonal entry.
                        j--;
                    }
                }

                // Note that the second column of a complex conjugate pair is
                // never allocated or computed. Obtaining it is left to the
                // user. If the positions si - 1, si mark a 2-by-2 block, then
                // the eigenvector corresponding to lambda = alpha + i beta
                // is X(:, si - 1) + i * X(:, si).
                // The complex conjugate eigenvector corresponding to
                // lambda = alpha - i beta can be derived as
                // conj(X) := X(:, si -1) - i * X(:, si).

                // The real part and the imaginary part are scaled alike.
                scales[si-1] = scales[si];

                // The ki-th complex eigenvector has been computed.
                // Recompute norm of the complex eigenvector.
                Xnorms[si] = vec_cmplx_infnorm(n, X_re, X_im);
                Xnorms[si-1] = Xnorms[si];

                // Record error flag.
                infos[si] = info;
                infos[si-1] = info;

                // We processed a complex conjugate pair of eigenvalues,
                // so skip the next entry.
                k--;

                // This eigenvector spans 2 cols. Update selected counter.
                si -= 2;
            }
        }
    }

#undef T
}



void starneig_eigvec_std_cpu_update(void *buffers[], void *cl_args)
{
    // T is n-by-m.
    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[0]);
    int n = STARPU_MATRIX_GET_NX(buffers[0]);
    int m = STARPU_MATRIX_GET_NY(buffers[0]);

    double *ubT = (double *) STARPU_VARIABLE_GET_PTR(buffers[1]);
    double tnorm = *ubT;

    double *Xin = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int num_rhs = STARPU_MATRIX_GET_NY(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    scaling_t *Xscales = (scaling_t *) STARPU_VECTOR_GET_PTR(buffers[3]);

    double *Xinnorms = (double *) STARPU_VECTOR_GET_PTR(buffers[4]);

    double *Y = (double *) STARPU_MATRIX_GET_PTR(buffers[5]);
    int ldY = STARPU_MATRIX_GET_LD(buffers[5]);

    scaling_t *Yscales = (scaling_t *) STARPU_VECTOR_GET_PTR(buffers[6]);

    double *Ynorms = (double *) STARPU_VECTOR_GET_PTR(buffers[7]);

    int *lambda_type = (int *) STARPU_VECTOR_GET_PTR(buffers[8]);

    // Pointer to X - either a copy or the original memory.
    double *X;

    // Pointer to norms of X - either a copy or the original memory.
    double *Xnorms;

    // Status flag if X has to be rescaled.
    int rescale_X = 0;

    // Status flag if Y has to be rescaled.
    int rescale_Y = 0;

    // Indicator if xinnorms has to be rescaled. Note that this only affects
    // consistency scaling and not overflow protection.
    int rescale_xnorms = 0;

    // Workspace to store locally computed scaling factors.
    scaling_t tmp_scales[num_rhs];


    //
    // Compute scaling factor.
    //

    for (int k = 0; k < num_rhs; k++) {
        if (Yscales[k] < Xscales[k]) {
            rescale_xnorms = 1;
            break;
        }
    }

    if (rescale_xnorms) {
        // As X is read-only, copy xnorms.
        Xnorms = (double *) malloc(num_rhs * sizeof(double));
        memcpy(Xnorms, Xinnorms, num_rhs * sizeof(double));

        // Simulate the consistency scaling.
        for (int k = 0; k < num_rhs; k++) {
            if (Yscales[k] < Xscales[k]) {
                // The common scaling factor is Yscales[k].
                const double s =
                    starneig_eigvec_std_compute_upscaling(Yscales[k], Xscales[k]);

                // Mark X for scaling. Physical rescaling is deferred.
                rescale_X = 1;

                // Update norm.
                Xnorms[k] = s*Xinnorms[k];
            }
            else if (Xscales[k] < Yscales[k]) {
                // The common scaling factor is Xscales[k].
                const double s =
                    starneig_eigvec_std_compute_upscaling(Xscales[k], Yscales[k]);

                // Mark Y for scaling. Physical rescaling is deferred.
                rescale_Y = 1;

                // Update norm: norm(s * Y) = s * norm(Y).
                Ynorms[k] = s*Ynorms[k];
            }
        }
    }
    else { // !rescale_xnorms
        // No changes to Xinnorms necessary. Operate on original memory.
        Xnorms = Xinnorms;

        // Xnorms does not need scaling, but Ynorms may do.
        for (int k = 0; k < num_rhs; k++) {
            if (Xscales[k] < Yscales[k]) {
                // The common scaling factor is Xscales[k].
                const double s =
                    starneig_eigvec_std_compute_upscaling(Xscales[k], Yscales[k]);

                // Mark Y for scaling. Phyiscal rescaling is deferred.
                rescale_Y = 1;

                // Update norm: norm(s * Y) = s * norm(Y).
                Ynorms[k] = s*Ynorms[k];
            }
        }
    }



    //
    // Apply scaling.
    //

    // Status flag if update is safe to execute or if rescaling is required.
    int rescale;

    // Compute scaling factors needed to survive the linear update.
    rescale = starneig_eigvec_std_protect_multi_rhs_update(
        Xnorms, num_rhs, tnorm, Ynorms, lambda_type, tmp_scales);

    if (rescale) {
        rescale_X = 1;
        rescale_Y = 1;
    }

    // If X has to be rescaled, take a copy of X and do scaling on the copy.
    if (rescale_X) {
        X = (double *) malloc((size_t)ldX * num_rhs * sizeof(double));

        for (int k = 0; k < num_rhs; k++) {
            if (Yscales[k] < Xscales[k]) {
                // Copy X and simultaneously rescale.
                const double s = starneig_eigvec_std_compute_combined_upscaling(
                    Yscales[k], Xscales[k], tmp_scales[k]);
                for (int i = 0; i < m; i++)
                    X[(size_t)ldX*k+i] = s*Xin[(size_t)ldX*k+i];
            }
            else if (Xscales[k] < Yscales[k]) {
                // Copy X and simultaneously rescale with robust update factor.
                const double s = starneig_eigvec_std_convert_scaling(tmp_scales[k]);
                for (int i = 0; i < m; i++)
                    X[(size_t)ldX*k+i] = s*Xin[(size_t)ldX*k+i];
            }
            else {
                // Xscales[k] == Yscales[k].

                // Copy X and simultaneously rescale with robust update factor.
                const double s = starneig_eigvec_std_convert_scaling(tmp_scales[k]);
                for (int i = 0; i < m; i++)
                    X[(size_t)ldX*k+i] = s*Xin[(size_t)ldX*k+i];
            }
        }
    }
    else { // !rescale_X
        // No changes to X necessary. Operate on original memory.
        X = Xin;
    }


    // If Y has to be rescaled, directly modify Y.
    if (rescale_Y) {
        for (int k = 0; k < num_rhs; k++) {
            if (Yscales[k] < Xscales[k]) {
                // The common scaling factor is Yscales[k]. Rescale Y with
                // robust update factor, if necessary.
                starneig_eigvec_std_scale(n, Y+(size_t)ldY*k, tmp_scales+k);
            }
            else if (Xscales[k] < Yscales[k]) {
                // The common scaling factor is Xscales[k]. Combine with
                // robust update scaling factor.
                const double s = starneig_eigvec_std_compute_combined_upscaling(
                    Xscales[k], Yscales[k], tmp_scales[k]);
                for (int i = 0; i < n; i++)
                    Y[(size_t)ldY*k+i] = s*Y[(size_t)ldY*k+i];
            }
            else {
                // Xscales[k] == Yscales[k].

                // Rescale Y with robust update factor, if necessary.
                starneig_eigvec_std_scale(n, Y+(size_t)ldY*k, tmp_scales+k);
            }
        }
    }


    //
    // Update global scaling of Y.
    //

#ifdef STARNEIG_ENABLE_INTEGER_SCALING
    for (int k = 0; k < num_rhs; k++)
        Yscales[k] = MIN(Yscales[k], Xscales[k])+tmp_scales[k];
#else
    for (int k = 0; k < num_rhs; k++)
        Yscales[k] = MIN(Yscales[k], Xscales[k])*tmp_scales[k];
#endif


    //
    // Compute update.
    //

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        n, num_rhs, m, -1.0, T, ldT, X, ldX, 1.0, Y, ldY);


    //
    // Recompute norms.
    //

    for (int k = 0; k < num_rhs; k++) {
        if (lambda_type[k] == 1) { // CMPLX
            // We store only one scaling factor per complex eigenvector pair.
            // So interpret columns as real and imaginary part.
            const double *Y_re = Y+(size_t)k*ldY;
            const double *Y_im = Y+(size_t)(k+1)*ldY;

            Ynorms[k] = vec_cmplx_infnorm(n, Y_re, Y_im);

            // Duplicate norm for real and imaginary column.
            Ynorms[k+1] = Ynorms[k];

            k++;
        }
        else { // lambda_type[k] == REAL
            Ynorms[k] = vec_real_infnorm(n, Y+(size_t)k*ldY);
        }
    }


    //
    // Clean up.
    //

    if (rescale_X)
        free(X);

    if (rescale_xnorms)
        free(Xnorms);
}



void starneig_eigvec_std_cpu_backtransform(void *buffers[], void *cl_args)
{
    double *Q = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldQ = STARPU_MATRIX_GET_LD(buffers[0]);

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY = STARPU_MATRIX_GET_LD(buffers[2]);
    int m = STARPU_MATRIX_GET_NX(buffers[2]);
    int n = STARPU_MATRIX_GET_NY(buffers[2]);

    int k;
    starpu_codelet_unpack_args(cl_args, &k);

    //   Yij  :=   Qi:  *   X:j
    // (m x n)   (m x k)  (k x n)

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, 1.0, Q, ldQ, X, ldX, 0.0, Y, ldY);

}
