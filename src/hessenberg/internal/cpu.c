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
#include "cpu.h"
#include "../common/common.h"
#include "../common/sanity.h"
#include "../common/tiles.h"
#include "../common/trace.h"
#include <starpu.h>
#include <starpu_scheduler.h>
#include <hwloc.h>
#include <cblas.h>
#include <omp.h>

void starneig_hessenberg_cpu_prepare_column(
    void *buffers[], void *cl_args)
{
    // LAPACK subroutine that generates a real elementary reflector H
    extern void dlarfg_(int const *, double *, double *, int const *, double *);

    int i; // the index of the currect column inside the panel
    struct range_packing_info v_pi;
    starpu_codelet_unpack_args(cl_args, &i, &v_pi);

    int k = 0;

    double *Y = NULL; int ldY = 0;
    if (0 < i) {
        Y = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
        ldY = STARPU_MATRIX_GET_LD(buffers[k]);
        k++;
    }

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[k]);
    int m = STARPU_MATRIX_GET_NX(buffers[k]);
    int nb = STARPU_MATRIX_GET_NY(buffers[k]);
    k++;

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    double *P = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldP = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    // an intemediate vector interface for the trailing matrix operation
    struct starpu_vector_interface **v_i =
        (struct starpu_vector_interface **)buffers + k;
    k += v_pi.handles;

    // current column
    double *p = P+i*ldP;

    //
    // update the current column
    //

    if (0 < i) {

        // A <- A - Y * V' (update column from the right)
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, i,
            -1.0, Y, ldY, V+i-1, ldV, 1.0, p, 1);

        //
        // update column from the left
        //

        // we use the last column of T as a work space
        double *w = T+(nb-1)*ldT;

        // w <- V1' * b1 (upper part)
        cblas_dcopy(i, p, 1, w, 1);
        cblas_dtrmv(
            CblasColMajor, CblasLower, CblasTrans, CblasUnit, i, V, ldV, w, 1);

        // w <- w + V2' * b2 (lower part)
        cblas_dgemv(CblasColMajor, CblasTrans, m-i, i,
            1.0, V+i, ldV, p+i, 1, 1.0, w, 1);

        // w <- T' * w
        cblas_dtrmv(
            CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, i,
            T, ldT, w, 1);

        // b2 <- b2 - V2 * w
        cblas_dgemv(CblasColMajor, CblasNoTrans, m-i, i,
            -1.0, V+i, ldV, w, 1, 1.0, p+i, 1);

        // b1 <- b1 - V1 * w
        cblas_dtrmv(
            CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, i,
            V, ldV, w, 1);
        cblas_daxpy(i, -1.0, w, 1, p, 1);
    }

    //
    // compute the current unit vector
    //

    int height = m-i;
    double tau, *v = V+i*ldV+i;
    memcpy(v, p+i, height*sizeof(double));
    dlarfg_(&height, p+i, v+1, (const int[]){1}, &tau);
    v[0] = 1.0;

    //
    // copy the current unit vector to the intemediate vector interface
    //

    starneig_join_range(&v_pi, v_i, v, 1);

    //
    // set elements below the subdiagonal to zero
    //

    for (int j = i+1; j < m; j++)
        p[j] = 0.0;

    //
    // store tau for future use
    //

    T[i*ldT+i] = tau;
}

void starneig_hessenberg_cpu_compute_column(
    void *buffers[], void *cl_args)
{
    struct packing_info A_pi;
    struct range_packing_info v_pi, y_pi;
    starpu_codelet_unpack_args(cl_args, &A_pi, &v_pi, &y_pi);

    //STARNEIG_EVENT_BEGIN(&A_pi, starneig_event_red);

    int k = 0;

    // involved trailing matrix tiles
    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += A_pi.handles;

    // intemediate vector interface for the trailing matrix operation
    struct starpu_vector_interface **v_i =
        (struct starpu_vector_interface **)buffers + k;
    k += v_pi.handles;

    // intemediate vector interface from the trailing matrix operation
    struct starpu_vector_interface **y_i =
        (struct starpu_vector_interface **)buffers + k;
    k += y_pi.handles;

    int t_rows = (A_pi.rend-1) / A_pi.bm + 1 - (A_pi.rbegin-1) / A_pi.bm;
    int t_cols = (A_pi.cend-1) / A_pi.bn + 1 - (A_pi.cbegin-1) / A_pi.bn;

    //
    // loop oper tile rows
    //

    for (int i = 0; i < t_rows; i++) {

        double *y = (double *) STARPU_VECTOR_GET_PTR(y_i[i]);

        int rbegin = MAX(     0,  A_pi.rbegin - i * A_pi.bm);
        int rend =   MIN(A_pi.bm, A_pi.rend   - i * A_pi.bm);

        for (int l = rbegin; l < rend; l++)
            y[l] = 0.0;

        //
        // loop over tile columns
        //

        for (int j = 0; j < t_cols; j++) {

            double *A = (double *) STARPU_MATRIX_GET_PTR(A_i[j*t_rows+i]);
            int ldA = STARPU_MATRIX_GET_LD(A_i[j*t_rows+i]);

            double *v = (double *) STARPU_VECTOR_GET_PTR(v_i[j]);

            int cbegin = MAX(      0, A_pi.cbegin - j * A_pi.bn);
            int cend =   MIN(A_pi.bn, A_pi.cend   - j * A_pi.bn);

            cblas_dgemv(
                CblasColMajor, CblasNoTrans, rend-rbegin, cend-cbegin,
                1.0, A+cbegin*ldA+rbegin, ldA, v+cbegin, 1, 1.0, y+rbegin, 1);
        }
    }

    //STARNEIG_EVENT_END();
}

void starneig_hessenberg_cpu_finish_column(
    void *buffers[], void *cl_args)
{
    struct range_packing_info y_pi;
    int i;
    starpu_codelet_unpack_args(cl_args, &i, &y_pi);

    int k = 0;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
    int m = STARPU_MATRIX_GET_NX(buffers[k]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    double *Y = (double *) STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldY = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    // intemediate vector interface from the trailing matrix operation
    struct starpu_vector_interface **y_i =
        (struct starpu_vector_interface **)buffers + k;
    k += y_pi.handles;

    double tau = T[i*ldT+i];
    double *v = V+i*ldV+i;

    //
    // finish Y update
    //

    starneig_join_range(&y_pi, y_i, Y+i*ldY, 0);

    // w <- V' * v (shared result)
    cblas_dgemv(CblasColMajor, CblasTrans, m-i, i,
        1.0, V+i, ldV, v, 1, 0.0, T+i*ldT, 1);

    // Y(:,i) <- Y(:,i) - Y * w
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, i,
        -1.0, Y, ldY, T+i*ldT, 1, 1.0, Y+i*ldY, 1);

    cblas_dscal(m, tau, Y+i*ldY, 1);

    //
    // update T
    //

    // w <- tau * w
    cblas_dscal(i, -tau, T+i*ldT, 1);

    // T(0:i,i) = T * w
    cblas_dtrmv(
        CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, i,
        T, ldT, T+i*ldT, 1);

    T[i*ldT+i] = tau;
}

void starneig_hessenberg_cpu_update_trail(
    void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb, offset;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb, &offset);

    STARNEIG_EVENT_BEGIN(&packing_info, starneig_event_red);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY = STARPU_MATRIX_GET_LD(buffers[2]);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int nA = STARPU_MATRIX_GET_NY(buffers[3]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[3]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[4]);
    int mW = STARPU_MATRIX_GET_NX(buffers[4]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[4]);

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + 5;

    int max_width = MIN(nA, mW);


    for (int i = 0; i < n; i += max_width) {

        //
        // join tiles and update from the right
        //

        starneig_join_sub_window(
            0, m, i, MIN(n, i+max_width), &packing_info, ldA, A_i, A, 0);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            m, MIN(max_width, n-i), nb, -1.0,
            Y, ldY, V+offset+i+nb-1, ldV, 1.0, A, ldA);

        //
        // update from the left
        //

        int width = MIN(max_width, n-i);
        if (0 < width) {
            for (int k = 0; k < nb; k++)
                cblas_dcopy(width, A+k, ldA, W+k*ldW, 1);

            cblas_dtrmm(
                CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                CblasUnit, width, nb, 1.0, V, ldV, W, ldW);

            if (nb < m)
                cblas_dgemm(
                    CblasColMajor, CblasTrans, CblasNoTrans, width, nb,
                    m-nb, 1.0, A+nb, ldA, V+nb, ldV, 1.0, W, ldW);

            cblas_dtrmm(
                CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, width, nb, 1.0, T, ldT, W, ldW);

            if (nb < m)
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasTrans, m-nb, width,
                    nb, -1.0, V+nb, ldV, W, ldW, 1.0, A+nb, ldA);

            cblas_dtrmm(
                CblasColMajor, CblasRight, CblasLower, CblasTrans,
                CblasUnit, width, nb, 1.0, V, ldV, W, ldW);

            for (int k = 0; k < nb; k++)
                cblas_daxpy(width, -1.0, W+k*ldW, 1, A+k, ldA);
        }

        //
        // copy tiles back
        //

        starneig_join_sub_window(
            0, m, i, MIN(n, i+max_width), &packing_info, ldA, A_i, A, 1);
    }

    STARNEIG_EVENT_END();
}


void starneig_hessenberg_cpu_update_right(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    STARNEIG_EVENT_BEGIN(&packing_info, starneig_event_blue);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[3]);

    struct starpu_matrix_interface **X_i =
        (struct starpu_matrix_interface **)buffers + 4;

    starneig_join_window(&packing_info, ldX, X_i, X, 0);

    for (int j = 0; j < nb; j++)
        cblas_dcopy(m, X+j*ldX, 1, W+j*ldW, 1);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
        m, nb, 1.0, V, ldV, W, ldW);

    if (nb < n)
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, m, nb, n-nb,
            1.0, X+nb*ldX, ldX, V+nb, ldV, 1.0, W, ldW);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, nb, 1.0, T, ldT, W, ldW);

    if (nb < n)
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasTrans, m, n-nb, nb,
            -1.0, W, ldW, V+nb, ldV, 1.0, X+nb*ldX, ldX);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
        m, nb, 1.0, V, ldV, W, ldW);

    for (int j = 0; j < nb; j++)
        cblas_daxpy(m, -1.0, W+j*ldW, 1, X+j*ldX, 1);

    starneig_join_window(&packing_info, ldX, X_i, X, 1);

    STARNEIG_EVENT_END();
}

void starneig_hessenberg_cpu_update_left(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    STARNEIG_EVENT_BEGIN(&packing_info, starneig_event_green);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[3]);

    struct starpu_matrix_interface **X_i =
        (struct starpu_matrix_interface **)buffers + 4;

    starneig_join_window(&packing_info, ldX, X_i, X, 0);

    for (int j = 0; j < nb; j++)
        cblas_dcopy(n, X+j, ldX, W+j*ldW, 1);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
        n, nb, 1.0, V, ldV, W, ldW);

    if (nb < m)
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans, n, nb, m-nb,
            1.0, X+nb, ldX, V+nb, ldV, 1.0, W, ldW);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
        n, nb, 1.0, T, ldT, W, ldW);

    if (nb < m)
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasTrans, m-nb, n, nb,
            -1.0, V+nb, ldV, W, ldW, 1.0, X+nb, ldX);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
        n, nb, 1.0, V, ldV, W, ldW);

    for (int j = 0; j < nb; j++)
        cblas_daxpy(n, -1.0, W+j*ldW, 1, X+j, ldX);

    starneig_join_window(&packing_info, ldX, X_i, X, 1);

    STARNEIG_EVENT_END();
}
