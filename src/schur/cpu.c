///
/// @file
///
/// @brief This file contains the CPU implementations of codelets that are used
/// in the StarPU-bases QR algorithm.
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
#include "common.h"
#include "cpu_utils.h"
#include "../common/common.h"
#include "../common/sanity.h"
#include "../common/tiles.h"
#include "../common/math.h"
#include "../common/trace.h"
#include <math.h>

#define _A(i,j) A[(j)*ldA+(i)]
#define _B(i,j) B[(j)*ldB+(i)]
#define _Q(i,j) Q[(j)*ldQ+(i)]
#define _Z(i,j) Z[(j)*ldZ+(i)]

void starneig_cpu_push_inf_top(void *buffers[], void *cl_arg)
{
    double thres_inf;
    struct packing_info packing_info_A, packing_info_B;
    int top, bottom;
    starpu_codelet_unpack_args(cl_arg,
        &thres_inf, &packing_info_A, &packing_info_B, &top, &bottom);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int window_size = packing_info_A.rend - packing_info_A.rbegin;

    int k = 0;

    // local left-hand size transformation matrix

    struct starpu_matrix_interface *lQ_i = buffers[k];
    double *Q = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    size_t ldQ = STARPU_MATRIX_GET_LD(lQ_i);
    k++;

    // local right-hand side transformation matrix

    struct starpu_matrix_interface *lZ_i = buffers[k];
    double *Z = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
    size_t ldZ = STARPU_MATRIX_GET_LD(lZ_i);
    k++;

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_B.handles;

    // join tiles and initialize

    size_t ldA;
    double *A = starneig_alloc_matrix(
        window_size, window_size, sizeof(double), &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB;
    double *B = starneig_alloc_matrix(
        window_size, window_size, sizeof(double), &ldB);
    starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);

    starneig_init_local_q(window_size, ldQ, Q);
    starneig_init_local_q(window_size, ldZ, Z);

    // detect infinite eigenvalues

    for (int i = 0; i < window_size; i++) {
        if (fabs(B[i*ldB+i]) < thres_inf)
            B[i*ldB+i] = 0.0;
    }

    // move detected infinite eigenvalues to the upper left corner and deflate
    // them if possible

    if (bottom)
        starneig_push_inf_top(
            0, window_size, window_size, ldQ, ldZ, ldA, ldB, Q, Z, A, B, top);
    else
        starneig_push_inf_top(
            0, window_size-1, window_size, ldQ, ldZ, ldA, ldB, Q, Z, A, B, top);

    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 1);
    starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 1);

    starneig_free_matrix(A);
    starneig_free_matrix(B);

    STARNEIG_EVENT_END();
}

void starneig_cpu_push_bulges(void *buffers[], void *cl_arg)
{
    double thres_a, thres_b, thres_inf;
    struct range_packing_info packing_info_shifts_real;
    struct range_packing_info packing_info_shifts_imag;
    struct range_packing_info packing_info_aftermath;
    struct packing_info packing_info_A, packing_info_B;
    bulge_chasing_mode_t mode;
    starpu_codelet_unpack_args(cl_arg, &thres_a, &thres_b, &thres_inf,
        &packing_info_shifts_real, &packing_info_shifts_imag,
        &packing_info_aftermath, &packing_info_A, &packing_info_B, &mode);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int generalized = 0 < packing_info_B.handles;
    int check_aftermath = 0 < packing_info_aftermath.handles;
    int window_size = packing_info_A.rend - packing_info_A.rbegin;
    int shifts =
        packing_info_shifts_real.end - packing_info_shifts_real.begin;

    int k = 0;

    // shifts (real parts)

    struct starpu_vector_interface **real_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_shifts_real.handles;

    // shifts (imaginary parts)

    struct starpu_vector_interface **imag_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_shifts_imag.handles;

    // local left-hand size transformation matrix

    struct starpu_matrix_interface *lQ_i = buffers[k];
    double *Q = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    size_t ldQ = STARPU_MATRIX_GET_LD(lQ_i);
    k++;

    // local right-hand side transformation matrix

    double *Z = Q;
    size_t ldZ = ldQ;
    if (generalized) {
        struct starpu_matrix_interface *lZ_i = buffers[k];
        Z = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        ldZ = STARPU_MATRIX_GET_LD(lZ_i);
        k++;
    }

    // deflation check vector

    struct starpu_vector_interface **aftermath_i = NULL;
    if (check_aftermath) {
        aftermath_i = (struct starpu_vector_interface **)buffers + k;
        k += packing_info_aftermath.handles;
    }

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;
    }

    // join tiles and initialize

    size_t ldA;
    double *A = starneig_alloc_matrix(
        window_size, window_size, sizeof(double), &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB = 0;
    double *B = NULL;
    if (generalized) {
        B = starneig_alloc_matrix(
            window_size, window_size, sizeof(double), &ldB);
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
    }

    starneig_init_local_q(window_size, ldQ, Q);
    if (Z != Q)
        starneig_init_local_q(window_size, ldZ, Z);

    double *real = malloc(shifts*sizeof(double));
    double *imag = malloc(shifts*sizeof(double));
    starneig_join_range(&packing_info_shifts_real, real_i, real, 0);
    starneig_join_range(&packing_info_shifts_imag, imag_i, imag, 0);

    // push bulges

    starneig_push_bulges(mode, shifts, window_size, ldQ, ldQ, ldA, ldB,
        thres_a, thres_b, thres_inf, real, imag, Q, Z, A, B);

    // check deflation

    if (check_aftermath) {
        int *aftermath = malloc(window_size*sizeof(bulge_chasing_aftermath_t));
        starneig_join_range(&packing_info_aftermath, aftermath_i, aftermath, 0);
        for (int i = 1; i < window_size; i++) {
            aftermath[i] = BULGE_CHASING_AFTERMATH_NONE;
            if (_A(i,i-1) == 0.0)
                aftermath[i] |= BULGE_CHASING_AFTERMATH_DEFLATED;
            if (B != NULL && _B(i,i) == 0.0)
                aftermath[i] |= BULGE_CHASING_AFTERMATH_INFINITY;
        }
        starneig_join_range(&packing_info_aftermath, aftermath_i, aftermath, 1);
        free(aftermath);
    }

    // store result

    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 1);
    if (generalized)
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 1);

    free(real);
    free(imag);
    starneig_free_matrix(A);
    starneig_free_matrix(B);

    STARNEIG_EVENT_END();
}

void starneig_cpu_aggressively_deflate(void *buffers[], void *cl_arg)
{
    double thres_a, thres_b, thres_inf;
    struct range_packing_info packing_info_shifts_real;
    struct range_packing_info packing_info_shifts_imag;
    struct packing_info packing_info_A, packing_info_B;
    starpu_codelet_unpack_args(cl_arg, &thres_a, &thres_b, &thres_inf,
        &packing_info_shifts_real, &packing_info_shifts_imag,
        &packing_info_A, &packing_info_B);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int generalized = 0 < packing_info_B.handles;
    int window_size = packing_info_A.rend - packing_info_A.rbegin;

    int k = 0;

    // returns status

    struct aed_status *status = (struct aed_status*)
        STARPU_VARIABLE_GET_PTR(buffers[k]);
    k++;

    // local left-hand side transformation matrix

    struct starpu_matrix_interface *lQ_i = buffers[k];
    double *Q = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    size_t ldQ = STARPU_MATRIX_GET_LD(lQ_i);
    starneig_init_local_q(window_size, ldQ, Q);
    k++;

    // local right-hand side transformation matrix

    double *Z = Q;
    size_t ldZ = ldQ;
    if (generalized) {
        struct starpu_matrix_interface *lZ_i = buffers[k];
        Z = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        ldZ = STARPU_MATRIX_GET_LD(lZ_i);
        k++;
    }

    // shifts (real parts)

    struct starpu_vector_interface **real_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_shifts_real.handles;

    // shifts (imaginary parts)

    struct starpu_vector_interface **imag_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_shifts_imag.handles;

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;
    }

    // join tiles and initialize

    size_t ldA;
    double *A = starneig_alloc_matrix(
        window_size, window_size, sizeof(double), &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB = 0;
    double *B = NULL;
    if (generalized) {
        B = starneig_alloc_matrix(
            window_size, window_size, sizeof(double), &ldB);
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
    }

    starneig_init_local_q(window_size, ldQ, Q);
    if (Z != Q)
        starneig_init_local_q(window_size, ldZ, Z);

    double *real = malloc(window_size*sizeof(double));
    double *imag = malloc(window_size*sizeof(double));

    // aggressively deflate, early

    int unconverged = 0, converged = 0;
    starneig_aggressively_deflate(
        window_size, ldQ, ldQ, ldA, ldB, thres_a, thres_b, thres_inf,
        real, imag, Q, Z, A, B, &unconverged, &converged);

    if (0 < converged || 2 <= unconverged)
        status->status = AED_STATUS_SUCCESS;
    else
        status->status = AED_STATUS_FAILURE;
    status->converged = converged;
    status->computed_shifts = unconverged;

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
    for (int i = 0; i < unconverged; i++)
        STARNEIG_SANITY_CHECK(
            real[i] != 0.0 || imag[i] != 0.0, "Some shifts are zero.");
    for (int i = 0; i < unconverged/2; i++)
        STARNEIG_SANITY_CHECK(imag[2*i] == -imag[2*i+1],
            "The shifts are not ordered correctly.");
#endif

    starneig_join_range(&packing_info_shifts_real, real_i, real, 1);
    starneig_join_range(&packing_info_shifts_imag, imag_i, imag, 1);

    // store the result back only when AED managed to converge something
    if (0 < converged) {
        starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 1);
        if (generalized)
            starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 1);
    }

    starneig_free_matrix(A);
    starneig_free_matrix(B);
    free(real);
    free(imag);

    STARNEIG_EVENT_END();
}

void starneig_cpu_small_schur(void *buffers[], void *cl_arg)
{
    double thres_a, thres_b, thres_inf;
    struct packing_info packing_info_A, packing_info_B;
    starpu_codelet_unpack_args(cl_arg,
        &thres_a, &thres_b, &thres_inf, &packing_info_A, &packing_info_B);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int generalized = 0 < packing_info_B.handles;
    int size = packing_info_A.rend - packing_info_A.rbegin;

    int k = 0;

    // returns status

    struct small_schur_status *status =
        (struct small_schur_status *) STARPU_VARIABLE_GET_PTR(buffers[k]);
    k++;

    // local left-hand side transformation matrix

    struct starpu_matrix_interface *lQ_i = buffers[k];
    double *Q = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    size_t ldQ = STARPU_MATRIX_GET_LD(lQ_i);
    k++;

    // local right-hand side transformation matrix

    double *Z = Q;
    size_t ldZ = ldQ;
    if (generalized) {
        struct starpu_matrix_interface *lZ_i = buffers[k];
        Z = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        ldZ = STARPU_MATRIX_GET_LD(lZ_i);
        k++;
    }

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;
    }

    starneig_init_local_q(size, ldQ, Q);
    if (generalized)
        starneig_init_local_q(size, ldZ, Z);

    // join tiles and initialize

    size_t ldA;
    double *A = starneig_alloc_matrix(size, size, sizeof(double), &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB = 0;
    double *B = NULL;
    if (generalized) {
        B = starneig_alloc_matrix(size, size, sizeof(double), &ldB);
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
    }

    double *real = malloc(size*sizeof(double));
    double *imag = malloc(size*sizeof(double));
    double *beta = malloc(size*sizeof(double));

    // reduce

    int info = starneig_schur_reduction(
        size, ldQ, ldZ, ldA, ldB, thres_a, thres_b, thres_inf,
        real, imag, beta, Q, Z, A, B);

    // store result

    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 1);
    if (generalized)
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 1);

    status->converged = size - info;

    starneig_free_matrix(A);
    starneig_free_matrix(B);
    free(real);
    free(imag);
    free(beta);

    STARNEIG_EVENT_END();
}

void starneig_cpu_small_hessenberg(void *buffers[], void *cl_arg)
{
    struct packing_info packing_info_A, packing_info_B;
    starpu_codelet_unpack_args(cl_arg, &packing_info_A, &packing_info_B);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int generalized = 0 < packing_info_B.handles;
    int size = packing_info_A.rend - packing_info_A.rbegin;

    int k = 0;

    // local left-hand side transformation matrix

    struct starpu_matrix_interface *lQ_i = buffers[k];
    double *Q = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    size_t ldQ = STARPU_MATRIX_GET_LD(lQ_i);
    k++;

    // local right-hand side transformation matrix

    double *Z = Q;
    size_t ldZ = ldQ;
    if (generalized) {
        struct starpu_matrix_interface *lZ_i = buffers[k];
        Z = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        ldZ = STARPU_MATRIX_GET_LD(lZ_i);
        k++;
    }

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;
    }

    starneig_init_local_q(size, ldQ, Q);
    if (generalized)
        starneig_init_local_q(size, ldZ, Z);

    // join tiles and initialize

    size_t ldA;
    double *A = starneig_alloc_matrix(size, size, sizeof(double), &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB = 0;
    double *B = NULL;
    if (generalized) {
        B = starneig_alloc_matrix(size, size, sizeof(double), &ldB);
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
    }

    // reduce

    starneig_hessenberg_reduction(size, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    // store result

    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 1);
    if (generalized)
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 1);

    starneig_free_matrix(A);
    starneig_free_matrix(B);

    STARNEIG_EVENT_END();
}

void starneig_cpu_form_spike(void *buffers[], void *cl_arg)
{
    struct packing_info packing_info;
    struct range_packing_info packing_info_spike;
    starpu_codelet_unpack_args(cl_arg, &packing_info, &packing_info_spike);

    // AED window size
    int size = packing_info_spike.end - packing_info_spike.begin;

    int k = 0;

    // first row from the AED transformation matrix
    struct starpu_matrix_interface **Q_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info.handles;

    // spike base
    struct starpu_vector_interface **spike_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info.handles;

    double *spike = malloc(size*sizeof(double));
    starneig_join_window(&packing_info, 1, Q_i, spike, 0);
    starneig_join_range(&packing_info_spike, spike_i, spike, 1);
    free(spike);
}

void starneig_cpu_embed_spike(void *buffers[], void *cl_arg)
{
    struct range_packing_info packing_info_spike;
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_arg, &packing_info_spike, &packing_info);

    int window_size = packing_info.rend - packing_info.rbegin;
    int spike_size = packing_info_spike.end - packing_info_spike.begin;

    int k = 0;

    // spike base
    struct starpu_vector_interface **spike_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_spike.handles;

    // column to the left of the AED window
    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info.handles;

    // extract the first sub-diagonal entry
    double sub;
    starneig_join_sub_window(0, 1, 0, 1, &packing_info, 1, A_i, &sub, 0);

    // form and embed the spike
    double *column = malloc(window_size*sizeof(double));
    starneig_join_range(&packing_info_spike, spike_i, column, 0);
    for (int i = 0; i < spike_size; i++)
        column[i] *= sub;
    for (int i = spike_size; i < window_size; i++)
        column[i] = 0.0;
    starneig_join_window(&packing_info, window_size, A_i, column, 1);
    free(column);
}

void starneig_cpu_deflate(void *buffers[], void *cl_arg)
{
    double thres_a;
    struct range_packing_info packing_info_spike;
    struct packing_info packing_info_A, packing_info_B;
    int offset, deflate, corner;
    starpu_codelet_unpack_args(cl_arg,
        &thres_a, &packing_info_spike, &packing_info_A, &packing_info_B,
        &offset, &deflate, &corner);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int generalized = 0 < packing_info_B.handles;
    int size = packing_info_A.rend - packing_info_A.rbegin;

    int k = 0;

    // spike inducer

    double sub = *((double *) STARPU_VARIABLE_GET_PTR(buffers[k]));
    k++;

    // returns status

    struct deflate_status *status = (struct deflate_status *)
        STARPU_VARIABLE_GET_PTR(buffers[k]);
    k++;

    // local left-hand side transformation matrix

    struct starpu_matrix_interface *lQ_i = buffers[k];
    double *lQ = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    size_t ldlQ = STARPU_MATRIX_GET_LD(lQ_i);
    k++;

    // local right-hand side transformation matrix

    double *lZ = lQ;
    size_t ldlZ = ldlQ;
    if (generalized) {
        struct starpu_matrix_interface *lZ_i = buffers[k];
        lZ = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        ldlZ = STARPU_MATRIX_GET_LD(lZ_i);
        k++;
    }

    // spike base

    struct starpu_vector_interface **spike_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_spike.handles;

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;
    }

    //
    // The corresponding section of the matrix pencil (A,B), the local
    // transformation matrices and the matching section of the spike (s) are
    // embedded to (size+1) X (size+1) matrices:
    //        Q                 A                B                Z
    // +--------------   +--------------  +--------------  +--------------
    // | 1           x   | A A A A A A s  | B B B B B B x  | 1           x
    // |   1         x   | A A A A A A s  |   B B B B B x  |   1         x
    // |     1       x   |     A A A A s  |     B B B B x  |     1       x
    // |       1     x   |     A A A A s  |       B B B x  |       1     x
    // |         1   x   |         A A s  |         B B x  |         1   x
    // |           1 x   |           A s  |           B x  |           1 x
    // | x x x x x x x   | x x x x x 0 x  | x x x x x x x  | x x x x x x x
    //
    // Since the spike is also embedded, it gets implicitly updated.
    //

    size_t ldA, ldQ;
    double *A = starneig_alloc_matrix(size+1, size+1, sizeof(double), &ldA);
    double *Q = starneig_alloc_matrix(size+1, size+1, sizeof(double), &ldQ);

    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);
    starneig_join_range(&packing_info_spike, spike_i, &_A(0,size), 0);
    _A(size,size-1) = 0.0;
    starneig_init_local_q(size, ldQ, Q);

    size_t ldB = 0, ldZ = ldQ;
    double *B = NULL;
    double *Z = Q;
    int lwork;
    if (generalized) {
        B = starneig_alloc_matrix(size+1, size+1, sizeof(double), &ldB);
        Z = starneig_alloc_matrix(size+1, size+1, sizeof(double), &ldZ);

        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
        starneig_init_local_q(size, ldZ, Z);

        lwork = 4*(size+1)+16;
    }
    else {
        lwork = size+1;
    }

    double *work = malloc(lwork*sizeof(double));

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
    //
    //        Q                 A                B                Z
    // +--------------   +--------------  +--------------  +--------------
    // | 1           0   | A A A A A A s  | B B B B B B 0  | 1           0
    // |   1         0   | A A A A A A s  |   B B B B B 0  |   1         0
    // |     1       0   |     A A A A s  |     B B B B 0  |     1       0
    // |       1     0   |     A A A A s  |       B B B 0  |       1     0
    // |         1   0   |         A A s  |         B B 0  |         1   0
    // |           1 0   |           A s  |           B 0  |           1 0
    // | 0 0 0 0 0 0 1   | 0 0 0 0 0 0 0  | 0 0 0 0 0 0 0  | 0 0 0 0 0 0 1
    //
    for (int i = 0; i < size+1; i++) {
        _A(size,i) = 0.0;
        _Q(size,i) = 0.0;
        _Q(i,size) = 0.0;
    }
    _Q(size,size) = 1.0;
    if (B != NULL) {
        for (int i = 0; i < size+1; i++) {
            _B(size,i) = 0.0;
            _B(i,size) = 0.0;
            _Z(size,i) = 0.0;
            _Z(i,size) = 0.0;
        }
        _Z(size,size) = 1.0;
    }

    STARNEIG_SANITY_CHECK_SCHUR(0, size+1, size+1, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, size+1, ldQ, ldZ, ldA, ldB, Q, Z, A, B);
#endif

    //
    // Some padding is added to the deflation window since the predetermined
    // window boundary may split a 2-by-2 block. The actual deflation
    // window begins from 'begin':
    //
    //      _______           ? ?......          * _______
    //     |*       <- begin  ?:* _____           |* *     <- begin
    //     |  *                :0|*     <- begin  |* *
    //     |    * *            : |  * *           |    * *
    //     |    * *            : |  * *           |    * *
    //
    //  :. visible area, |_ actual window, ** diagonal block, ?? unknown entry
    //
    // If the window is located to the very top of the AED window, then no
    // padding is required.
    //
    int begin;
    if (offset == 1 || _A(1,0) != 0.0)
        begin = 0;
    else
        begin = 1;

    //
    // Similarly, the actual deflation windows ends to 'end'-1:
    //
    //      * *    |          * *  | :            * *    |
    //      * *    |          * *  | :            * *    |
    //          *  |          ____*| :                * *|
    //      ______*|          ....0.*:? <- bottom ____*_*|
    //               <- bottom      ? ?                   * <- bottom
    //
    //  :. visible area, _| actual window, ** diagonal block, ?? unknown entry
    //
    // If the current window involves any deflation or is located to the bottom
    // right corner of the AED window, then no padding is required.
    //
    int end;
    if (deflate || corner || _A(size-1,size-2) != 0.0)
        end = size;
    else
        end = size-1;

    //
    // The existing undeflated blocks that fall within the current window are
    // located to [_begin,_end[. A 2-by-2 block is included if and only if the
    // upper half of the block is included.
    //

    // undeflated block are moved to 'top'
    int top = begin;

    //
    // move undeflated blocks to the upper left corner of the window
    //

    int _begin = begin, _end = begin;
    if (status->inherited) {

        _begin = MAX(begin, status->begin - offset);
        if (0 <= _begin-1 && _A(_begin,_begin-1) != 0.0)
            _begin++;

        _end = MIN(end, status->end - offset);
        if (_end+1 < end && _A(_end+1,_end) != 0.0)
            _end++;

        int i = _begin;
        while (i < _end) {
            if (i+1 < _end && _A(i+1,i) != 0.0) {
                top = starneig_move_block(i, top, size+1, ldQ, ldZ, ldA, ldB,
                    lwork, Q, Z, A, B, work);
                top += 2;
                i = MAX(top, i+2);
            }
            else {
                top = starneig_move_block(i, top, size+1, ldQ, ldZ, ldA, ldB,
                    lwork, Q, Z, A, B, work);
                top++;
                i = MAX(top, i+1);
            }
        }
        _begin = begin;
        _end = top;
    }

    //
    // deflation detection loop
    //

    if (deflate) {

        //
        // norm stable deflation condition
        //

        if (0.0 < thres_a) {
            int i = end-1;
            while (top <= i) {

                // if we are dealing with a 2-by-2 block, ...
                if (top <= i-1 && _A(i,i-1) != 0.0) {
                    // and the 2-by-2 block is deflatable, ...
                    if (fabs(sub*_A(i-1,size)) < thres_a &&
                    fabs(sub*_A(i,size)) < thres_a) {
                        // decrease the AED window
                        i -= 2;
                    }
                    // otherwise, ...
                    else {
                        // move the 2-by-2 block out of the way
                        top = starneig_move_block(
                            i, top, size+1, ldQ, ldZ, ldA, ldB,
                            lwork, Q, Z, A, B, work);
                        top += 2;
                    }
                }
                // otherwise, ...
                else {
                    // if the 1-by-1 block is deflatable, ...
                    if (fabs(sub*_A(i,size)) < thres_a) {
                        // decrease the AED window
                        i--;
                    }
                    // otherwise, ...
                    else {
                        // move the 1-by-1 block out of the way
                        top = starneig_move_block(
                            i, top, size+1, ldQ, ldZ, ldA, ldB,
                            lwork, Q, Z, A, B, work);
                        top++;
                    }
                }
            }
        }

        //
        // LAPACK-style deflation condition
        //

        else {
            const double safmin = dlamch("Safe minimum");
            const double ulp = dlamch("Precision");
            double smlnum = safmin*(packing_info_A.n/ulp);

            int i = end-1;
            while (top <= i) {

                // if we are dealing with a 2-by-2 block, ...
                if (top <= i-1 && _A(i,i-1) != 0.0) {
                    double foo = fabs(_A(i,i)) +
                        sqrt(fabs(_A(i,i-1))) * sqrt(fabs(_A(i-1,i)));
                    if (foo == 0.0)
                        foo = fabs(sub);

                    // and the 2-by-2 block is deflatable, ...
                    if (MAX(fabs(sub*_A(i-1,size)), fabs(sub*_A(i,size))) <
                    MAX(smlnum, ulp*foo)) {
                        // decrease the AED window
                        i -= 2;
                    }
                    // otherwise, ...
                    else {
                        // move the 2-by-2 block out of the way
                        top = starneig_move_block(
                            i, top, size+1, ldQ, ldZ, ldA, ldB,
                            lwork, Q, Z, A, B, work);
                        top += 2;
                    }
                }
                // otherwise, ...
                else {
                    double foo = fabs(_A(i,i));
                    if (foo == 0.0)
                        foo = fabs(sub);

                    // if the 1-by-1 block is deflatable, ...
                    if (fabs(sub*_A(i,size)) < MAX(smlnum, ulp*foo)) {
                        // decrease the AED window
                        i--;
                    }
                    // otherwise, ...
                    else {
                        // move the 1-by-1 block out of the way
                        top = starneig_move_block(
                            i, top, size+1, ldQ, ldZ, ldA, ldB,
                            lwork, Q, Z, A, B, work);
                        top++;
                    }
                }
            }
        }

        _begin = begin;
        _end = top;
    }

    status->begin = _begin + offset;
    status->end = _end + offset;

    STARNEIG_SANITY_CHECK_SCHUR(0, size+1, size+1, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, size+1, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    //
    // copy the updated window and the local transformation matrićes
    //

    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 1);
    starneig_join_range(&packing_info_spike, spike_i, &_A(0,size), 1);
    starneig_copy_matrix(size, size, ldQ, ldlQ, sizeof(double), Q, lQ);

    if (generalized) {
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 1);
        starneig_copy_matrix(size, size, ldZ, ldlZ, sizeof(double), Z, lZ);
    }

    starneig_free_matrix(A);
    starneig_free_matrix(B);
    starneig_free_matrix(Q);
    if (Z != Q)
        starneig_free_matrix(Z);
    free(work);

    STARNEIG_EVENT_END();
}

void starneig_cpu_extract_shifts(void *buffers[], void *cl_args)
{
    struct packing_info packing_info_A, packing_info_B;
    struct range_packing_info packing_info_real, packing_info_imag;
    starpu_codelet_unpack_args(cl_args,
        &packing_info_A, &packing_info_B, &packing_info_real,
        &packing_info_imag);

    int size = packing_info_A.rend - packing_info_A.rbegin;
    int generalized = 0 < packing_info_B.handles;

    int k = 0;

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **) buffers + k;
    k += packing_info_A.handles;

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **) buffers + k;
        k += packing_info_B.handles;
    }

    struct starpu_vector_interface **real_i =
        (struct starpu_vector_interface **) buffers + k;
    k += packing_info_real.handles;

    struct starpu_vector_interface **imag_i =
        (struct starpu_vector_interface **) buffers + k;
    k += packing_info_imag.handles;

    size_t ldA;
    double *A = starneig_alloc_matrix(size, size, sizeof(double), &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB = 0;
    double *B = NULL;
    if (generalized) {
        B = starneig_alloc_matrix(size, size, sizeof(double), &ldB);
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
    }

    double *real = malloc(size*sizeof(double));
    double *imag = malloc(size*sizeof(double));

    starneig_extract_shifts(size, ldA, ldB, A, B, real, imag);

    starneig_join_range(&packing_info_real, real_i, real, 1);
    starneig_join_range(&packing_info_imag, imag_i, imag, 1);

    starneig_free_matrix(A);
    starneig_free_matrix(B);
    free(real);
    free(imag);
}

void starneig_cpu_compute_norm_a(void *buffers[], void *cl_args)
{
    int rbegin, rend, cbegin, cend;
    starpu_codelet_unpack_args(cl_args, &rbegin, &rend, &cbegin, &cend);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[0]);

    double dot = 0.0;
    for (int j = cbegin; j < cend; j++)
        for (int i = rbegin; i < rend; i++)
            dot += squ(A[j*ldA+i]);

    *((double *)STARPU_VARIABLE_GET_PTR(buffers[1])) = dot;
}

void starneig_cpu_compute_norm_b(void *buffers[], void *cl_args)
{
    int size;
    starpu_codelet_unpack_args(cl_args, &size);

    double dot = 0.0;
    for (int i = 0; i < size; i++)
        dot += *((double *)STARPU_VARIABLE_GET_PTR(buffers[i]));

    *((double *)STARPU_VARIABLE_GET_PTR(buffers[size])) = sqrt(dot);
}
