///
/// @file
///
/// @brief This file contains the CPU implementations of codelets that are
/// shared among all components of the library.
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
#include "tiles.h"
#include "sanity.h"
#include "trace.h"
#include <math.h>
#include <starpu.h>

extern void dgemm_(char const *, char const *, int const *, int const *,
    int const *, double const *, double const *, int const *, double const *,
    int const *, double const *, double*, int const *);

void starneig_cpu_left_gemm_update(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    STARNEIG_EVENT_BEGIN(&packing_info, starneig_event_green);

    // local Q matrix
    double *lq_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int lq_ld = STARPU_MATRIX_GET_LD(buffers[0]);

    STARNEIG_SANITY_CHECK_ORTHOGONALITY(
        STARPU_MATRIX_GET_NX(buffers[0]), lq_ld, lq_ptr, "lQ");

    // scratch buffers
    int st1_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    int st2_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    double *st1_ptr = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    double *st2_ptr = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);

    // corresponding tiles from the A matrix
    struct starpu_matrix_interface **a_i =
        (struct starpu_matrix_interface **)buffers + 3;

    // st1 <- Y
    starneig_join_window(&packing_info, st1_ld, a_i, st1_ptr, 0);

    STARNEIG_SANITY_CHECK_INF(
        0, packing_info.rend - packing_info.rbegin,
        0, packing_info.cend - packing_info.cbegin,
        st1_ld, st1_ptr, "A (in)");


    // st2 <- Q^T * st1

    int n = packing_info.rend - packing_info.rbegin;
    int m = packing_info.cend - packing_info.cbegin;
    int k = packing_info.rend - packing_info.rbegin;

    double one = 1.0;
    double zero = 0.0;

    dgemm_("T", "N", &n, &m, &k,
        &one, lq_ptr, &lq_ld, st1_ptr, &st1_ld, &zero, st2_ptr, &st2_ld);

    STARNEIG_SANITY_CHECK_INF(
        0, packing_info.rend - packing_info.rbegin,
        0, packing_info.cend - packing_info.cbegin,
        st2_ld, st2_ptr, "A (out)");

    // Y <- st2
    starneig_join_window(&packing_info, st2_ld, a_i, st2_ptr, 1);

    STARNEIG_EVENT_END();
}

void starneig_cpu_right_gemm_update(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    STARNEIG_EVENT_BEGIN(&packing_info, starneig_event_blue);

    // local Q matrix
    double *lq_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int lq_ld = STARPU_MATRIX_GET_LD(buffers[0]);

    STARNEIG_SANITY_CHECK_ORTHOGONALITY(
        STARPU_MATRIX_GET_NX(buffers[0]), lq_ld, lq_ptr, "lQ");

    // scratch buffers
    int st1_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    int st2_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    double *st1_ptr = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    double *st2_ptr = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);

    // corresponding tiles from the A matrix
    struct starpu_matrix_interface **a_i =
        (struct starpu_matrix_interface **)buffers + 3;

    // st1 <- Y
    starneig_join_window(&packing_info, st1_ld, a_i, st1_ptr, 0);

    STARNEIG_SANITY_CHECK_INF(
        0, packing_info.rend - packing_info.rbegin,
        0, packing_info.cend - packing_info.cbegin,
        st1_ld, st1_ptr, "A (in)");

    // st2 <- st1 * Q
    int n = packing_info.rend - packing_info.rbegin;
    int m = packing_info.cend - packing_info.cbegin;
    int k = packing_info.cend - packing_info.cbegin;

    double one = 1.0;
    double zero = 0.0;

    dgemm_("N", "N", &n, &m, &k,
        &one, st1_ptr, &st1_ld, lq_ptr, &lq_ld, &zero, st2_ptr, &st2_ld);

    STARNEIG_SANITY_CHECK_INF(
        0, packing_info.rend - packing_info.rbegin,
        0, packing_info.cend - packing_info.cbegin,
        st2_ld, st2_ptr, "A (out)");

    // Y <- st2
    starneig_join_window(&packing_info, st2_ld, a_i, st2_ptr, 1);

    STARNEIG_EVENT_END();
}

void starneig_cpu_copy_matrix_to_handle(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    int k = 0;

    struct starpu_matrix_interface **source_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info.handles;

    double *T = (double *)STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    starneig_join_window(&packing_info, ldT, source_i, T, 0);

    STARNEIG_SANITY_CHECK_INF(
        0, packing_info.rend - packing_info.rbegin,
        0, packing_info.cend - packing_info.cbegin,
        ldT, T, "T");
}

void starneig_cpu_copy_handle_to_matrix(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    int k = 0;

    double *T = (double *)STARPU_MATRIX_GET_PTR(buffers[k]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    struct starpu_matrix_interface **dest_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info.handles;

    STARNEIG_SANITY_CHECK_INF(
        0, packing_info.rend - packing_info.rbegin,
        0, packing_info.cend - packing_info.cbegin,
        ldT, T, "T");

    starneig_join_window(&packing_info, ldT, dest_i, T, 1);
}

void starneig_cpu_copy_matrix(void *buffers[], void *cl_args)
{
    struct packing_info packing_info_source, packing_info_dest;
    starpu_codelet_unpack_args(
        cl_args, &packing_info_source, &packing_info_dest);

    int m = packing_info_source.rend - packing_info_source.rbegin;
    int n = packing_info_source.cend - packing_info_source.cbegin;

    int k = 0;

    struct starpu_matrix_interface **source_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_source.handles;

    double *T = (double *)STARPU_MATRIX_GET_PTR(buffers[k]);
    int _m = STARPU_MATRIX_GET_NX(buffers[k]);
    int _n = STARPU_MATRIX_GET_NY(buffers[k]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[k]);
    k++;

    struct starpu_matrix_interface **dest_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_dest.handles;

    for (int i = 0; i < m; i += _m) {
        for (int j = 0; j < n; j += _n) {
            starneig_join_sub_window(
                i, MIN(m, i+_m), j, MIN(n, j+_n),
                &packing_info_source, ldT, source_i, T, 0);

            STARNEIG_SANITY_CHECK_INF(
                0, MIN(_m, m-i), 0, MIN(_n, n-i), ldT, T, "T");

            starneig_join_sub_window(
                i, MIN(m, i+_m), j, MIN(n, j+_n),
                &packing_info_dest, ldT, dest_i, T, 1);
        }
    }
}

void starneig_cpu_set_to_identity(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    int size = packing_info.rend - packing_info.rbegin;

    struct starpu_matrix_interface **dest_i =
        (struct starpu_matrix_interface **)buffers;

    size_t ld;
    double *tmp = starneig_alloc_matrix(size, size, sizeof(double), &ld);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            tmp[i*ld+j] = i == j ? 1.0 : 0.0;

    starneig_join_window(&packing_info, ld, dest_i, tmp, 1);

    starneig_free_matrix(tmp);
}

void starneig_cpu_scan_diagonal(void *buffers[], void *cl_args)
{
    //
    // extract arguments
    //

    int num_masks, rbegin, cbegin;
    void (*func)(int, int, int, int, int, int, int, void const *, void const *,
        void const *, void **);
    void const *arg;
    struct packing_info packing_info_A, packing_info_B;
    struct range_packing_info packing_info_mask[SCAN_DIAGONAL_MAX_MASKS];

    starpu_codelet_unpack_args(cl_args,
        &num_masks, &rbegin, &cbegin, &func, &arg,
        &packing_info_A, &packing_info_B, packing_info_mask);

    int m = packing_info_A.rend - packing_info_A.rbegin;
    int n = packing_info_A.cend - packing_info_A.cbegin;
    int mask_size = packing_info_mask[0].end - packing_info_mask[0].begin;
    int generalized = 0 < packing_info_B.handles;

    //
    // extract buffers
    //

    int k = 0;

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    struct starpu_matrix_interface **B_i = NULL;
    if (generalized) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;
    }

    struct starpu_vector_interface **mask_i[SCAN_DIAGONAL_MAX_MASKS];
    for (int i = 0; i < num_masks; i++) {
        mask_i[i] = (struct starpu_vector_interface **)buffers + k;
        k += packing_info_mask[i].handles;
    }

    //
    // allocate and initialize
    //

    size_t ldA;
    void *A =
        starneig_alloc_matrix(m, n, packing_info_A.elemsize, &ldA);
    starneig_join_diag_window(&packing_info_A, ldA, A_i, A, 0);

    size_t ldB = 0;
    void *B = NULL;
    if (generalized) {
        B = starneig_alloc_matrix(
            m, n, packing_info_B.elemsize, &ldB);
        starneig_join_diag_window(&packing_info_B, ldB, B_i, B, 0);
    }

    void *mask[SCAN_DIAGONAL_MAX_MASKS];
    memset(mask, 0, sizeof(mask));
    for (int i = 0; i < num_masks; i++)
        mask[i] = malloc(mask_size*packing_info_mask[i].elemsize);

    //
    // process
    //

    func(mask_size, rbegin, cbegin, m, n, ldA, ldB, arg, A, B, mask);

    for (int i = 0; i < num_masks; i++)
        starneig_join_range(&packing_info_mask[i], mask_i[i], mask[i], 1);

    //
    // cleanup
    //

    starneig_free_matrix(A);
    starneig_free_matrix(B);

    for (int i = 0; i < num_masks; i++)
        free(mask[i]);
}

void starneig_cpu_set_to_zero(void *buffers[], void *cl_args)
{
    void *ptr = (void *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int m = STARPU_MATRIX_GET_NX(buffers[0]);
    int n = STARPU_MATRIX_GET_NY(buffers[0]);
    int ld = STARPU_MATRIX_GET_LD(buffers[0]);
    size_t elemsize = STARPU_MATRIX_GET_ELEMSIZE(buffers[0]);

    for (int i = 0; i < n; i++)
        memset(ptr+(size_t)i*ld*elemsize, 0, m*elemsize);
}
