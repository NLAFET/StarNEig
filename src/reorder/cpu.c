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
#include "common.h"
#include "lapack.h"
#include "../common/common.h"
#include "../common/sanity.h"
#include "../common/tiles.h"
#include "../common/math.h"
#include "../common/trace.h"

#include <math.h>
#include <starpu.h>

static void mark_tainted(int n, int *select)
{
    for (int i = 0; i < n; i++) {
        if (select[i] == 0)
            select[i] = TAINTED_DESELECTED;
        if (select[i] == 1)
            select[i] = TAINTED_SELECTED;
    }
}

static int reorder_window(
    int window_size, int threshold, int n, int ldQ, int ldZ, int ldA, int ldB,
    int *select, double *Q, double *Z, double *A, double *B)
{
    STARNEIG_SANITY_CHECK_SCHUR(0, n, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    int ret = 0;

    double *lQ = NULL;
    double *lZ = NULL;
    double *vT = NULL;
    double *hT = NULL;
    double *work = NULL;

    // check against tainted blocks
    for (int i = 0; i < n; i++) {
        if (1 < select[i]) {
            mark_tainted(n, select);
            goto cleanup;
        }
    }

    // allocate work space for dtgsen/dtrsen
    if (B != NULL)
        work = malloc((7*n+16)*sizeof(double));
    else
        work = malloc(3*n*sizeof(double));

    // make sure that the window is big enough and call
    // *_starneig_reorder_window directly if it is not
    if (n < threshold) {
        int m;

        if (B != NULL)
            ret = starneig_dtgsen(
                0, n, ldQ, ldZ, ldA, ldB, &m, select, Q, Z, A, B, work);
        else
            ret =  starneig_dtrsen(
                0, n, ldQ, ldA, &m, select, Q, A, work);

        // if an error occurred, mark the whole window tainted
        if (ret != 0)
            for (int i = 0; i < n; i++)
                select[i] = TAINTED_UNDEFINED;

        goto cleanup;
    }

    // scratch buffers for GEMM kernels
    size_t ldvT, ldhT;
    vT = starneig_alloc_matrix(n, window_size, sizeof(double), &ldvT);
    hT = starneig_alloc_matrix(window_size, n, sizeof(double), &ldhT);

    // local left-hand side transformation matrix
    size_t ldlQ;
    lQ = starneig_alloc_matrix(window_size, window_size, sizeof(double), &ldlQ);

    // local right-hand side transformation matrix
    size_t ldlZ = ldlQ;
    lZ = lQ;
    if (B != NULL)
        lZ = starneig_alloc_matrix(
            window_size, window_size, sizeof(double), &ldlZ);

    int begin = 0;
    int end = 0;

    // repeat until all chains have been processed
    while (1) {

        // place the window chain
        int in_chain = 0;
        for (int i = end; in_chain < window_size/2 && i < n; i++) {
            if (select[i]) {
                in_chain++;
                end = i+1;
            }
        }

        // extend the chain if it splits a 2-by-2 tile
        if (end < n && A[(end-1)*ldA+end] != 0.0) {
            in_chain++;
            end++;
        }

        // quit if the chain is empty
        if (in_chain == 0)
            goto cleanup;

        // place the first window such that it does not split any 2-by-2 tiles
        int wbegin = MAX(begin, end-window_size);
        if (0 < wbegin && A[(wbegin-1)*ldA+wbegin] != 0.0)
            wbegin++;
        int wend = end;

        // repeat until all windows in the current chain have been processed
        while(1) {
            // calculate window size
            int wsize = wend-wbegin;

            // initialize the local transformation matrices lQ and lZ
            starneig_init_local_q(wsize, ldlQ, lQ);
            if (B != NULL)
                starneig_init_local_q(wsize, ldlZ, lZ);

            // process the window
            int in_window;
            if (B != NULL)
                ret = starneig_dtgsen(wbegin, wend, ldlQ, ldlZ,
                    ldA, ldB, &in_window, select, lQ, lZ, A, B, work);
            else
                ret = starneig_dtrsen(
                    wbegin, wend, ldlQ, ldA, &in_window, select, lQ, A, work);

            starneig_small_gemm_updates(
                wbegin, wend, n, ldlQ, ldlZ, ldQ, ldZ, ldA, ldB, ldhT, ldvT,
                lQ, lZ, Q, Z, A, B, hT, vT);

            // if an error occurred, mark the current window and everything
            // below it tainted
            if (ret != 0) {
                mark_tainted(n, select);
                for (int i = wbegin; i < wend; i++)
                    select[i] = TAINTED_UNDEFINED;
                goto cleanup;
            }

            // quit if this was the topmost window in the chain
            if (wbegin == begin)
                break;

            // place the next window such that it does not split any 2-by-2
            // tiles
            wend = wbegin + in_window;
            wbegin = MAX(begin, wend-window_size);
            if (0 < wbegin && A[(wbegin-1)*ldA+wbegin] != 0.0)
                wbegin++;
        }

        // advance downwards
        begin += in_chain;
    }

cleanup:

    free(work);
    starneig_free_matrix(vT);
    starneig_free_matrix(hT);
    starneig_free_matrix(lQ);
    if (B != NULL)
        starneig_free_matrix(lZ);

    STARNEIG_SANITY_CHECK_SCHUR(0, n, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_cpu_reorder_window(void *buffers[], void *cl_arg)
{
    struct packing_info packing_info_A, packing_info_B;
    struct range_packing_info packing_info_selected;
    int window_size, threshold, swaps;
    starpu_codelet_unpack_args(cl_arg,
        &packing_info_selected, &packing_info_A, &packing_info_B,
        &window_size, &threshold, &swaps);

    STARNEIG_EVENT_BEGIN(&packing_info_A, starneig_event_red);

    int size = packing_info_A.rend - packing_info_A.rbegin;
    int general = packing_info_B.handles != 0;

    int k = 0;

    // local matrix Q
    struct starpu_matrix_interface *lQ_i =
        buffers[k++];
    double *lQ_ptr = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    int lQ_ld = STARPU_MATRIX_GET_LD(lQ_i);
    starneig_init_local_q(size, lQ_ld, lQ_ptr);

    // local matrix Z
    double *lZ_ptr = NULL;
    int lZ_ld = 0;
    if (general) {
        struct starpu_matrix_interface *lZ_i = buffers[k++];
        lZ_ptr = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        lZ_ld = STARPU_MATRIX_GET_LD(lZ_i);
        starneig_init_local_q(size, lZ_ld, lZ_ptr);
    }

    // local matrix A
    struct starpu_matrix_interface *lA_i = buffers[k++];
    double *lA_ptr = (double*) STARPU_MATRIX_GET_PTR(lA_i);
    int lA_ld = STARPU_MATRIX_GET_LD(lA_i);

    // local matrix B
    double *lB_ptr = NULL;
    int lB_ld = 0;
    if (general) {
        struct starpu_matrix_interface *lB_i = buffers[k++];
        lB_ptr = (double*) STARPU_MATRIX_GET_PTR(lB_i);
        lB_ld = STARPU_MATRIX_GET_LD(lB_i);
    }

    // eigenvalue selection bitmap

    int *selected = malloc((size)*sizeof(int));

    struct starpu_vector_interface **select_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_selected.handles;

    starneig_join_range(&packing_info_selected, select_i, selected, 0);

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    starneig_join_diag_window(&packing_info_A, lA_ld, A_i, lA_ptr, 0);

    // corresponding tiles from the matrix B

    struct starpu_matrix_interface **B_i = NULL;
    if (general) {
        B_i = (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;

        starneig_join_diag_window(&packing_info_B, lB_ld, B_i, lB_ptr, 0);
    }

    // reorder
    reorder_window(
        window_size, threshold, size, lQ_ld, lZ_ld, lA_ld, lB_ld,
        selected, lQ_ptr, lZ_ptr, lA_ptr, lB_ptr);

    // store result

    starneig_join_range(&packing_info_selected, select_i, selected, 1);

    starneig_join_diag_window(&packing_info_A, lA_ld, A_i, lA_ptr, 1);

    if (general)
        starneig_join_diag_window(&packing_info_B, lB_ld, B_i, lB_ptr, 1);

    free(selected);

    STARNEIG_EVENT_END();
}
