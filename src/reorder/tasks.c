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
#include "tasks.h"
#include "cpu.h"
#ifdef STARNEIG_ENABLE_CUDA
#include "cuda.h"
#endif
#include "../common/common.h"
#include "../common/tiles.h"

#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for push_bulges codelet.
///
static void reorder_window_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info_A, packing_info_B;
    struct range_packing_info packing_info_selected;
    int window_size, threshold, swaps;
    starpu_codelet_unpack_args(task->cl_arg,
        &packing_info_selected, &packing_info_A, &packing_info_B,
        &window_size, &threshold, &swaps);

    parameters[0] = packing_info_B.handles != 0 ? 2.0 : 1.0;
    parameters[1] = packing_info_A.rend - packing_info_A.rbegin;
    parameters[2] = swaps;
}

///
/// @brief Multiple regression performance model for push_bulges codelet.
///
static struct starpu_perfmodel reorder_window_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_reorder_window_pm",
    .parameters = &reorder_window_parameters,
    .nparameters = 2,
    .parameters_names = (const char*[]) { "M", "N", "S" },
    .combinations = (unsigned*[]) {
        (unsigned[]) { 1U, 2U, 1U },
        (unsigned[]) { 1U, 2U, 0U }
    },
    .ncombinations = 2
};

#else

///
/// @brief Size base function for reorder_window codelet.
///
static size_t reorder_window_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info_A, packing_info_B;
    struct range_packing_info packing_info_selected;
    int window_size, threshold, swaps;
    starpu_codelet_unpack_args(task->cl_arg,
        &packing_info_selected, &packing_info_A, &packing_info_B,
        &window_size, &threshold, &swaps);

    int general = packing_info_B.handles != 0;

    if (general)
        return 2 * swaps * (packing_info_A.rend - packing_info_A.rbegin);
    else
        return swaps * (packing_info_A.rend - packing_info_A.rbegin);
}

struct starpu_perfmodel reorder_window_pm = {
    .type = STARPU_REGRESSION_BASED,
    .symbol = "starneig_reorder_window_pm",
    .size_base = &reorder_window_size_base
};

#endif

///
/// @brief reorder_window codelet performs necessary computations inside a
/// diagonal computation window and accumulates the related orthogonal
/// transformations into accumulator matrices (a.k.a local Q and Z matrices).
///
///  Windows that are smaller than a given threshold are processed in a scalar
///  manner. Larger windows are processed in a blocked manner. The threshold
///  and the "small" window size are given as arguments.
///
///  Arguments:
///   - eigenvalue selection vector packing information
///   - matrix A packing information
///   - matrix B packing information
///   - small window size
///   - small window threshold
///   - total number of involved diagonal block swaps (for performance models)
///
///  Buffers:
///   - local Q matrix (STARPU_W, end-begin rows/columns)
///   - local Z matrix (STARPU_W, end-begin rows/columns, optional)
///   - scratch matrix (STARPU_SCRATCH, end-begin rows/columns)
///   - scratch matrix (STARPU_SCRATCH, end-begin rows/columns, optional)
///   - eigenvalue selection bitmap tiles that correspond to the computation
///     window (STARPU_RW)
///   - matrix A tiles that correspond to the computation window (STARPU_RW,
///     non-zero tiles in column-major order)
///   - matrix B tiles that correspond to the computation window (STARPU_RW,
///     non-zero tiles in column-major order, optional)
///
static struct starpu_codelet reorder_window_cl = {
    .name = "starneig_reorder_window",
    .cpu_funcs = { starneig_cpu_reorder_window },
    .cpu_funcs_name = { "starneig_cpu_reorder_window" },
#if defined(STARNEIG_ENABLE_CUDA) && \
defined(STARNEIG_ENABLE_CUDA_REORDER_WINDOW)
    .cuda_funcs = { starneig_cuda_reorder_window },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &reorder_window_pm
};

void starneig_reorder_insert_window(
    int prio, int small_window_size, int small_window_threshold,
    struct window *window, starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    mpi_info_t mpi)
{
    window->lq_h = window->lz_h = NULL;

    int window_size = window->end - window->begin;
    if (window_size < 1)
        return;

#ifdef STARNEIG_ENABLE_MPI
    // figure out who is going to own the accumulator matrices
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(
            window->begin, window->begin, matrix_a);
#endif

    struct packing_helper *helper = starneig_init_packing_helper();

    // local Q matrix
    starpu_matrix_data_register(&window->lq_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            window->lq_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, window->lq_h, helper, 0);

    // local Z matrix
    if (matrix_b != NULL) {
        starpu_matrix_data_register(&window->lz_h, -1, 0,
            window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL)
            starpu_mpi_data_register_comm(
                window->lz_h, mpi->tag_offset++, owner,
                starneig_mpi_get_comm());
#endif
        starneig_pack_handle(STARPU_W, window->lz_h, helper, 0);
    }

    // scratch matrices
    starneig_pack_cached_scratch_matrix(
        window_size, window_size, sizeof(double), helper);
    if (matrix_b != NULL)
        starneig_pack_cached_scratch_matrix(
            window_size, window_size, sizeof(double), helper);

    // eigenvalue selection bitmap
    struct range_packing_info packing_info_selected;
    starneig_pack_range(STARPU_RW, window->begin, window->end, selected, helper,
        &packing_info_selected, 0);

    // corresponding tiles from the matrix A
    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, window->begin, window->end, matrix_a,
        helper, &packing_info_A, PACKING_MODE_UPPER_HESSENBERG);

    // corresponding tiles from the matrix B
    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, window->begin, window->end, matrix_b,
        helper, &packing_info_B, PACKING_MODE_UPPER_TRIANGULAR);

    //
    // insert task
    //

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &reorder_window_cl,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_selected, sizeof(packing_info_selected),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &small_window_size, sizeof(small_window_size),
            STARPU_VALUE, &small_window_threshold,
                sizeof(small_window_threshold),
            STARPU_VALUE, &window->swaps, sizeof(window->swaps),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &reorder_window_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_selected, sizeof(packing_info_selected),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &small_window_size, sizeof(small_window_size),
            STARPU_VALUE, &small_window_threshold,
                sizeof(small_window_threshold),
            STARPU_VALUE, &window->swaps, sizeof(window->swaps),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

#ifdef STARNEIG_ENABLE_MPI

    //
    // broadcast the transformation matrices
    //

    if (mpi != NULL) {
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), window->lq_h);
        starpu_mpi_data_set_rank_comm(
            window->lq_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        if (matrix_b != NULL) {
            starpu_mpi_get_data_on_all_nodes_detached(
                starneig_mpi_get_comm(), window->lz_h);
            starpu_mpi_data_set_rank_comm(
                window->lz_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        }
    }
#endif

    starneig_free_packing_helper(helper);
}
