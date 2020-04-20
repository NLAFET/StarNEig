///
/// @file This file contains the Hessenberg reduction specific task definitions
/// and task insertion functions.
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
#include <limits.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

///
/// @brief Reduces a column inside a (rend-rbegin) X (cend-cbegin) panel.
///
///  Arguments:
///   - (int)                       The index of the currect column inside the
///                                 panel.
///   - (struct range_packing_info) Packing info for an intemediate vector
///                                 interface used in the trailing matrix
///                                 operation.
///
///  Buffers:
///   - Y matrix (STARPU_R, rend-rbegin rows, cend-cbegin columns, defined only
///     if 0 < i).
///   - V matrix (STARPU_RW if 0 < i, STARPU_W otherwise, rend-rbegin rows,
///     cend-cbegin columns).
///   - T matrix (STARPU_RW if 0 < i, STARPU_W otherwise, cend-cbegin
///     rows/columns).
///   - Panel matrix (STARPU_RW, rend-rbegin rows, cend-cbegin columns).
///
static struct starpu_codelet prepare_column_cl = {
    .name = "starneig_prepare_column",
    .cpu_funcs = { starneig_hessenberg_cpu_prepare_column },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_prepare_column" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_prepare_column_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Performs a (rend-rbegin) X (cend-cbegin) trailing matrix operation.
///
///  Arguments:
///   - (struct packing_info)       Packing info for the trailing matrix.
///   - (struct range_packing_info) Packing info for an intemediate vector
///                                 interface used in the trailing matrix
///                                 operation.
///   - (struct range_packing_info) Packing info for an intemediate vector
///                                 interface computed in the trailing matrix
///                                 operation.
///
///  Buffers:
///   - Matrix tiles that correspond to the trailing matrix (STARPU_R).
///   - Intemediate vector tiles (STARPU_R).
///   - Intemediate vector tiles (STARPU_W).
///
static struct starpu_codelet compute_column_cl = {
    .name = "starneig_compute_column",
    .cpu_funcs = { starneig_hessenberg_cpu_compute_column },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_compute_column" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_cuda_compute_column },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_compute_column_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Finalizes a column reduction inside a (rend-rbegin) X (cend-cbegin)
/// panel.
///
///  Arguments:
///   - (int)                       The index of the currect column inside the
///   - (struct range_packing_info) Packing info for an intemediate vector
///                                 interface used in the trailing matrix
///                                 operation.
///
///  Buffers:
///   - V matrix (STARPU_R, rend-rbegin rows, cend-cbegin columns).
///   - T matrix (STARPU_RW,  cend-cbegin rows/columns).
///   - Y matrix (STARPU_RW, rend-rbegin rows, cend-cbegin columns).
///   - Intemediate vector tiles (STARPU_R).
///
static struct starpu_codelet finish_column_cl = {
    .name = "starneig_finish_column",
    .cpu_funcs = { starneig_hessenberg_cpu_finish_column },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_finish_column" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_finish_column_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Updates a block column of the trailing matrix from right.
///
///  Arguments:
///   - packing_info  block column packing information
///   - nb            panel width
///   - roffset       first row of the of the trailing matrix that belongs to
///                   the block row
///   - coffset       first column of the of the trailing matrix that belongs to
///                   the block column
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - Y matrix (STARPU_R, rend-rbegin rows, nb columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - matrix A tiles that correspond to the block column (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_trail_right_cl = {
    .name = "starneig_update_trail_right",
    .cpu_funcs = { starneig_hessenberg_cpu_update_trail_right },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_update_trail_right" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_cuda_update_trail_right },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_update_trail_right_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Updates a block column of the trailing matrix from left.
///
///  Arguments:
///   - packing_info  block column packing information
///   - nb            panel width
///   - offset        first column of the of the trailing matrix that belongs to
///                   the block column
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH, cend-cbegin rows, nb columns)
///   - matrix A tiles that correspond to the block column (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_left_a_cl = {
    .name = "starneig_update_left_a",
    .cpu_funcs = { starneig_hessenberg_cpu_update_left_a },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_update_left_a" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_cuda_update_left_a },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_update_left_a_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Updates a block column of the trailing matrix from left.
///
///  Arguments:
///   - packing_info  block column packing information
///   - nb            panel width
///   - offset        first column of the of the trailing matrix that belongs to
///                   the block column
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH, cend-cbegin rows, nb columns)
///   - matrix A tiles that correspond to the block column (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_left_b_cl = {
    .name = "starneig_update_left_b",
    .cpu_funcs = { starneig_hessenberg_cpu_update_left_b },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_update_left_b" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_cuda_update_left_b },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_update_left_b_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Updates a block column of the trailing matrix from right.
///
///  Arguments:
///   - packing_info  block column packing information
///   - nb            panel width
///   - offset        first column of the of the trailing matrix that belongs to
///                   the block column
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH, cend-cbegin rows, nb columns)
///   - matrix A tiles that correspond to the block column (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_right_a_cl = {
    .name = "starneig_update_right_a",
    .cpu_funcs = { starneig_hessenberg_cpu_update_right_a },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_update_right_a" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_update_right_a_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Updates a block column of the trailing matrix from right.
///
///  Arguments:
///   - packing_info  block column packing information
///   - nb            panel width
///   - offset        first column of the of the trailing matrix that belongs to
///                   the block column
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH, cend-cbegin rows, nb columns)
///   - matrix A tiles that correspond to the block column (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_right_b_cl = {
    .name = "starneig_update_right_b",
    .cpu_funcs = { starneig_hessenberg_cpu_update_right_b },
    .cpu_funcs_name = { "starneig_hessenberg_cpu_update_right_b" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_update_right_b_pm"
    }}
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_hessenberg_insert_prepare_column(
    int prio, int i, int begin, int end,
    starpu_data_handle_t Y_h, starpu_data_handle_t V_h,
    starpu_data_handle_t T_h, starpu_data_handle_t P_h,
    starneig_vector_t v, mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    if (0 < i) {
        starneig_pack_handle(STARPU_R,  Y_h, helper, 0);
        starneig_pack_handle(STARPU_RW, V_h, helper, 0);
        starneig_pack_handle(STARPU_RW, T_h, helper, 0);
    }
    else {
        starneig_pack_handle(STARPU_W, V_h, helper, 0);
        starneig_pack_handle(STARPU_W, T_h, helper, 0);
    }
    starneig_pack_handle(STARPU_RW, P_h, helper, 0);

    struct range_packing_info v_pi;
    starneig_pack_range(STARPU_W, begin+i, end, v, helper, &v_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &prepare_column_cl,
            STARPU_PRIORITY, prio,
            STARPU_EXECUTE_ON_NODE,
                starneig_vector_get_elem_owner(0, v),
            STARPU_VALUE, &i, sizeof(i),
            STARPU_VALUE, &v_pi, sizeof(v_pi),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &prepare_column_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &i, sizeof(i),
            STARPU_VALUE, &v_pi, sizeof(v_pi),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_compute_column(
    int prio, int rbegin, int rend, int cbegin, int cend,
    starneig_matrix_t matrix_a, starneig_vector_t v,
    starneig_vector_t y, mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    struct packing_info A_pi;
    starneig_pack_window(
        STARPU_R, rbegin, rend, cbegin, cend, matrix_a, helper, &A_pi, 0);

    struct range_packing_info v_pi;
    starneig_pack_range(STARPU_R, cbegin, cend, v, helper, &v_pi, 0);

    struct range_packing_info y_pi;
    starneig_pack_range(
        STARPU_RW | STARPU_COMMUTE, rbegin, rend, y, helper, &y_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &compute_column_cl,
            STARPU_PRIORITY, prio,
            STARPU_EXECUTE_ON_NODE,
                starneig_matrix_get_elem_owner(rbegin, cbegin, matrix_a),
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &v_pi, sizeof(v_pi),
            STARPU_VALUE, &y_pi, sizeof(y_pi),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &compute_column_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &v_pi, sizeof(v_pi),
            STARPU_VALUE, &y_pi, sizeof(y_pi),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_finish_column(
    int prio, int i, int begin, int end,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starpu_data_handle_t Y_h, starneig_vector_t y, mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R,  V_h, helper, 0);
    starneig_pack_handle(STARPU_RW, T_h, helper, 0);
    if (0 < i)
        starneig_pack_handle(STARPU_RW, Y_h, helper, 0);
    else
        starneig_pack_handle(STARPU_W, Y_h, helper, 0);

    struct range_packing_info y_pi;
    starneig_pack_range(STARPU_R, begin, end, y, helper, &y_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &finish_column_cl,
            STARPU_PRIORITY, prio,
            STARPU_EXECUTE_ON_NODE,
                starneig_vector_get_elem_owner(0, y),
            STARPU_VALUE, &i, sizeof(i),
            STARPU_VALUE, &y_pi, sizeof(y_pi),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &finish_column_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &i, sizeof(i),
            STARPU_VALUE, &y_pi, sizeof(y_pi),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_update_trail_right(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    int roffset, int coffset, starpu_data_handle_t V_h,
    starpu_data_handle_t T_h, starpu_data_handle_t Y_h,
    starneig_matrix_t matrix_a, mpi_info_t mpi)
{
    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, V_h, helper, 0);
    starneig_pack_handle(STARPU_R, Y_h, helper, 0);

    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);

    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, rbegin, rend, cbegin, cend,
        matrix_a, helper, &packing_info, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_trail_right_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_matrix_get_elem_owner(rbegin, cbegin, matrix_a),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &roffset, sizeof(roffset),
            STARPU_VALUE, &coffset, sizeof(coffset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_trail_right_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &roffset, sizeof(roffset),
            STARPU_VALUE, &coffset, sizeof(coffset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_update_left_a(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb, int offset,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_t A, starneig_matrix_t W, mpi_info_t mpi)
{
    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, V_h, helper, 0);
    starneig_pack_handle(STARPU_R, T_h, helper, 0);

    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(
        cend-cbegin, nb, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(
        cend-cbegin, nb, sizeof(double), helper);

    struct packing_info A_pi;
    starneig_pack_window(
        STARPU_R, rbegin, rend, cbegin, cend, A, helper, &A_pi, 0);

    struct packing_info W_pi;
    starneig_pack_window(
        STARPU_RW | STARPU_COMMUTE, cbegin, cend, 0, nb, W, helper, &W_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_left_a_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_matrix_get_elem_owner(rbegin, cbegin, A),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_left_a_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_update_left_b(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb, int offset,
    starpu_data_handle_t V_h, starneig_matrix_t W,
    starneig_matrix_t A, mpi_info_t mpi)
{
    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, V_h, helper, 0);

    starneig_pack_cached_scratch_matrix(
        cend-cbegin, nb, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);

    struct packing_info W_pi;
    starneig_pack_window(
        STARPU_R, cbegin, cend, 0, nb, W, helper, &W_pi, 0);

    struct packing_info A_pi;
    starneig_pack_window(
        STARPU_RW, rbegin, rend, cbegin, cend, A, helper, &A_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_left_b_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_matrix_get_elem_owner(rbegin, cbegin, A),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_left_b_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_update_right_a(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb, int offset,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_t A, starneig_matrix_t W, mpi_info_t mpi)
{
    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, V_h, helper, 0);
    starneig_pack_handle(STARPU_R, T_h, helper, 0);

    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(
        rend-rbegin, nb, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(
        rend-rbegin, nb, sizeof(double), helper);

    struct packing_info A_pi;
    starneig_pack_window(
        STARPU_R, rbegin, rend, cbegin, cend, A, helper, &A_pi, 0);

    struct packing_info W_pi;
    starneig_pack_window(
        STARPU_RW | STARPU_COMMUTE, rbegin, rend, 0, nb, W, helper, &W_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_right_a_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_matrix_get_elem_owner(rbegin, cbegin, A),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_right_a_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_insert_update_right_b(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb, int offset,
    starpu_data_handle_t V_h, starneig_matrix_t W,
    starneig_matrix_t A, mpi_info_t mpi)
{
    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, V_h, helper, 0);

    starneig_pack_cached_scratch_matrix(
        rend-rbegin, nb, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);

    struct packing_info W_pi;
    starneig_pack_window(
        STARPU_R, rbegin, rend, 0, nb, W, helper, &W_pi, 0);

    struct packing_info A_pi;
    starneig_pack_window(
        STARPU_RW, rbegin, rend, cbegin, cend, A, helper, &A_pi, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_right_b_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_matrix_get_elem_owner(rbegin, cbegin, A),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_right_b_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &A_pi, sizeof(A_pi),
            STARPU_VALUE, &W_pi, sizeof(W_pi),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &offset, sizeof(offset),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}
