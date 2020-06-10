///
/// @file
///
/// @brief This file contains task definitions and related task insertion
/// functions that are used in the StarPU-bases QR algorithm.
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
#include "../common/common.h"
#include "../common/utils.h"
#include "../common/tiles.h"

#include <starpu_scheduler.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

///
/// @brief Size base function for push_inf_top codelet.
///
static size_t push_inf_top_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    double thres_inf;
    struct packing_info packing_info_A, packing_info_B;
    int top, bottom;
    starpu_codelet_unpack_args(task->cl_arg,
        &thres_inf, &packing_info_A, &packing_info_B, &top, &bottom);

    return packing_info_A.rend - packing_info_A.rbegin;
}

///
/// @brief push_inf_top codelet takes a matrix pencil (A,B) and pushes a set of
/// infinite eigenvalues across a diagonal window. The related orthogonal
/// transformations are accumulated to local transformation matrices. The
/// codelet can also deflate the infite eigenvalues from the top of the window.
///
///  Arguments:
///   - tiny diagonal entry threshold for the matrix B
///   - packing information for the matrix A
///   - packing information for the matrix B
///   - top flag
///   - bottom flag
///
///  Buffers:
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix B tiles that correspond to the diagonal window (STARPU_RW,
///     optional)
///
static struct starpu_codelet push_inf_top_cl = {
    .name = "starneig_schur_push_inf_top",
    .cpu_funcs = { starneig_cpu_push_inf_top },
    .cpu_funcs_name = { "starneig_cpu_push_inf_top" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_NL_REGRESSION_BASED,
        .symbol = "starneig_schur_push_inf_top_pm",
        .size_base = &push_inf_top_size_base
    }}
};

////////////////////////////////////////////////////////////////////////////////

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for push_bulges codelet.
///
static void push_bulges_parameters(
    struct starpu_task *task, double *parameters)
{
    double thres_a, thres_b, thres_inf;
    struct range_packing_info packing_info_shifts_real;
    struct range_packing_info packing_info_shifts_imag;
    struct range_packing_info packing_info_aftermath;
    struct packing_info packing_info_A, packing_info_B;
    bulge_chasing_mode_t mode;
    starpu_codelet_unpack_args(task->cl_arg, &thres_a, &thres_b, &thres_inf,
        &packing_info_shifts_real, &packing_info_shifts_imag,
        &packing_info_aftermath, &packing_info_A, &packing_info_B, &mode);

    parameters[0] = packing_info_A.rend - packing_info_A.rbegin;
    parameters[1] =
        packing_info_shifts_real.end - packing_info_shifts_real.begin;
}

///
/// @brief Multiple regression performance model for push_bulges codelet.
///
static struct starpu_perfmodel push_bulges_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_push_bulges_pm",
    .parameters = &push_bulges_parameters,
    .nparameters = 2,
    .parameters_names = (const char*[]) { "N", "S" },
    .combinations = (unsigned*[]) {
        (unsigned[]) { 2U, 1U },
        (unsigned[]) { 2U, 0U }
    },
    .ncombinations = 2
};

#else

///
/// @brief Size base function for push_bulges codelet.
///
static size_t push_bulges_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    double thres_a, thres_b, thres_inf;
    struct range_packing_info packing_info_shifts_real;
    struct range_packing_info packing_info_shifts_imag;
    struct range_packing_info packing_info_aftermath;
    struct packing_info packing_info_A, packing_info_B;
    bulge_chasing_mode_t mode;
    starpu_codelet_unpack_args(task->cl_arg, &thres_a, &thres_b, &thres_inf,
        &packing_info_shifts_real, &packing_info_shifts_imag,
        &packing_info_aftermath, &packing_info_A, &packing_info_B, &mode);

    return packing_info_A.rend - packing_info_A.rbegin;
}

///
/// @brief Linear regression performance model for push_bulges codelet.
///
static struct starpu_perfmodel push_bulges_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_push_bulges_pm",
    .size_base = &push_bulges_size_base
};

#endif

///
/// @brief push_bulges codelet takes a matrix pencil (A,B) and pushes a set of
/// bulges across a diagonal window. The related orthogonal transformations are
/// accumulated to local transformation matrices. The codelet can also perform
/// vigilant deflation and infinite eigenvalue checks.
///
///  Arguments:
///   - packing information for the shift vector (real parts)
///   - packing information for the shift vector (imaginary parts)
///   - packing information for the aftermath vector
///   - packing information for the matrix A
///   - packing information for the matrix B
///   - bulge chaining mode
///
///  Buffers:
///   - tiny off-diagonal entry threshold for the matrix A
///   - tiny off-diagonal entry threshold for the matrix B
///   - tiny diagonal entry threshold for the matrix B
///   - shifts vector (real parts) tiles that correspond to the used shifts
///     (STARPU_R)
///   - shifts vector (imaginary parts) tiles that correspond to the used shifts
///     (STARPU_R)
///   - local left-hand side transformation matrix (STARPU_W)
///   - local right-hand side transformation matrix (STARPU_W, optional)
///   - aftermath vector tiles that correspond to the diagonal window
///     (STARPU_RW, optional)
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix B tiles that correspond to the diagonal window (STARPU_RW,
///     optional)
///
static struct starpu_codelet push_bulges_cl = {
    .name = "starneig_schur_push_bulges",
    .cpu_funcs = { starneig_cpu_push_bulges },
    .cpu_funcs_name = { "starneig_cpu_push_bulges" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &push_bulges_pm
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Size base function for aggressively_deflate codelet.
///
static size_t aggressively_deflate_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    double thres_a, thres_b, thres_inf;
    struct range_packing_info packing_info_shifts_real;
    struct range_packing_info packing_info_shifts_imag;
    struct packing_info packing_info_A, packing_info_B;
    starpu_codelet_unpack_args(task->cl_arg, &thres_a, &thres_b, &thres_inf,
        &packing_info_shifts_real, &packing_info_shifts_imag,
        &packing_info_A, &packing_info_B);

    return packing_info_A.rend - packing_info_A.rbegin;
}

///
/// @brief aggressively_deflate_sep codelet takes a matrix A and performs an
/// aggressive early deflation procedure inside a diagonal window. The related
/// orthogonal transformations are accumulated to local transformation matrix.
///
///  The codelet assumes that the diagonal window and the local transformation
///  matrices are padded from top and left with one row and one column.
///
///  Arguments:
///   - tiny off-diagonal entry threshold for the matrix A
///   - tiny off-diagonal entry threshold for the matrix B
///   - tiny diagonal entry threshold for the matrix B
///   - packing information for the shift vector (real parts)
///   - packing information for the shift vector (imaginary parts)
///   - packing information for the matrix A
///
///  Buffers:
///   - return status (STARPU_W)
///   - local transformation matrix (STARPU_W)
///   - shifts vector (real parts) tiles that correspond to the used shifts
///     (STARPU_W)
///   - shifts vector (imaginary parts) tiles that correspond to the used shifts
///     (STARPU_W)
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///
static struct starpu_codelet aggressively_deflate_sep_cl = {
    .name = "starneig_schur_aggressively_deflate",
    .cpu_funcs = { starneig_cpu_aggressively_deflate },
    .cpu_funcs_name = { "starneig_cpu_aggressively_deflate" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_schur_aggressively_deflate_sep_pm",
        .size_base = &aggressively_deflate_size_base
    }}
};

///
/// @brief aggressively_deflate_gep codelet takes a matrix pencil (A,B) and
/// performs an aggressive early deflation procedure inside a diagonal window.
/// The related orthogonal transformations are accumulated to local
/// transformation matrices.
///
///  The codelet assumes that the diagonal window and the local transformation
///  matrices are padded from top and left with one row and one column.
///
///  Arguments:
///   - tiny off-diagonal entry threshold for the matrix A
///   - tiny off-diagonal entry threshold for the matrix B
///   - tiny diagonal entry threshold for the matrix B
///   - packing information for the shift vector (real parts)
///   - packing information for the shift vector (imaginary parts)
///   - packing information for the matrix A
///   - packing information for the matrix B
///
///  Buffers:
///   - return status (STARPU_W)
///   - local left-hand side transformation matrix (STARPU_W)
///   - local right-hand side transformation matrix (STARPU_W)
///   - shifts vector (real parts) tiles that correspond to the used shifts
///     (STARPU_W)
///   - shifts vector (imaginary parts) tiles that correspond to the used shifts
///     (STARPU_W)
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix B tiles that correspond to the diagonal window (STARPU_RW)
///
static struct starpu_codelet aggressively_deflate_gep_cl = {
    .name = "starneig_schur_aggressively_deflate",
    .cpu_funcs = { starneig_cpu_aggressively_deflate },
    .cpu_funcs_name = { "starneig_cpu_aggressively_deflate" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_schur_aggressively_deflate_gep_pm",
        .size_base = &aggressively_deflate_size_base,
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Size base function for small_schur codelet.
///
static size_t small_schur_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    double thres_a, thres_b, thres_inf;
    struct packing_info packing_info_A, packing_info_B;
    starpu_codelet_unpack_args(task->cl_arg,
        &thres_a, &thres_b, &thres_inf, &packing_info_A, &packing_info_B);

    return packing_info_A.rend - packing_info_A.rbegin;
}

///
/// @brief small_schur codelet takes a matrix pencil (A,B) and reduces a
/// diagonal window to a Schur form. The related orthogonal transformations are
/// accumulated to local transformation matrices.
///
///  Arguments:
///   - tiny off-diagonal entry threshold for the matrix A
///   - tiny off-diagonal entry threshold for the matrix B
///   - tiny diagonal entry threshold for the matrix B
///   - packing information for the sub-matrix that is part of the matrix A
///   - packing information for the sub-matrix that is part of the matrix B
///
///  Buffers:
///
///   - return status (STARPU_W)
///   - local left-hand side transformation matrix (STARPU_W)
///   - local right-hand side transformation matrix (STARPU_W, optional)
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix B tiles that correspond to the diagonal window (STARPU_RW,
///     optional)
///
static struct starpu_codelet small_schur_cl = {
    .name = "starneig_schur_small_schur",
    .cpu_funcs = { starneig_cpu_small_schur },
    .cpu_funcs_name = { "starneig_cpu_small_schur" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_schur_small_schur_pm",
        .size_base = &small_schur_size_base
    }}
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Size base function for small_hessenberg codelet.
///
static size_t small_hessenberg_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info_A, packing_info_B;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info_A, &packing_info_B);

    return packing_info_A.rend - packing_info_A.rbegin;
}

///
/// @brief small_hessenberg codelet takes a matrix pencil (A,B) and reduces a
/// diagonal window to a Hessenberg-triangular form. The related orthogonal
/// transformations are accumulated to local transformation matrices.
///
///  Arguments:
///   - packing information for the sub-matrix that is part of the matrix A
///   - packing information for the sub-matrix that is part of the matrix B
///
///  Buffers:
///   - local left-hand side transformation matrix (STARPU_W)
///   - local right-hand side transformation matrix (STARPU_W, optional)
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix B tiles that correspond to the diagonal window (STARPU_RW,
///     optional)
///
static struct starpu_codelet small_hessenberg_cl = {
    .name = "starneig_schur_small_hessenberg",
    .cpu_funcs = { starneig_cpu_small_hessenberg },
    .cpu_funcs_name = { "starneig_cpu_small_hessenberg" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_schur_small_hessenberg_pm",
        .size_base = &small_hessenberg_size_base
    }}
};

///
/// @brief form_spike codelet extracts the first row from the left-hand side AED
/// transformation matrix. That is, codelet forms the spike base.
///
///  The task assumes that the AED window and the AED transformation matrices
///  are padded from top and left with one row and one column. The first row
///  from left-hand side AED transformation matrix (spike base) and the
///  sub-diagonal entry to the left of the AED window (spike inducer) are stored
///  separately. The spike base is also assumed to be padded from top with one
///  row.
///
///  Arguments:
///   - packing information for the first row of the left-hand side AED
///     transformation matrix
///   - packing information for the spike base
///
///  Buffers:
///   - matrix tiles that correspond to the first row of the left-hand side AED
///     transformation matrix (STARPU_R)
///   - vector tiles that correspond to the spike base (STARPU_W)
///
static struct starpu_codelet form_spike_cl = {
    .name = "starneig_schur_form_spike",
    .cpu_funcs = { starneig_cpu_form_spike },
    .cpu_funcs_name = { "starneig_cpu_form_spike" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

///
/// @brief embed_spike codelet embeds the spike to the padded AED window.
///
///  The task assumes that the AED window and the AED transformation matrices
///  are padded from top and left with one row and one column. The first row
///  from left-hand side AED transformation matrix (spike base) and the
///  sub-diagonal entry to the left of the AED window (spike inducer) are stored
///  separately. The spike base is also assumed to be padded from top with one
///  row.
///
///  Arguments:
///   - packing information for the spike base
///   - packing information for the column to the left of the AED window
///
///  Buffers:
///   - vector tiles that correspond to the spike base (STARPU_R)
///   - matrix tiles that correspond to the column to the left of the AED window
///     (STARPU_RW)
///
static struct starpu_codelet embed_spike_cl = {
    .name = "starneig_schur_embed_spike",
    .cpu_funcs = { starneig_cpu_embed_spike },
    .cpu_funcs_name = { "starneig_cpu_embed_spike" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

///
/// @brief deflate codelet codelet takes a matrix pencil (A,B) and performs
/// deflation checks inside a diagonal window. The related orthogonal
/// transformations are accumulated to local transformation matrices.
///
///  The task assumes that the AED window and the AED transformation matrices
///  are padded from top and left with one row and one column. The first row
///  from left-hand side AED transformation matrix (spike base) and the
///  sub-diagonal entry to the left of the AED window (spike inducer) are stored
///  separately. The spike base is also assumed to be padded from top with one
///  row.
///
///  Arguments:
///   - tiny off-diagonal entry threshold for the matrix A
///   - packing information for the spike base
///   - packing information for the matrix A
///   - packing information for the matrix B
///   - window offset
///   - deflation flag (if zero, then only reordering is performed)
///   - corner flag (if zero, the window is padded from bottom right corner)
///
///  Buffers:
///   - spike inducer (STARPU_R)
///   - return status (STARPU_RW)
///   - local left-hand side transformation matrix (STARPU_W)
///   - local right-hand side transformation matrix (STARPU_W, optional)
///   - spike base tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix A tiles that correspond to the diagonal window (STARPU_RW)
///   - matrix B tiles that correspond to the diagonal window (STARPU_RW,
///     optional)
///
static struct starpu_codelet deflate_cl = {
    .name = "starneig_schur_deflate",
    .cpu_funcs = { starneig_cpu_deflate },
    .cpu_funcs_name = { "starneig_cpu_deflate" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = (struct starpu_perfmodel[]) {{
        .type = STARPU_REGRESSION_BASED,
        .symbol = "starneig_schur_deflate_pm"
    }}
};

///
/// @brief extract_shifts codelet extracts shifts from the diagonal of a matrix
/// pencil (A, B).
///
///  Arguments:
///   - matrix A packing information
///   - matrix B packing information
///   - shift vector packing information (real parts)
///   - shift vector packing information (imaginary parts)
///   - first diagonal entry of the matrix to be extracted
///   - last diagonal entry of the matrix to be extracted + 1
///
///  Buffers:
///   - matrix tiles that correspond to the matrix A (STARPU_R)
///   - matrix tiles that correspond to the matrix B (STARPU_R, optional)
///   - vector tiles that correspond to the shifts vector (real parts)
///     (STARPU_RW)
///   - vector tiles that correspond to the vector vector (imaginary parts)
///     (STARPU_RW)
///
static struct starpu_codelet extract_shifts_cl = {
    .name = "starneig_extract_shifts",
    .cpu_funcs = { starneig_cpu_extract_shifts},
    .cpu_funcs_name = { "starneig_cpu_extract_shifts" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

///
/// @brief compute_norm_a codelet compute the square of the Frobenius norm of
/// a tile.
///
///  Arguments:
///   - first row of the tile to be processed
///   - last row of the tile to be processed + 1
///   - first column of the tile to processed
///   - last column of the tile to be processed + 1
///
///  Buffers:
///   - tile (STARPU_R)
///   - square of the Frobenius norm (STARPU_W)
///
static struct starpu_codelet compute_norm_a_cl = {
    .name = "starneig_compute_norm_a",
    .cpu_funcs = { starneig_cpu_compute_norm_a},
    .cpu_funcs_name = { "starneig_cpu_compute_norm_a" },
    .modes = { STARPU_R, STARPU_W },
    .nbuffers = 2
};

///
/// @brief compute_norm_b codelet finishes the Frobenius norm computation.
///
///  Arguments:
///   - the number of partila sums
///
///  Buffers:
///   - partial sums (STARPU_R)
///   - Frobenius norm (STARPU_W)
///
static struct starpu_codelet compute_norm_b_cl = {
    .name = "starneig_compute_norm_b",
    .cpu_funcs = { starneig_cpu_compute_norm_b},
    .cpu_funcs_name = { "starneig_cpu_compute_norm_b" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_schur_insert_push_inf_top(
    int begin, int end, int top, int bottom, int prio, double thres_inf,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *lQ_h, starpu_data_handle_t *lZ_h, mpi_info_t mpi)
{
    *lQ_h = *lZ_h = NULL;

    int window_size = end - begin;
    if (window_size < 1)
        return;

    // figure out who is going to execute the task

#ifdef STARNEIG_ENABLE_MPI
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(begin, begin, matrix_a);
#endif

    struct packing_helper *helper = starneig_init_packing_helper();

    // local left-hand size transformation matrix

    starpu_matrix_data_register(lQ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *lQ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *lQ_h, helper, 0);

    // local right-hand side transformation matrix

    starpu_matrix_data_register(lZ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *lZ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *lZ_h, helper, 0);

    // corresponding tiles from the matrix A

    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_a, helper,
        &packing_info_A, PACKING_MODE_UPPER_HESSENBERG);

    // corresponding tiles from the matrix B

    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_b, helper,
        &packing_info_B, PACKING_MODE_UPPER_TRIANGULAR);

    //
    // insert task
    //

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &push_inf_top_cl,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &top, sizeof(top),
            STARPU_VALUE, &bottom, sizeof(bottom),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &push_inf_top_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &top, sizeof(top),
            STARPU_VALUE, &bottom, sizeof(bottom),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL) {
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), *lQ_h);
        starpu_mpi_data_set_rank_comm(
            *lQ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        if (*lZ_h != *lQ_h) {
            starpu_mpi_get_data_on_all_nodes_detached(
                starneig_mpi_get_comm(), *lZ_h);
            starpu_mpi_data_set_rank_comm(
                *lZ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        }
    }
#endif

    starneig_free_packing_helper(helper);
}

void starneig_schur_insert_push_bulges(
    int begin, int end, int shifts_begin, int shifts_end,
    bulge_chasing_mode_t mode, int prio,
    double thres_a, double thres_b, double thres_inf,
    starneig_vector_descr_t shifts_real, starneig_vector_descr_t shifts_imag,
    starneig_vector_descr_t aftermath,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *lQ_h, starpu_data_handle_t *lZ_h, mpi_info_t mpi)
{
    *lQ_h = *lZ_h = NULL;

    int window_size = end - begin;
    if (window_size < 1)
        return;

    // figure out who is going to execute the task

#ifdef STARNEIG_ENABLE_MPI
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(begin, begin, matrix_a);
#endif

    struct packing_helper *helper = starneig_init_packing_helper();

    // shifts (real parts)

    struct range_packing_info packing_info_shifts_real;
    starneig_pack_range(STARPU_R, shifts_begin, shifts_end, shifts_real,
        helper, &packing_info_shifts_real, 0);

    // shifts (imaginary parts)

    struct range_packing_info packing_info_shifts_imag;
    starneig_pack_range(STARPU_R, shifts_begin, shifts_end, shifts_imag,
        helper, &packing_info_shifts_imag, 0);

    // local left-hand size transformation matrix

    starpu_matrix_data_register(lQ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *lQ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *lQ_h, helper, 0);

    // local right-hand side transformation matrix

    *lZ_h = *lQ_h;
    if (matrix_b != NULL) {
        starpu_matrix_data_register(lZ_h, -1, 0,
            window_size, window_size, window_size, sizeof(double));
    #ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL)
            starpu_mpi_data_register_comm(
                *lZ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
    #endif
        starneig_pack_handle(STARPU_W, *lZ_h, helper, 0);
    }

    // deflation check vector

    struct range_packing_info packing_info_aftermath;
    starneig_pack_range(
        STARPU_RW, begin, end, aftermath, helper, &packing_info_aftermath, 0);

    // corresponding tiles from the matrix A

    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_a, helper,
        &packing_info_A, PACKING_MODE_UPPER_HESSENBERG);

    // corresponding tiles from the matrix B

    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_b, helper,
        &packing_info_B, PACKING_MODE_UPPER_TRIANGULAR);

    //
    // insert task
    //

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &push_bulges_cl,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_a, sizeof(thres_a),
            STARPU_VALUE, &thres_b, sizeof(thres_b),
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_shifts_real,
                sizeof(packing_info_shifts_real),
            STARPU_VALUE, &packing_info_shifts_imag,
                sizeof(packing_info_shifts_imag),
            STARPU_VALUE, &packing_info_aftermath,
                sizeof(packing_info_aftermath),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &mode, sizeof(mode),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &push_bulges_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_a, sizeof(thres_a),
            STARPU_VALUE, &thres_b, sizeof(thres_b),
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_shifts_real,
                sizeof(packing_info_shifts_real),
            STARPU_VALUE, &packing_info_shifts_imag,
                sizeof(packing_info_shifts_imag),
            STARPU_VALUE, &packing_info_aftermath,
                sizeof(packing_info_aftermath),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &mode, sizeof(mode),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL) {
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), *lQ_h);
        starpu_mpi_data_set_rank_comm(
            *lQ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        if (*lZ_h != *lQ_h) {
            starpu_mpi_get_data_on_all_nodes_detached(
                starneig_mpi_get_comm(), *lZ_h);
            starpu_mpi_data_set_rank_comm(
                *lZ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        }
    }
#endif

    starneig_free_packing_helper(helper);
}

double starneig_predict_aggressively_deflate(int generalized, int window_size)
{
    // find a CPU worker
    int cpu_worker = starpu_worker_get_by_type(STARPU_CPU_WORKER, 0);

    // build a "fake" task

    double thres_a, thres_b, thres_inf;
    struct range_packing_info packing_info_shifts_real;
    struct range_packing_info packing_info_shifts_imag;
    struct packing_info packing_info_A = {
        .rbegin = 0,
        .rend = window_size,
        .cbegin = 0,
        .cend = window_size,
    };
    struct packing_info packing_info_B;

    struct starpu_codelet *codelet =
        generalized ?
            &aggressively_deflate_gep_cl : &aggressively_deflate_sep_cl;

    struct starpu_task *task = starpu_task_build(
        codelet,
        STARPU_VALUE, &thres_a, sizeof(thres_a),
        STARPU_VALUE, &thres_b, sizeof(thres_b),
        STARPU_VALUE, &thres_inf, sizeof(thres_inf),
        STARPU_VALUE, &packing_info_shifts_real,
            sizeof(packing_info_shifts_real),
        STARPU_VALUE, &packing_info_shifts_imag,
            sizeof(packing_info_shifts_imag),
        STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
        STARPU_VALUE, &packing_info_B, sizeof(packing_info_B), 0);

    // predict execution time on the cpu worker

    struct starpu_perfmodel_arch *arch =
        starpu_worker_get_perf_archtype(cpu_worker, 0);

    double expected_length = starpu_task_expected_length(task, arch, 0);

    task->destroy = 0;
    starpu_task_destroy(task);

    return expected_length;
}

void starneig_schur_insert_aggressively_deflate(
    int begin, int end, int prio,
    double thres_a, double thres_b, double thres_inf,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starneig_vector_descr_t shifts_real, starneig_vector_descr_t shifts_imag,
    starpu_data_handle_t *status_h, starpu_data_handle_t *lQ_h,
    starpu_data_handle_t *lZ_h, mpi_info_t mpi)
{
    *status_h = NULL;
    *lQ_h = *lZ_h = NULL;

    int window_size = end - begin;
    if (window_size < 1)
        return;

    // figure out who is going to execute the task

#ifdef STARNEIG_ENABLE_MPI
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(begin, begin, matrix_a);
#endif

    struct packing_helper *helper = starneig_init_packing_helper();

    // return status

    starpu_variable_data_register(status_h, -1, 0, sizeof(struct aed_status));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *status_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *status_h, helper, 0);

    // local left-hand side transformation matrix

    starpu_matrix_data_register(lQ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *lQ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *lQ_h, helper, 0);

    // local right-hand side transformation matrix

    *lZ_h = *lQ_h;
    if (matrix_b != NULL) {
        starpu_matrix_data_register(lZ_h, -1, 0,
            window_size, window_size, window_size, sizeof(double));
    #ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL)
            starpu_mpi_data_register_comm(
                *lZ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
    #endif
        starneig_pack_handle(STARPU_W, *lZ_h, helper, 0);
    }

    // shifts (real parts)

    struct range_packing_info packing_info_shifts_real;
    starneig_pack_range(STARPU_W, 0, window_size-1, shifts_real,
        helper, &packing_info_shifts_real, 0);

    // shifts (imaginary parts)

    struct range_packing_info packing_info_shifts_imag;
    starneig_pack_range(STARPU_W, 0, window_size-1, shifts_imag,
        helper, &packing_info_shifts_imag, 0);

    // corresponding tiles from the matrix A

    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_a, helper,
        &packing_info_A, PACKING_MODE_UPPER_HESSENBERG);

    // corresponding tiles from the matrix B

    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_b, helper,
        &packing_info_B, PACKING_MODE_UPPER_HESSENBERG);

    //
    // insert task
    //

    struct starpu_codelet *codelet =
        matrix_b != NULL ?
            &aggressively_deflate_gep_cl : &aggressively_deflate_sep_cl;

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            codelet,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_a, sizeof(thres_a),
            STARPU_VALUE, &thres_b, sizeof(thres_b),
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_shifts_real,
                sizeof(packing_info_shifts_real),
            STARPU_VALUE, &packing_info_shifts_imag,
                sizeof(packing_info_shifts_imag),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            codelet,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_a, sizeof(thres_a),
            STARPU_VALUE, &thres_b, sizeof(thres_b),
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_shifts_real,
                sizeof(packing_info_shifts_real),
            STARPU_VALUE, &packing_info_shifts_imag,
                sizeof(packing_info_shifts_imag),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL) {
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), *lQ_h);
        starpu_mpi_data_set_rank_comm(
            *lQ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        if (*lZ_h != *lQ_h) {
            starpu_mpi_get_data_on_all_nodes_detached(
                starneig_mpi_get_comm(), *lZ_h);
            starpu_mpi_data_set_rank_comm(
                *lZ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        }
    }
#endif

    starneig_free_packing_helper(helper);
}

void starneig_schur_insert_small_schur(
    int begin, int end, int prio,
    double thres_a, double thres_b, double thres_inf,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *status_h, starpu_data_handle_t *lQ_h,
    starpu_data_handle_t *lZ_h, mpi_info_t mpi)
{
    *status_h = NULL;
    *lQ_h = *lZ_h = NULL;

    int window_size = end - begin;
    if (window_size < 1)
        return;

    // figure out who is going to execute the task

#ifdef STARNEIG_ENABLE_MPI
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(begin, begin, matrix_a);
#endif

    struct packing_helper *helper = starneig_init_packing_helper();

    // return status

    starpu_variable_data_register(
        status_h, -1, 0, sizeof(struct small_schur_status));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *status_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *status_h, helper, 0);

    // local left-hand side transformation matrix

    starpu_matrix_data_register(lQ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *lQ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *lQ_h, helper, 0);

    // local right-hand side transformation matrix

    *lZ_h = *lQ_h;
    if (matrix_b != NULL) {
        starpu_matrix_data_register(lZ_h, -1, 0,
            window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL)
            starpu_mpi_data_register_comm(
                *lZ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
        starneig_pack_handle(STARPU_W, *lZ_h, helper, 0);
    }

    // corresponding tiles from the matrix A

    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_a, helper,
        &packing_info_A, PACKING_MODE_UPPER_HESSENBERG);

    // corresponding tiles from the matrix B

    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_b, helper,
        &packing_info_B, PACKING_MODE_UPPER_TRIANGULAR);

    //
    // insert task
    //

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &small_schur_cl,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_a, sizeof(thres_a),
            STARPU_VALUE, &thres_b, sizeof(thres_b),
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &small_schur_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &thres_a, sizeof(thres_a),
            STARPU_VALUE, &thres_b, sizeof(thres_b),
            STARPU_VALUE, &thres_inf, sizeof(thres_inf),
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL) {
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), *lQ_h);
        starpu_mpi_data_set_rank_comm(
            *lQ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        if (*lZ_h != *lQ_h) {
            starpu_mpi_get_data_on_all_nodes_detached(
                starneig_mpi_get_comm(), *lZ_h);
            starpu_mpi_data_set_rank_comm(
                *lZ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        }
    }
#endif

    starneig_free_packing_helper(helper);
}

void starneig_schur_insert_small_hessenberg(
    int begin, int end, int prio, starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b, starpu_data_handle_t *lQ_h,
    starpu_data_handle_t *lZ_h, mpi_info_t mpi)
{
    *lQ_h = *lZ_h = NULL;

    int window_size = end - begin;
    if (window_size < 1)
        return;

    // figure out who is going to execute the task

#ifdef STARNEIG_ENABLE_MPI
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(begin, begin, matrix_a);
#endif

    struct packing_helper *helper = starneig_init_packing_helper();

    // local left-hand side transformation matrix

    starpu_matrix_data_register(lQ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *lQ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *lQ_h, helper, 0);

    // local right-hand side transformation matrix

    *lZ_h = *lQ_h;
    if (matrix_b != NULL) {
        starpu_matrix_data_register(lZ_h, -1, 0,
            window_size, window_size, window_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL)
            starpu_mpi_data_register_comm(
                *lZ_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
        starneig_pack_handle(STARPU_W, *lZ_h, helper, 0);
    }

    // corresponding tiles from the matrix A

    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_a, helper,
        &packing_info_A, PACKING_MODE_DEFAULT);

    // corresponding tiles from the matrix B

    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_b, helper,
        &packing_info_B, PACKING_MODE_UPPER_TRIANGULAR);

    //
    // insert task
    //

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &small_hessenberg_cl,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &small_hessenberg_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL) {
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), *lQ_h);
        starpu_mpi_data_set_rank_comm(
            *lQ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        if (*lZ_h != *lQ_h) {
            starpu_mpi_get_data_on_all_nodes_detached(
                starneig_mpi_get_comm(), *lZ_h);
            starpu_mpi_data_set_rank_comm(
                *lZ_h, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
        }
    }
#endif

    starneig_free_packing_helper(helper);
}

void starneig_schur_insert_form_spike(
    int prio, starneig_matrix_descr_t matrix_q, starneig_vector_descr_t *spike)
{
    *spike = starneig_init_matching_vector_descr(
        matrix_q, sizeof(double), NULL, NULL);

    if (STARNEIG_MATRIX_N(matrix_q) < 2)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    // first row from transformation matrix that resulted from the preceding QR
    // step
    struct packing_info packing_info;
    starneig_pack_window(STARPU_R,
        1, 2, 1, STARNEIG_MATRIX_N(matrix_q), matrix_q, helper, &packing_info,
        0);

    // spike base
    struct range_packing_info packing_info_spike;
    starneig_pack_range(STARPU_W,
        1, STARNEIG_VECTOR_M(*spike), *spike, helper, &packing_info_spike, 0);

    // insert task
    starpu_task_insert(
        &form_spike_cl,
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &packing_info, sizeof(packing_info),
        STARPU_VALUE, &packing_info_spike, sizeof(packing_info_spike),
        STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

   starneig_free_packing_helper(helper);
}

void starneig_schur_insert_embed_spike(
    int end, int prio, starneig_vector_descr_t spike,
    starneig_matrix_descr_t matrix_a)
{
    if (STARNEIG_MATRIX_M(matrix_a) < 2)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    // spike base
    struct range_packing_info packing_info_spike;
    starneig_pack_range(
        STARPU_R, 1, end, spike, helper, &packing_info_spike, 0);

    // column to the left of the AED window
    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, 1, STARNEIG_MATRIX_M(matrix_a), 0, 1,
        matrix_a, helper, &packing_info, 0);

    // insert task
    starpu_task_insert(
        &embed_spike_cl,
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &packing_info_spike, sizeof(packing_info_spike),
        STARPU_VALUE, &packing_info, sizeof(packing_info),
        STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

   starneig_free_packing_helper(helper);
}

void starneig_schur_insert_deflate(
    int begin, int end, int deflate, int prio,
    double thres_a, starpu_data_handle_t inducer_h,
    starpu_data_handle_t status_h, starneig_vector_descr_t base,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *lQ_h, starpu_data_handle_t *lZ_h)
{
    *lQ_h = *lZ_h = NULL;

    int window_size = end - begin;
    if (window_size < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    // spike inducer

    starneig_pack_handle(STARPU_R, inducer_h, helper, 0);

    // return status

    starneig_pack_handle(STARPU_RW, status_h, helper, 0);

    // local left-hand side transformation matrix

    starpu_matrix_data_register(lQ_h, -1, 0,
        window_size, window_size, window_size, sizeof(double));
    starneig_pack_handle(STARPU_W, *lQ_h, helper, 0);

    // local right-hand side transformation matrix

    *lZ_h = *lQ_h;
    if (matrix_b != NULL) {
        starpu_matrix_data_register(lZ_h, -1, 0,
            window_size, window_size, window_size, sizeof(double));
        starneig_pack_handle(STARPU_W, *lZ_h, helper, 0);
    }

    // spike base

    struct range_packing_info packing_info_spike;
    starneig_pack_range(
        STARPU_RW, begin, end, base, helper, &packing_info_spike, 0);

    // corresponding tiles from the matrix A

    struct packing_info packing_info_A;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_a, helper,
        &packing_info_A, PACKING_MODE_UPPER_HESSENBERG);

    // corresponding tiles from the matrix B

    struct packing_info packing_info_B;
    starneig_pack_diag_window(STARPU_RW, begin, end, matrix_b, helper,
        &packing_info_B, PACKING_MODE_UPPER_TRIANGULAR);

    //
    // insert task
    //

    int corner = end == STARNEIG_MATRIX_M(matrix_a);

    starpu_task_insert(
        &deflate_cl,
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &thres_a, sizeof(thres_a),
        STARPU_VALUE, &packing_info_spike, sizeof(packing_info_spike),
        STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
        STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
        STARPU_VALUE, &begin, sizeof(begin),
        STARPU_VALUE, &deflate, sizeof(deflate),
        STARPU_VALUE, &corner, sizeof(corner),
        STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_schur_insert_extract_shifts(
    int begin, int end, int prio, starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b, starneig_vector_descr_t real,
    starneig_vector_descr_t imag, mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    struct packing_info packing_info_A;
    starneig_pack_diag_window(
        STARPU_R, begin, end, matrix_a, helper, &packing_info_A,
        PACKING_MODE_UPPER_HESSENBERG);

    struct packing_info packing_info_B;
    starneig_pack_diag_window(
        STARPU_R, begin, end, matrix_b, helper, &packing_info_B,
        PACKING_MODE_UPPER_TRIANGULAR);

    struct range_packing_info packing_info_real;
    starneig_pack_range(STARPU_RW,
        0, end-begin, real, helper, &packing_info_real, 0);

    struct range_packing_info packing_info_imag;
    starneig_pack_range(STARPU_RW,
        0, end-begin, imag, helper, &packing_info_imag, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &extract_shifts_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_get_elem_owner_vector_descr(begin, real),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &packing_info_real, sizeof(packing_info_real),
            STARPU_VALUE, &packing_info_imag, sizeof(packing_info_imag),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &extract_shifts_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
            STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
            STARPU_VALUE, &packing_info_real, sizeof(packing_info_real),
            STARPU_VALUE, &packing_info_imag, sizeof(packing_info_imag),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

starpu_data_handle_t starneig_schur_insert_compute_norm(
    int prio, starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
#ifdef STARNEIG_ENABLE_MPI
    int my_rank = starneig_mpi_get_comm_rank();
#endif

    int bm = STARNEIG_MATRIX_BM(matrix);
    int bn = STARNEIG_MATRIX_BN(matrix);

    int rbbegin = STARNEIG_MATRIX_RBEGIN(matrix) / bm;
    int rbend = (STARNEIG_MATRIX_REND(matrix)-1) / bm + 1;
    int cbbegin = STARNEIG_MATRIX_CBEGIN(matrix) / bn;
    int cbend = (STARNEIG_MATRIX_CEND(matrix)-1) / bn + 1;

    int tiles = 0;
    struct starpu_data_descr *descrs = malloc(
        ((rbend-rbbegin)*(cbend-cbbegin)+1)*sizeof(struct starpu_data_descr));

    //
    // process tiles
    //

    for (int j = cbbegin; j < cbend; j++) {
        for (int i = rbbegin; i < rbend; i++) {

            starpu_data_handle_t tile =
                starneig_get_tile_from_matrix_descr(i, j, matrix);

            starpu_data_handle_t handle;
            starpu_variable_data_register(&handle, -1, 0, sizeof(double));

            // pack the handle and the access mode for the compute_norm_b task
            descrs[tiles].handle = handle;
            descrs[tiles].mode = STARPU_R;
            tiles++;

            int rbegin = MAX(0, STARNEIG_MATRIX_RBEGIN(matrix) - i*bm);
            int rend = MIN(bm, STARNEIG_MATRIX_REND(matrix) - i*bm);

            int cbegin = MAX(0, STARNEIG_MATRIX_CBEGIN(matrix) - j*bn);
            int cend = MIN(bn, STARNEIG_MATRIX_CEND(matrix) - j*bn);

#ifdef STARNEIG_ENABLE_MPI
            if (mpi != NULL) {
                int owner = starpu_mpi_data_get_rank(tile);
                starpu_mpi_data_register_comm(
                    handle, mpi->tag_offset++, owner, starneig_mpi_get_comm());

                if (my_rank == owner)
                    starpu_mpi_task_insert(
                        starneig_mpi_get_comm(),
                        &compute_norm_a_cl,
                        STARPU_EXECUTE_ON_NODE, owner,
                        STARPU_PRIORITY, prio,
                        STARPU_VALUE, &rbegin, sizeof(rbegin),
                        STARPU_VALUE, &rend, sizeof(rend),
                        STARPU_VALUE, &cbegin, sizeof(cbegin),
                        STARPU_VALUE, &cend, sizeof(cend),
                        STARPU_R, tile, STARPU_W, handle, 0);

                // gather result to all nodes
                starpu_mpi_get_data_on_all_nodes_detached(
                    starneig_mpi_get_comm(), handle);
                starpu_mpi_data_set_rank_comm(
                    handle, STARPU_MPI_PER_NODE, starneig_mpi_get_comm());
            }
            else
#endif
            {
                starpu_task_insert(
                    &compute_norm_a_cl,
                    STARPU_PRIORITY, prio,
                    STARPU_VALUE, &rbegin, sizeof(rbegin),
                    STARPU_VALUE, &rend, sizeof(rend),
                    STARPU_VALUE, &cbegin, sizeof(cbegin),
                    STARPU_VALUE, &cend, sizeof(cend),
                    STARPU_R, tile, STARPU_W, handle, 0);
            }
        }
    }

    //
    // compute final result
    //

    starpu_data_handle_t norm;
    starpu_variable_data_register(&norm, -1, 0, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            norm, mpi->tag_offset++, STARPU_MPI_PER_NODE,
            starneig_mpi_get_comm());
#endif

    descrs[tiles].handle = norm;
    descrs[tiles].mode = STARPU_W;

    starpu_task_insert(
        &compute_norm_b_cl,
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &tiles, sizeof(tiles),
        STARPU_DATA_MODE_ARRAY, descrs, tiles+1, 0);

    //
    // cleanup
    //

    for (int i = 0; i < tiles; i++)
        starpu_data_unregister_submit(descrs[i].handle);

    free(descrs);

    return norm;
}
