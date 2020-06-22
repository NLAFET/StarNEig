///
/// @file This file contains the Hessenberg reduction specific task definitions
/// and task insertion functions.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
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
#include <starneig/node.h>
#include <starpu_mpi.h>
#endif

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for process_panel codelet.
///
static void process_panel_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    parameters[0] = packing_info.rend - packing_info.rbegin;
    parameters[1] = MAX(0, packing_info.cend - packing_info.cbegin - nb/2);
    parameters[3] = nb;
}

///
/// @brief Multiple regression performance model for process_panel codelet.
///
static struct starpu_perfmodel process_panel_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_process_panel_pm",
    .parameters = &process_panel_parameters,
    .nparameters = 3,
    .parameters_names = (const char*[]) { "M", "N", "NB" },
    .combinations = (unsigned*[]) { (unsigned[]) { 1, 1, 1 } },
    .ncombinations = 1
};

#else

///
/// @brief Size base function for process_panel codelet.
///
static size_t process_panel_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    return (size_t) nb * (packing_info.rend - packing_info.rbegin) *
        (packing_info.cend - packing_info.cbegin - nb/2);
}

///
/// @brief Linear regression performance model for process_panel codelet.
///
static struct starpu_perfmodel process_panel_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_process_panel_pm",
    .size_base = &process_panel_size_base
};

#endif

///
/// @brief process_panel codelet reduces a panel to Hessenberg form.
///
///  NOTE: Blocks that from the panel should have STARPU_RW access. Other tiles
///  (the ones that form the trailing matrix) can have STARPU_R access.
///
///  Arguments:
///   - packing_info  tile packing information
///   - nb            panel width
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_W, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_W, nb rows/columns)
///   - Y matrix (STARPU_W, rend-rbegin rows, nb columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, nb columns)
///   - matrix A tiles that correspond to the panel and the trailing matrix
///         (STARPU_RW, in column-major order)
///
static struct starpu_codelet process_panel_cl = {
    .name = "starneig_process_panel",
    .cpu_funcs = { starneig_hessenberg_ext_cpu_process_panel_bind },
    .cpu_funcs_name = { "starneig_hessenberg_ext_cpu_process_panel_bind" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_ext_cuda_process_panel },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .type = STARPU_FORKJOIN,
    .max_parallelism = INT_MAX,
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &process_panel_pm
};

///
/// @brief process_panel_single codelet reduces a panel to Hessenberg form.
///
///  NOTE: Blocks that from the panel should have STARPU_RW access. Other tiles
///  (the ones that form the trailing matrix) can have STARPU_R access.
///
///  Arguments:
///   - packing_info  tile packing information
///   - nb            panel width
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_W, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_W, nb rows/columns)
///   - Y matrix (STARPU_W, rend-rbegin rows, nb columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, nb columns)
///   - matrix A tiles that correspond to the panel and the trailing matrix
///         (STARPU_RW, in column-major order)
///
static struct starpu_codelet process_panel_single_cl = {
    .name = "starneig_process_panel",
    .cpu_funcs = { starneig_hessenberg_ext_cpu_process_panel_single },
    .cpu_funcs_name = { "starneig_hessenberg_ext_cpu_process_panel_single" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_ext_cuda_process_panel },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &process_panel_pm
};

////////////////////////////////////////////////////////////////////////////////

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for update_trail codelet.
///
static void update_trail_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    parameters[0] = packing_info.rend - packing_info.rbegin;
    parameters[1] = packing_info.cend - packing_info.cbegin;
    parameters[2] = nb;
}

///
/// @brief Multiple regression performance model for update_trail codelet.
///
static struct starpu_perfmodel update_trail_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_update_trail_pm",
    .parameters = &update_trail_parameters,
    .nparameters = 3,
    .parameters_names = (const char*[]) { "M", "N", "NB" },
    .combinations = (unsigned*[]) { (unsigned[]) { 1, 1, 1 } },
    .ncombinations = 1
};

#else

///
/// @brief Size base function for update_trail codelet.
///
static size_t update_trail_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    return (size_t) nb * (packing_info.rend - packing_info.rbegin) *
        (packing_info.cend - packing_info.cbegin);
}

///
/// @brief Linear regression performance model for update_trail codelet.
///
static struct starpu_perfmodel update_trail_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_update_trail_pm",
    .size_base = &update_trail_size_base
};

#endif

///
/// @brief Updates the trailing matrix from left and right.
///
///  Arguments:
///   - packing_info  tile packing information
///   - nb            panel width
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - Y matrix (STARPU_R, rend-rbegin rows, nb columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, <X> columns)
///   - scratch matrix (STARPU_SCRATCH, <X> rows, nb columns)
///   - matrix A tiles that correspond to the trailing matrix (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_trail_cl = {
    .name = "starneig_update_trail",
    .cpu_funcs = { starneig_hessenberg_ext_cpu_update_trail_bind },
    .cpu_funcs_name = { "starneig_hessenberg_ext_cpu_update_trail_bind" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_ext_cuda_update_trail },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .type = STARPU_FORKJOIN,
    .max_parallelism = INT_MAX,
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &update_trail_pm
};

///
/// @brief Updates the trailing matrix from left and right.
///
///  Arguments:
///   - packing_info  tile packing information
///   - nb            panel width
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - Y matrix (STARPU_R, rend-rbegin rows, nb columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, <X> columns)
///   - scratch matrix (STARPU_SCRATCH, <X> rows, nb columns)
///   - matrix A tiles that correspond to the trailing matrix (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_trail_single_cl = {
    .name = "starneig_update_trail",
    .cpu_funcs = { starneig_hessenberg_ext_cpu_update_trail_single },
    .cpu_funcs_name = { "starneig_hessenberg_ext_cpu_update_trail_single" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_ext_cuda_update_trail },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &update_trail_pm
};

////////////////////////////////////////////////////////////////////////////////

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for update_right codelet.
///
static void update_right_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    parameters[0] = packing_info.rend - packing_info.rbegin;
    parameters[1] = packing_info.cend - packing_info.cbegin;
    parameters[2] = nb;
}

///
/// @brief Multiple regression performance model for update_right codelet.
///
static struct starpu_perfmodel update_right_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_update_right_pm",
    .parameters = &update_right_parameters,
    .nparameters = 3,
    .parameters_names = (const char*[]) { "M", "N", "NB" },
    .combinations = (unsigned*[]) { (unsigned[]) { 1, 1, 1 } },
    .ncombinations = 1
};

#else

///
/// @brief Size base function for update_right codelet.
///
static size_t update_right_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    return (size_t) (packing_info.rend - packing_info.rbegin) *
        (packing_info.cend - packing_info.cbegin - nb) * nb;
}

///
/// @brief Linear regression performance model for update_right codelet.
///
static struct starpu_perfmodel update_right_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_update_right_pm",
    .size_base = &update_right_size_base
};

#endif

///
/// @brief Updates a section of a matrix from the right.
///
///  Arguments:
///   - packing_info  tile packing information
///   - nb            panel width
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbein columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, nb columns)
///   - matrix A tiles that correspond to the trailing matrix (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_right_cl = {
    .name = "starneig_update_right",
    .cpu_funcs = { starneig_hessenberg_ext_cpu_update_right },
    .cpu_funcs_name = { "starneig_hessenberg_ext_cpu_update_right" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_ext_cuda_update_right },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &update_right_pm
};

////////////////////////////////////////////////////////////////////////////////

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for update_left codelet.
///
static void update_left_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    parameters[0] = packing_info.rend - packing_info.rbegin;
    parameters[1] = packing_info.cend - packing_info.cbegin;
    parameters[2] = nb;
}

///
/// @brief Multiple regression performance model for update_left codelet.
///
static struct starpu_perfmodel update_left_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_update_left_pm",
    .parameters = &update_left_parameters,
    .nparameters = 3,
    .parameters_names = (const char*[]) { "M", "N", "NB" },
    .combinations = (unsigned*[]) { (unsigned[]) { 1, 1, 1 } },
    .ncombinations = 1
};

#else

///
/// @brief Size base function for update_left codelet.
///
static size_t update_left_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info, &nb);

    return (size_t) nb * (packing_info.rend - packing_info.rbegin) *
        (packing_info.cend - packing_info.cbegin);
}

///
/// @brief Linear regression performance model for update_left codelet.
///
static struct starpu_perfmodel update_left_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_update_left_pm",
    .size_base = &update_left_size_base
};

#endif

///
/// @brief Updates a section of a matrix from the left.
///
///  Arguments:
///   - packing_info  tile packing information
///   - nb            panel width
///
///  Buffers:
///   - V matrix, i.e., reflectors (STARPU_R, rend-rbegin rows, nb columns)
///   - T matrix (STARPU_R, nb rows/columns)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH, cend-cbegin rows, nb columns)
///   - matrix A tiles that correspond to the trailing matrix (STARPU_RW,
///         in column-major order)
///
static struct starpu_codelet update_left_cl = {
    .name = "starneig_update_left",
    .cpu_funcs = { starneig_hessenberg_ext_cpu_update_left },
    .cpu_funcs_name = { "starneig_hessenberg_ext_cpu_update_left" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_hessenberg_ext_cuda_update_left },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &update_left_pm
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_hessenberg_ext_insert_process_panel(
    unsigned ctx, int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starneig_matrix_descr_t matrix_a, starpu_data_handle_t *V_h,
    starpu_data_handle_t *T_h, starpu_data_handle_t *Y_h, int parallel,
    mpi_info_t mpi)
{
    *V_h = *T_h = *Y_h = NULL;

    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

#ifdef STARNEIG_ENABLE_MPI
    int owner = 0;
    if (mpi != NULL)
        owner = starneig_get_elem_owner_matrix_descr(rbegin, cbegin, matrix_a);
#endif

    int m = rend-rbegin;

    struct packing_helper *helper = starneig_init_packing_helper();

    starpu_matrix_data_register(V_h, -1, 0, m, m, nb, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *V_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *V_h, helper, 0);

    starpu_matrix_data_register(T_h, -1, 0, nb, nb, nb, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *T_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *T_h, helper, 0);

    starpu_matrix_data_register(Y_h, -1, 0, m, m, nb, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            *Y_h, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif
    starneig_pack_handle(STARPU_W, *Y_h, helper, 0);

    starneig_pack_cached_scratch_matrix(m, nb, sizeof(double), helper);

    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, rbegin, rend, cbegin, cend,
        matrix_a, helper, &packing_info, 0);

    struct starpu_codelet *codelet = &process_panel_cl;
    if (!parallel)
        codelet = &process_panel_single_cl;

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            codelet,
            STARPU_EXECUTE_ON_NODE, owner,
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            codelet,
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_ext_insert_update_trail(
    unsigned ctx, int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starpu_data_handle_t Y_h, starneig_matrix_descr_t matrix_a, int parallel,
    mpi_info_t mpi)
{
    if (nb < 1 || rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, V_h, helper, 0);
    starneig_pack_handle(STARPU_R, T_h, helper, 0);
    starneig_pack_handle(STARPU_R, Y_h, helper, 0);

    int width = cend-cbegin < 512 ? cend-cbegin : (cend-cbegin)/4;
    starneig_pack_cached_scratch_matrix(
        rend-rbegin, width, sizeof(double), helper);
    starneig_pack_cached_scratch_matrix(width, nb, sizeof(double), helper);

    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, rbegin, rend, cbegin, cend,
        matrix_a, helper, &packing_info, 0);

    struct starpu_codelet *codelet = &update_trail_cl;
    if (!parallel)
        codelet = &update_trail_single_cl;

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            codelet,
            STARPU_EXECUTE_ON_NODE,
            starneig_get_elem_owner_matrix_descr(rbegin, cbegin, matrix_a),
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            codelet,
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_ext_insert_update_right(
    unsigned ctx, int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_descr_t matrix_a, mpi_info_t mpi)
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

    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, rbegin, rend, cbegin, cend,
        matrix_a, helper, &packing_info, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_right_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_get_elem_owner_matrix_descr(rbegin, cbegin, matrix_a),
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_right_cl,
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_hessenberg_ext_insert_update_left(
    unsigned ctx, int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_descr_t matrix_a, mpi_info_t mpi)
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

    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, rbegin, rend, cbegin, cend,
        matrix_a, helper, &packing_info, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &update_left_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_get_elem_owner_matrix_descr(rbegin, cbegin, matrix_a),
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &update_left_cl,
            STARPU_SCHED_CTX, ctx,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

   starneig_free_packing_helper(helper);
}
