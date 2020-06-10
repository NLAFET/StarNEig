///
/// @file
///
/// @brief This file contains task definitions and task insertion function that
/// are shared among all components of the library.
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
#include "common.h"
#include "tiles.h"
#include "scratch.h"
#include "math.h"
#include "cpu.h"
#ifdef STARNEIG_ENABLE_CUDA
#include "cuda.h"
#endif
#include <limits.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for left_gemm_update codelet.
///
static void left_gemm_update_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info);

    parameters[0] = packing_info.rend - packing_info.rbegin;
    parameters[1] = packing_info.cend - packing_info.cbegin;
}

///
/// @brief Multiple regression performance model for left_gemm_update codelet.
///
static struct starpu_perfmodel left_gemm_update_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_left_gemm_update_pm",
    .parameters = &left_gemm_update_parameters,
    .nparameters = 2,
    .parameters_names = (const char*[]) { "M", "N" },
    .combinations = (unsigned*[]) {
        (unsigned[]) { 2U, 1U },
        (unsigned[]) { 1U, 1U }
    },
    .ncombinations = 2
};

#else

///
/// @brief Size base function for left_gemm_update codelet.
///
static size_t left_gemm_update_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info);

    return (size_t)
        (packing_info.rend - packing_info.rbegin) *
        (packing_info.rend - packing_info.rbegin) *
        (packing_info.cend - packing_info.cbegin);
}

///
/// @brief Linear regression performance model for left_gemm_update codelet.
///
static struct starpu_perfmodel left_gemm_update_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_left_gemm_update_pm",
    .size_base = &left_gemm_update_size_base
};

#endif

///
/// @brief left_gemm_update codelet performs a left-hand side update using a
/// given local Q matrix.
///
///  Arguments:
///   - matrix tile packing information
///
///  Buffers:
///   - local Q matrix (STARPU_R; cend-cbegin rows/columns)
///   - scratch matrix (STARPU_SCRATCH; rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH; rend-rbegin rows, cend-cbegin columns)
///   - matrix tiles that correspond to the update window (STARPU_RW)
///
static struct starpu_codelet left_gemm_update_cl = {
    .name = "starneig_left_gemm_update",
    .cpu_funcs = { starneig_cpu_left_gemm_update },
    .cpu_funcs_name = { "starneig_cpu_left_gemm_update" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_cuda_left_gemm_update },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &left_gemm_update_pm
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if defined STARNEIG_ENABLE_MRM && \
(1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION)

///
/// @brief Parameters function for right_gemm_update codelet.
///
static void right_gemm_update_parameters(
    struct starpu_task *task, double *parameters)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info);

    parameters[0] = packing_info.rend - packing_info.rbegin;
    parameters[1] = packing_info.cend - packing_info.cbegin;
}

///
/// @brief Multiple regression performance model for right_gemm_update codelet.
///
static struct starpu_perfmodel right_gemm_update_pm = {
    .type = STARPU_MULTIPLE_REGRESSION_BASED,
    .symbol = "starneig_right_gemm_update_pm",
    .parameters = &right_gemm_update_parameters,
    .nparameters = 2,
    .parameters_names = (const char*[]) { "M", "N" },
    .combinations = (unsigned*[]) {
        (unsigned[]) { 1U, 2U },
        (unsigned[]) { 1U, 1U }
    },
    .ncombinations = 2
};

#else

///
/// @brief Size base function for right_gemm_update codelet.
///
static size_t right_gemm_update_size_base(
    struct starpu_task *task, unsigned nimpl)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(task->cl_arg, &packing_info);

    return (size_t)
        (packing_info.rend - packing_info.rbegin) *
        (packing_info.cend - packing_info.cbegin) *
        (packing_info.cend - packing_info.cbegin);
}

///
/// @brief Linear regression performance model for right_gemm_update codelet.
///
static struct starpu_perfmodel right_gemm_update_pm = {
    .type = STARPU_NL_REGRESSION_BASED,
    .symbol = "starneig_right_gemm_update_pm",
    .size_base = &right_gemm_update_size_base
};

#endif

///
/// @brief right_gemm_update codelet performs a right-hand side update using a
/// given local Q matrix.
///
///  Arguments:
///   - matrix tile packing information
///
///  Buffers:
///   - local Q matrix (STARPU_R; cend-cbegin rows/columns)
///   - scratch matrix (STARPU_SCRATCH; rend-rbegin rows, cend-cbegin columns)
///   - scratch matrix (STARPU_SCRATCH; rend-rbegin rows, cend-cbegin columns)
///   - matrix tiles that correspond to the update window (STARPU_RW)
///
static struct starpu_codelet right_gemm_update_cl = {
    .name = "starneig_right_gemm_update",
    .cpu_funcs = { starneig_cpu_right_gemm_update },
    .cpu_funcs_name = { "starneig_cpu_right_gemm_update" },
#ifdef STARNEIG_ENABLE_CUDA
    .cuda_funcs = { starneig_cuda_right_gemm_update },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = STARPU_VARIABLE_NBUFFERS,
    .model = &right_gemm_update_pm
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief copy_matrix_to_handle codelet copies a section of a matrix to a
/// data handle.
///
///  Arguments:
///   - source matrix packing information
///
///  Buffers:
///   - matrix tiles that correspond to the source matrix (STARPU_R)
///   - destination handle (STARPU_W, rend-rbegin rows, cend-cbegin columns)
///
static struct starpu_codelet copy_matrix_to_handle_cl = {
    .name = "starneig_copy_matrix_to_handle",
    .cpu_funcs = { starneig_cpu_copy_matrix_to_handle },
    .cpu_funcs_name = { "starneig_cpu_copy_matrix_to_handle" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

///
/// @brief copy_handle_to_matrix codelet copies a data handle to a section of
/// a matrix.
///
///  Arguments:
///   - destination matrix packing information
///
///  Buffers:
///   - source handle (STARPU_R, rend-rbegin rows, cend-cbegin columns)
///   - matrix tiles that correspond to the destination matrix (STARPU_RW)
///
static struct starpu_codelet copy_handle_to_matrix_cl = {
    .name = "starneig_copy_handle_to_matrix",
    .cpu_funcs = { starneig_cpu_copy_handle_to_matrix },
    .cpu_funcs_name = { "starneig_cpu_copy_handle_to_matrix" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

///
/// @brief copy_matrix codelet copies a section of a matrix.
///
///  Arguments:
///   - source matrix packing information
///   - destination matrix packing information
///
///  Buffers:
///   - matrix tiles that correspond to the source matrix (STARPU_R)
///   - scratch matrix (STARPU_SCRATCH, rend-rbegin rows, cend-cbegin columns)
///   - matrix tiles that correspond to the destination matrix (STARPU_RW)
///
static struct starpu_codelet copy_matrix_cl = {
    .name = "starneig_copy_matrix",
    .cpu_funcs = { starneig_cpu_copy_matrix },
    .cpu_funcs_name = { "starneig_cpu_copy_matrix" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

///
/// @brief set_to_identity codelet performs an identity matrix initialization.
///
///  Arguments:
///   - matrix packing information
///
///  Buffers:
///   - matrix tiles (STARPU_RW)
///
static struct starpu_codelet set_to_identity_cl = {
    .name = "starneig_set_to_identity",
    .cpu_funcs = { starneig_cpu_set_to_identity },
    .cpu_funcs_name = { "starneig_cpu_set_to_identity" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief scan_diagonal_cl codelet scans the diagonal of a matrix pencil (A,B)
/// using a provided scanning function. The scanning function is expected
/// to store the outcome of the scan to a mask vector.
///
///  Arguments:
///   - row offset for the first diagonal element
///   - column offset for the first diagonal element
///   - pointer to the scanning function:
///         - return type: void
///         - arg  0: (int) number of diagonal entries to scan
///         - arg  1: (int) row offset for the first diagonal element
///         - arg  2: (int) column offset for the first diagonal element
///         - arg  3: (int) number of rows in the scanning window
///         - arg  4: (int) number of columns in the scanning window
///         - arg  5: (int) leading dimension of the matrix A
///         - arg  6: (int) leading dimension of the matrix B
///         - arg  7: (void const *) optional argument
///         - arg  8: (void const *) scanning window from the matrix A
///         - arg  9: (void const *) scanning window from the matrix B
///         - arg 10: (void **) scanning mask vectors
///   - optional argument
///   - matrix A packing information
///   - matrix B packing information
///   - mask vector packing information
///
///  Buffers:
///   - matrix A tiles that correspond to the scanning window (STARPU_R,
///     non-zero tiles in column-major order)
///   - matrix B tiles that correspond to the scanning window (STARPU_R,
///     non-zero tiles in column-major order, optional)
///   - scanning vector tiles that correspond to the scanning window (STARPU_RW)
///
static struct starpu_codelet scan_diagonal_cl = {
    .name = "starneig_scan_diagonal",
    .cpu_funcs = { starneig_cpu_scan_diagonal },
    .cpu_funcs_name = { "starneig_cpu_scan_diagonal" },
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static struct starpu_codelet set_to_zero_cl = {
    .name = "starneig_set_to_zero",
    .cpu_funcs = { starneig_cpu_set_to_zero },
    .cpu_funcs_name = { "starneig_cpu_set_to_zero" },
    .nbuffers = 1,
    .modes = { STARPU_W }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_insert_left_gemm_update(
    int rbegin, int rend, int cbegin, int cend, int splice, int prio,
    starpu_data_handle_t lQ_h, starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    if (rend-rbegin < 1 || cend-cbegin < 1)
        return;

    rbegin = MAX(0, rbegin);
    rend = MIN(STARNEIG_MATRIX_M(matrix), rend);
    cbegin = MAX(0, cbegin);
    cend = MIN(STARNEIG_MATRIX_N(matrix), cend);

#if defined(STARNEIG_ENABLE_MPI) && defined(STARNEIG_ENABLE_PRUNING)
    int my_rank = starneig_mpi_get_comm_rank();
    int owner = 0;
    if (mpi != NULL) {
        owner = starpu_mpi_data_get_rank(lQ_h);

        // initial prune
        if (my_rank != owner && !starneig_involved_with_part_of_matrix_descr(
        rbegin, rend, cbegin, cend, matrix)) {
            starneig_flush_section_matrix_descr(
                rbegin, rend, cbegin, cend, matrix);
            return;
        }
    }
#endif

    int sn = STARNEIG_MATRIX_SN(matrix);
    int bn = STARNEIG_MATRIX_BN(matrix);

    if (splice < 1 || sn < splice)
        splice = sn;
    splice = MAX(bn, (splice/bn)*bn);

    //
    // register scratch matrices
    //

    starpu_data_handle_t scratch1_h =
        starneig_scratch_get_matrix(rend-rbegin, splice, sizeof(double));
    starpu_data_handle_t scratch2_h =
        starneig_scratch_get_matrix(rend-rbegin, splice, sizeof(double));

    //
    // loop over sections
    //

    int section_offset = ((STARNEIG_MATRIX_CBEGIN(matrix)+cbegin)/sn)*sn -
        STARNEIG_MATRIX_CBEGIN(matrix);
    for (int i = section_offset; i < cend; i += sn) {

        int sbegin = MAX(cbegin, i);
        int send = MIN(cend, i+sn);

        //
        // prune
        //

#if defined(STARNEIG_ENABLE_MPI) && defined(STARNEIG_ENABLE_PRUNING)
        if (mpi != NULL && my_rank != owner &&
        !starneig_involved_with_part_of_matrix_descr(
        rbegin, rend, sbegin, send, matrix)) {
            starneig_flush_section_matrix_descr(
                rbegin, rend, sbegin, send, matrix);
            continue;
        }
#endif

        //
        // loop over splices
        //

        int offset = ((STARNEIG_MATRIX_CBEGIN(matrix)+sbegin)/splice)*splice -
            STARNEIG_MATRIX_CBEGIN(matrix);
        for (int j = offset; j < send; j += splice) {

            int begin = MAX(sbegin, j);
            int end = MIN(send, j+splice);

            //
            // pack data handles
            //

            struct packing_helper *helper = starneig_init_packing_helper();

            // local Q matrix
            starneig_pack_handle(STARPU_R, lQ_h, helper, 0);

            // scratch matrices
            starneig_pack_handle(STARPU_SCRATCH, scratch1_h, helper, 0);
            starneig_pack_handle(STARPU_SCRATCH, scratch2_h, helper, 0);

            // corresponding matrix tiles
            struct packing_info packing_info;
            starneig_pack_window(STARPU_RW, rbegin, rend, begin, end,
                matrix, helper, &packing_info, 0);

            //
            // insert task
            //

            double flops = 2.0*(end-begin)*(rend-rbegin)*(rend-rbegin);

#ifdef STARNEIG_ENABLE_MPI
            if (mpi != NULL)
                starpu_mpi_task_insert(
                    starneig_mpi_get_comm(),
                    &left_gemm_update_cl,
                    STARPU_EXECUTE_ON_NODE,
                    starneig_get_elem_owner_matrix_descr(rbegin, begin, matrix),
                    STARPU_PRIORITY, prio,
                    STARPU_FLOPS, flops,
                    STARPU_VALUE, &packing_info, sizeof(packing_info),
                    STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
            else
#endif
                starpu_task_insert(
                    &left_gemm_update_cl,
                    STARPU_PRIORITY, prio,
                    STARPU_FLOPS, flops,
                    STARPU_VALUE, &packing_info, sizeof(packing_info),
                    STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

            starneig_free_packing_helper(helper);
        }
    }

    starneig_scratch_flush();
}

void starneig_insert_right_gemm_update(
    int rbegin, int rend, int cbegin, int cend, int splice, int prio,
    starpu_data_handle_t lQ_h, starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    if (rend-rbegin < 1 || cend-cbegin < 1)
        return;

    rbegin = MAX(0, rbegin);
    rend = MIN(STARNEIG_MATRIX_M(matrix), rend);
    cbegin = MAX(0, cbegin);
    cend = MIN(STARNEIG_MATRIX_N(matrix), cend);

    int sm =  STARNEIG_MATRIX_SM(matrix);
    int bm =  STARNEIG_MATRIX_BM(matrix);

    if (splice < 1 || sm < splice)
        splice = STARNEIG_MATRIX_SM(matrix);
    splice = MAX(bm, (splice/bm)*bm);

#if defined(STARNEIG_ENABLE_MPI) && defined(STARNEIG_ENABLE_PRUNING)
    int my_rank = starneig_mpi_get_comm_rank();
    int owner = 0;
    if (mpi != NULL) {
        owner = starpu_mpi_data_get_rank(lQ_h);

        // initial prune
        if (my_rank != owner && !starneig_involved_with_part_of_matrix_descr(
        rbegin, rend, cbegin, cend, matrix)) {
            starneig_flush_section_matrix_descr(
                rbegin, rend, cbegin, cend, matrix);
            return;
        }
    }
#endif

    //
    // register scratch matrices
    //

    starpu_data_handle_t scratch1_h =
        starneig_scratch_get_matrix(splice, cend-cbegin, sizeof(double));
    starpu_data_handle_t scratch2_h =
        starneig_scratch_get_matrix(splice, cend-cbegin, sizeof(double));

    //
    // loop over sections
    //

    int section_offset = ((STARNEIG_MATRIX_RBEGIN(matrix)+rbegin)/sm)*sm -
        STARNEIG_MATRIX_RBEGIN(matrix);
    for (int i = section_offset; i < rend; i += sm) {

        int sbegin = MAX(rbegin, i);
        int send = MIN(rend, i+sm);

        //
        // prune
        //

#if defined(STARNEIG_ENABLE_MPI) && defined(STARNEIG_ENABLE_PRUNING)
        if (mpi != NULL && my_rank != owner &&
        !starneig_involved_with_part_of_matrix_descr(
        sbegin, send, cbegin, cend, matrix)) {
            starneig_flush_section_matrix_descr(
                sbegin, send, cbegin, cend, matrix);
            continue;
        }
#endif

        //
        // loop over splices
        //

        int offset = ((STARNEIG_MATRIX_RBEGIN(matrix)+sbegin)/splice)*splice -
            STARNEIG_MATRIX_RBEGIN(matrix);
        for (int j = offset; j < send; j += splice) {

            int begin = MAX(sbegin, j);
            int end = MIN(send, j+splice);

            //
            // pack data handles
            //

            struct packing_helper *helper = starneig_init_packing_helper();

            // local Q matrix
            starneig_pack_handle(STARPU_R, lQ_h, helper, 0);

            // scratch matrices
            starneig_pack_handle(STARPU_SCRATCH, scratch1_h, helper, 0);
            starneig_pack_handle(STARPU_SCRATCH, scratch2_h, helper, 0);

            // corresponding matrix tiles
            struct packing_info packing_info;
            starneig_pack_window(STARPU_RW, begin, end, cbegin, cend,
                matrix, helper, &packing_info, 0);

            //
            // insert task
            //

            double flops = 2.0*(cend-cbegin)*(end-begin)*(cend-cbegin);

#ifdef STARNEIG_ENABLE_MPI
            if (mpi != NULL)
                starpu_mpi_task_insert(
                    starneig_mpi_get_comm(),
                    &right_gemm_update_cl,
                    STARPU_EXECUTE_ON_NODE,
                    starneig_get_elem_owner_matrix_descr(begin, cbegin, matrix),
                    STARPU_PRIORITY, prio,
                    STARPU_FLOPS, flops,
                    STARPU_VALUE, &packing_info, sizeof(packing_info),
                    STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
            else
#endif
                starpu_task_insert(
                    &right_gemm_update_cl,
                    STARPU_PRIORITY, prio,
                    STARPU_FLOPS, flops,
                    STARPU_VALUE, &packing_info, sizeof(packing_info),
                    STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

            starneig_free_packing_helper(helper);
        }
    }

    starneig_scratch_flush();
}

void insert_copy_matrix_reverse(
    int sr, int sc, int dr, int dc, int m, int n, int prio,
    starneig_matrix_descr_t source, starneig_matrix_descr_t dest,
    mpi_info_t mpi)
{
    int my_rank = starneig_mpi_get_comm_rank();

    //
    // loop over source matrix section rows
    //

    int roffset = ((STARNEIG_MATRIX_RBEGIN(source) + sr) /
        STARNEIG_MATRIX_SM(source)) * STARNEIG_MATRIX_SM(source) -
        STARNEIG_MATRIX_RBEGIN(source);
    for (int i = roffset; i < sr+m; i += STARNEIG_MATRIX_SM(source)) {

        // source row cordinates
        int _rbegin = MAX(sr, i);
        int _rend = MIN(sr+m, i + STARNEIG_MATRIX_SM(source));

        // destination row cordinates
        int __rbegin = _rbegin + dr - sr;
        int __rend = _rend + dr - sr;

        //
        // loop over source matrix section columns
        //

        int coffset = ((STARNEIG_MATRIX_CBEGIN(source) + sc) /
            STARNEIG_MATRIX_SN(source)) * STARNEIG_MATRIX_SN(source) -
            STARNEIG_MATRIX_CBEGIN(source);
        for (int j = coffset; j < sc+n; j += STARNEIG_MATRIX_SN(source)) {

            // source column cordinates
            int _cbegin = MAX(sc, j);
            int _cend = MIN(sc+n, j + STARNEIG_MATRIX_SN(source));

            // destination column cordinates
            int __cbegin = _cbegin + dc - sc;
            int __cend = _cend + dc - sc;

            int source_rank = starneig_get_elem_owner_matrix_descr(
                _rbegin, _cbegin, source);
            int dest_rank = starneig_get_elem_owner_matrix_descr(
                __rbegin, __cbegin, dest);

            //
            // prepare the data handle
            //

            starpu_data_handle_t handle;
            starpu_matrix_data_register(&handle, -1, 0,
                _rend-_rbegin, _rend-_rbegin, _cend-_cbegin, sizeof(double));

#ifdef STARNEIG_ENABLE_MPI
            if (mpi != NULL)
                starpu_mpi_data_register_comm(
                    handle, mpi->tag_offset++, source_rank,
                    starneig_mpi_get_comm());
#endif

            //
            // source copies the window to the data handle
            //

            if (my_rank == source_rank) {
                struct packing_helper *helper = starneig_init_packing_helper();

                struct packing_info packing_info;
                starneig_pack_window(STARPU_R, _rbegin, _rend, _cbegin, _cend,
                    source, helper, &packing_info, 0);

                starneig_pack_handle(STARPU_W, handle, helper, 0);

                starpu_task_insert(
                    &copy_matrix_to_handle_cl,
                    STARPU_PRIORITY, prio,
                    STARPU_VALUE, &packing_info, sizeof(packing_info),
                    STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count,
                    0);

                starneig_free_packing_helper(helper);
            }

#ifdef STARNEIG_ENABLE_MPI
            if (mpi != NULL && (my_rank == source_rank || my_rank == dest_rank))
                starpu_mpi_get_data_on_node_detached(
                    starneig_mpi_get_comm(), handle, dest_rank, NULL, NULL);
#endif

            //
            // destination copies the window from the data handle
            //

            if (my_rank == dest_rank) {
                struct packing_helper *helper = starneig_init_packing_helper();

                starneig_pack_handle(STARPU_R, handle, helper, 0);

                struct packing_info packing_info;
                starneig_pack_window(
                    STARPU_RW | STARPU_COMMUTE,
                    __rbegin, __rend, __cbegin, __cend,
                    dest, helper, &packing_info, 0);

                starpu_task_insert(
                    &copy_handle_to_matrix_cl,
                    STARPU_PRIORITY, prio,
                    STARPU_VALUE, &packing_info, sizeof(packing_info),
                    STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count,
                    0);

                starneig_free_packing_helper(helper);
            }

            starneig_flush_section_matrix_descr(
                __rbegin, __rend, __cbegin, __cend, dest);

            starpu_data_unregister_submit(handle);
        }
    }
}

void starneig_insert_copy_matrix(
    int sr, int sc, int dr, int dc, int m, int n, int prio,
    starneig_matrix_descr_t source, starneig_matrix_descr_t dest,
    mpi_info_t mpi)
{
    if (m < 1 || n < 1)
        return;

    STARNEIG_ASSERT(0 <= sr && sr+m <= STARNEIG_MATRIX_M(source));
    STARNEIG_ASSERT(0 <= sc && sc+n <= STARNEIG_MATRIX_N(source));
    STARNEIG_ASSERT(0 <= dr && dr+m <= STARNEIG_MATRIX_M(dest));
    STARNEIG_ASSERT(0 <= dc && dc+n <= STARNEIG_MATRIX_N(dest));

    if (mpi == NULL) {

        //
        // pack data handles
        //

        struct packing_helper *helper = starneig_init_packing_helper();

        struct packing_info packing_info_source;
        starneig_pack_window(STARPU_R, sr, sr+m, sc, sc+n,
            source, helper, &packing_info_source, 0);

        starneig_pack_cached_scratch_matrix(
            256, 256, sizeof(double), helper);

        struct packing_info packing_info_dest;
        starneig_pack_window(STARPU_RW, dr, dr+m, dc, dc+n,
            dest, helper, &packing_info_dest, 0);

        //
        // insert task
        //

        starpu_task_insert(
            &copy_matrix_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info_source,
                sizeof(packing_info_source),
            STARPU_VALUE, &packing_info_dest,
                sizeof(packing_info_dest),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

        starneig_free_packing_helper(helper);
    }
    else {

        //
        // loop over destination section rows
        //

        int roffset = ((STARNEIG_MATRIX_RBEGIN(dest) + dr) /
            STARNEIG_MATRIX_SM(dest)) * STARNEIG_MATRIX_SM(dest) -
            STARNEIG_MATRIX_RBEGIN(dest);
        for (int i = roffset; i < dr+m; i += STARNEIG_MATRIX_SM(dest)) {

            // destination row cordinates
            int _rbegin = MAX(dr, i);
            int _rend = MIN(dr+m, i + STARNEIG_MATRIX_SM(dest));

            //
            // loop over destination section columns
            //

            int coffset = ((STARNEIG_MATRIX_CBEGIN(dest) + dc) /
                STARNEIG_MATRIX_SN(dest)) * STARNEIG_MATRIX_SN(dest) -
                STARNEIG_MATRIX_CBEGIN(dest);
            for (int j = coffset; j < dc+n; j += STARNEIG_MATRIX_SN(dest)) {

                // destination column cordinates
                int _cbegin = MAX(dc, j);
                int _cend = MIN(dc+n, j + STARNEIG_MATRIX_SN(dest));

                insert_copy_matrix_reverse(
                    _rbegin + sr - dr, _cbegin + sc - dc, _rbegin, _cbegin,
                    _rend - _rbegin, _cend - _cbegin, prio, source, dest, mpi);
            }
        }
    }
}

void starneig_insert_copy_matrix_to_handle(
    int rbegin, int rend, int cbegin, int cend, int prio,
    starneig_matrix_descr_t source, starpu_data_handle_t dest,
    mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    struct packing_info packing_info;
    starneig_pack_window(
        STARPU_R, rbegin, rend, cbegin, cend, source, helper, &packing_info, 0);

    starneig_pack_handle(STARPU_W, dest, helper, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &copy_matrix_to_handle_cl,
            STARPU_EXECUTE_ON_NODE,
            starpu_mpi_data_get_rank(dest),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &copy_matrix_to_handle_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_insert_copy_handle_to_matrix(
    int rbegin, int rend, int cbegin, int cend, int prio,
    starpu_data_handle_t source, starneig_matrix_descr_t dest,
    mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, source, helper, 0);

    struct packing_info packing_info;
    starneig_pack_window(
        STARPU_RW, rbegin, rend, cbegin, cend, dest, helper, &packing_info,0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &copy_handle_to_matrix_cl,
            STARPU_EXECUTE_ON_NODE,
            starpu_mpi_data_get_rank(source),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &copy_handle_to_matrix_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_insert_set_to_identity(
    int prio, starneig_matrix_descr_t descr, mpi_info_t mpi)
{
    struct packing_helper *helper = starneig_init_packing_helper();

    struct packing_info packing_info;
    starneig_pack_window(
        STARPU_RW, 0, STARNEIG_MATRIX_M(descr), 0, STARNEIG_MATRIX_N(descr),
        descr, helper, &packing_info, 0);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_task_insert(
            starneig_mpi_get_comm(),
            &set_to_identity_cl,
            STARPU_EXECUTE_ON_NODE,
            starneig_get_elem_owner_matrix_descr(0, 0, descr),
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
    else
#endif
        starpu_task_insert(
            &set_to_identity_cl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &packing_info, sizeof(packing_info),
            STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}

void starneig_insert_scan_diagonal(
    int begin, int end, int mask_begin,
    int up, int down, int left, int right, int prio,
    void (*func)(
        int, int, int, int, int, int, int, void const *, void const *,
        void const *, void **masks),
    void const *arg, starneig_matrix_descr_t A, starneig_matrix_descr_t B,
    mpi_info_t mpi, ...)
{
    //
    // extract mask vectors
    //

    starneig_vector_descr_t mask[SCAN_DIAGONAL_MAX_MASKS];
    starneig_vector_descr_t first = NULL;
    int num_masks = 0;
    {
        va_list vl;
        va_start(vl, mpi);

        first = mask[num_masks++] = va_arg(vl, starneig_vector_descr_t);
        if (first == NULL) {
            va_end(vl);
            return;
        }

        starneig_vector_descr_t val = va_arg(vl, starneig_vector_descr_t);
        while (val != NULL) {
            STARNEIG_ASSERT(num_masks < SCAN_DIAGONAL_MAX_MASKS);
            mask[num_masks++] = val;
            val = va_arg(vl, starneig_vector_descr_t);
        }

        va_end(vl);
    }

    //
    // loop over mask tiles (align everyting with the first mask vector)
    //

    int bm = STARNEIG_VECTOR_BM(first);
    int rbegin = STARNEIG_VECTOR_RBEGIN(first);
    int mask_end = mask_begin + (end-begin);

    for (int i = (rbegin+mask_begin)/bm; i < (rbegin+mask_end-1)/bm+1; i++) {

        int _begin = MAX(mask_begin,   i*bm - rbegin);
        int _end   = MIN(mask_end, (i+1)*bm - rbegin);

        //
        // compute padded window boundaries
        //

        int _rbegin =
            MAX(0,                  _begin + begin - mask_begin - up);
        int _rend   =
            MIN(STARNEIG_MATRIX_M(A), _end + begin - mask_begin + down);
        int _cbegin =
            MAX(0,                  _begin + begin - mask_begin - left);
        int _cend   =
            MIN(STARNEIG_MATRIX_N(A), _end + begin - mask_begin + right);

        //
        // pack buffers
        //

        struct packing_helper *helper = starneig_init_packing_helper();

        struct packing_info packing_info_A;
        starneig_pack_window(STARPU_R, _rbegin, _rend, _cbegin, _cend,
            A, helper, &packing_info_A, PACKING_MODE_DEFAULT);

        struct packing_info packing_info_B;
        starneig_pack_window(STARPU_R, _rbegin, _rend, _cbegin, _cend,
            B, helper, &packing_info_B, PACKING_MODE_DEFAULT);

        struct range_packing_info packing_info_mask[SCAN_DIAGONAL_MAX_MASKS];
        for (int j = 0; j < num_masks; j++)
            starneig_pack_range(STARPU_RW, _begin, _end,
                mask[j], helper, &packing_info_mask[j], PACKING_MODE_DEFAULT);

        //
        // insert task
        //

        int __rbegin = MIN(up,   _rbegin - begin);
        int __cbegin = MIN(left, _cbegin - begin);

#ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL)
            starpu_mpi_task_insert(
                starneig_mpi_get_comm(),
                &scan_diagonal_cl,
                STARPU_EXECUTE_ON_NODE,
                starneig_get_tile_owner_vector_descr(i, first),
                STARPU_PRIORITY, prio,
                STARPU_VALUE, &num_masks, sizeof(num_masks),
                STARPU_VALUE, &__rbegin, sizeof(__rbegin),
                STARPU_VALUE, &__cbegin, sizeof(__cbegin),
                STARPU_VALUE, &func, sizeof(func),
                STARPU_VALUE, &arg, sizeof(arg),
                STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
                STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
                STARPU_VALUE, packing_info_mask,
                    num_masks*sizeof(packing_info_mask[0]),
                STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);
        else
#endif
            starpu_task_insert(
                &scan_diagonal_cl,
                STARPU_PRIORITY, prio,
                STARPU_VALUE, &num_masks, sizeof(num_masks),
                STARPU_VALUE, &__rbegin, sizeof(__rbegin),
                STARPU_VALUE, &__cbegin, sizeof(__cbegin),
                STARPU_VALUE, &func, sizeof(func),
                STARPU_VALUE, &arg, sizeof(arg),
                STARPU_VALUE, &packing_info_A, sizeof(packing_info_A),
                STARPU_VALUE, &packing_info_B, sizeof(packing_info_B),
                STARPU_VALUE, packing_info_mask,
                    num_masks*sizeof(packing_info_mask[0]),
                STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

        starneig_free_packing_helper(helper);
    }
}

static void extract_eigenvalues_func(
    int size, int rbegin, int cbegin, int m, int n, int ldA, int ldB,
    void const *arg, void const *_A, void const *_B, void **masks)
{
    double const *A = _A;
    double const *B = _B;
    double *real = masks[0];
    double *imag = masks[1];
    double *beta = masks[2];

    int i = 0;
    if (0 < rbegin && 0 < cbegin && A[(cbegin-1)*ldA+rbegin] != 0.0)
        i--;

    while (i < size) {

        int _i = rbegin+i;
        int _j = cbegin+i;

        if (_i+1 < m && A[_j*ldA+_i+1] != 0.0) {
            double real1, imag1, real2, imag2, beta1, beta2;
            starneig_compute_complex_eigenvalue(
                ldA, ldB, &A[_j*ldA+_i], B ? &B[_j*ldB+_i] : NULL,
                &real1, &imag1, &real2, &imag2, &beta1, &beta2);
            if (0 <= i) {
                real[i] = real1;
                imag[i] = imag1;
                if (beta)
                    beta[i] = beta1;
            }
            if (i+1 < size) {
                real[i+1] = real2;
                imag[i+1] = imag2;
                if (beta)
                    beta[i+1] = beta2;
            }
            i += 2;
        }
        else {
            if (B != NULL) {
                real[i] = A[_j*ldA+_i];
                imag[i] = 0.0;
                beta[i] = B[_j*ldB+_i];
            }
            else {
                real[i] = A[_j*ldA+_i];
                imag[i] = 0.0;
            }
            i++;
        }
    }
}

void starneig_insert_extract_eigenvalues(
    int prio,
    starneig_matrix_descr_t A, starneig_matrix_descr_t B,
    starneig_vector_descr_t real, starneig_vector_descr_t imag,
    starneig_vector_descr_t beta, mpi_info_t mpi)
{
    starneig_insert_scan_diagonal(
        0, STARNEIG_MATRIX_N(A), 0, 1, 1, 1, 1, prio,
        extract_eigenvalues_func, NULL, A, B, mpi, real, imag, beta, NULL);
}

void starneig_insert_set_to_zero(int prio, starpu_data_handle_t tile)
{
    starpu_task_insert(&set_to_zero_cl,
        STARPU_PRIORITY, prio, STARPU_W, tile, 0);
}
