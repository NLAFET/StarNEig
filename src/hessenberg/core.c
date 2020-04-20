///
/// @file This file contains the Hessenberg reduction task insertion function.
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
#include "core.h"
#include "tasks.h"
#include "../common/scratch.h"
#include "../common/tasks.h"

#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

///
/// @brief The noncritical updates are added to a special update chain and
/// inserted separately.
///
struct update {
    int i;
    int nb;
    starpu_data_handle_t P_h;
    starpu_data_handle_t V_h;
    starpu_data_handle_t T_h;
    struct update *next;
};

///
/// @brief Inserts left-hand side updates.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] rbegin
///         First row that belongs to the update window.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last column that belongs to the update window + 1.
///
/// @param[in] nb
///         Panel width.
///
/// @param[in] V_h
///         Matrix V handle.
///
/// @param[in]  T_h
///         Matrix T handle.
///
/// @param[in,out] matrix_a
///         Pointer to the A matrix descriptor structure.
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_left_updates(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_t matrix_a, mpi_info_t mpi)
{
    starneig_matrix_t W = starneig_matrix_init(
        STARNEIG_MATRIX_N(matrix_a), nb, STARNEIG_MATRIX_BN(matrix_a), nb,
        STARNEIG_MATRIX_SN(matrix_a), nb, sizeof(double),
        starneig_single_owner_matrix_descr, (int[]){0}, mpi);

    //
    // loop over tile columns
    //

    int _cbegin;
    _cbegin = cbegin;
    while (_cbegin < cend) {
        int _rbegin, offset;
        int _cend = MIN(cend,
            starneig_matrix_cut_hor_right(_cbegin+1, matrix_a));

        //
        // loop over tile rows, compute W <- A^T V T
        //

        _rbegin = rbegin; offset = 0;
        while (_rbegin < rend) {
            int _rend =
                MIN(rend, starneig_matrix_cut_ver_down(_rbegin+1, matrix_a));
            starneig_hessenberg_insert_update_left_a(
                prio, _rbegin, _rend, _cbegin, _cend, nb, offset,
                V_h, T_h, matrix_a, W, mpi);
            offset += _rend - _rbegin;
            _rbegin = _rend;
        }

        _cbegin = _cend;
    }

    //
    // loop over tile columns
    //

    _cbegin = cbegin;
    while (_cbegin < cend) {
        int _rbegin, offset;
        int _cend = MIN(cend,
            starneig_matrix_cut_hor_right(_cbegin+1, matrix_a));

        //
        // loop over tile rows, compute A <- A - V W^T
        //

        _rbegin = rbegin; offset = 0;
        while (_rbegin < rend) {
            int _rend =
                MIN(rend, starneig_matrix_cut_ver_down(_rbegin+1, matrix_a));
            starneig_hessenberg_insert_update_left_b(
                prio, _rbegin, _rend, _cbegin, _cend, nb, offset,
                V_h, W, matrix_a, mpi);
            offset += _rend - _rbegin;
            _rbegin = _rend;
        }

        _cbegin = _cend;
    }

    starneig_matrix_free(W);
}

///
/// @brief Inserts right-hand side updates.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] rbegin
///         First row that belongs to the update window.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last column that belongs to the update window + 1.
///
/// @param[in] nb
///         Panel width.
///
/// @param[in] V_h
///         Matrix V handle.
///
/// @param[in]  T_h
///         Matrix T handle.
///
/// @param[in,out] matrix_a
///         Pointer to the A matrix descriptor structure.
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_right_updates(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_t matrix_a, mpi_info_t mpi)
{
    starneig_matrix_t W = starneig_matrix_init(
        STARNEIG_MATRIX_M(matrix_a), nb, STARNEIG_MATRIX_BM(matrix_a), nb,
        STARNEIG_MATRIX_SM(matrix_a), nb, sizeof(double),
        starneig_single_owner_matrix_descr, (int[]){0}, mpi);

    //
    // loop over tile rows
    //

    int _rbegin;
    _rbegin = rbegin;
    while (_rbegin < rend) {
        int _cbegin, offset;
        int _rend = MIN(rend,
            starneig_matrix_cut_ver_down(_rbegin+1, matrix_a));

        //
        // loop over tile columns, compute W <- A V T
        //

        _cbegin = cbegin; offset = 0;
        while (_cbegin < cend) {
            int _cend =
                MIN(cend, starneig_matrix_cut_hor_right(_cbegin+1, matrix_a));
            starneig_hessenberg_insert_update_right_a(
                prio, _rbegin, _rend, _cbegin, _cend, nb, offset,
                V_h, T_h, matrix_a, W, mpi);
            offset += _cend - _cbegin;
            _cbegin = _cend;
        }

        _rbegin = _rend;
    }

    //
    // loop over tile rows
    //

    _rbegin = rbegin;
    while (_rbegin < rend) {
        int _cbegin, offset;
        int _rend = MIN(rend,
            starneig_matrix_cut_ver_down(_rbegin+1, matrix_a));

        //
        // loop over tile columns, compute A <- A - W V
        //

        _cbegin = cbegin; offset = 0;
        while (_cbegin < cend) {
            int _cend =
                MIN(cend, starneig_matrix_cut_hor_right(_cbegin+1, matrix_a));
            starneig_hessenberg_insert_update_right_b(
                prio, _rbegin, _rend, _cbegin, _cend, nb, offset,
                V_h, W, matrix_a, mpi);
            offset += _cend - _cbegin;
            _cbegin = _cend;
        }

        _rbegin = _rend;
    }

    starneig_matrix_free(W);
}

///
/// @brief Inserts noncritical updates.
///
/// @param[in] panel_width
///         Panel width.
///
/// @param[in] begin
///         First row/column to be reduced.
///
/// @param[in] end
///         Last row/column to be reduced + 1.
///
/// @param[in] critical_prio
///         Panel reduction and trailing matrix update task priority.
///
/// @param[in] update_prio
///         Update tasks priority.
///
/// @param[in] misc_prio
///         Miscellaneous task priority.
///
/// @param[in,out] matrix_q
///         Matrix Q.
///
/// @param[in,out] matrix_a
///         Matrix A.
///
/// @param[in] updates
///         Pointer to a update list.
///
/// @param[in,out] tag_offset
///         MPI info
///
static void insert_remaining(
    int panel_width, int begin, int end,
    int critical_prio, int update_prio, int misc_prio,
    starneig_matrix_t matrix_q, starneig_matrix_t matrix_a,
    struct update **updates, mpi_info_t mpi)
{
    int n = STARNEIG_MATRIX_N(matrix_a);
    int m = STARNEIG_MATRIX_M(matrix_a);

    while (*updates != NULL) {
        int i = (*updates)->i;
        int nb = (*updates)->nb;
        starpu_data_handle_t P_h = (*updates)->P_h;
        starpu_data_handle_t V_h = (*updates)->V_h;
        starpu_data_handle_t T_h = (*updates)->T_h;

        starneig_insert_copy_handle_to_matrix(i+1, end, i, i+nb,
            critical_prio, P_h, matrix_a, mpi);

        // update A from the right
        {
            int cut = starneig_matrix_cut_ver_up(i+1, matrix_a);
            insert_right_updates(critical_prio, cut, i+1, i+1, end, nb,
                V_h, T_h, matrix_a, mpi);
            insert_right_updates(update_prio, 0, cut, i+1, end, nb,
            V_h, T_h, matrix_a, mpi);
        }

        // update A from the left
        {
            int cut = starneig_matrix_cut_hor_right(end, matrix_a);
            insert_left_updates(critical_prio, i+1, end, end, cut, nb,
                V_h, T_h, matrix_a, mpi);
            insert_left_updates(update_prio, i+1, end, cut, n, nb,
                V_h, T_h, matrix_a, mpi);
        }

        // update Q from the right
        insert_right_updates(misc_prio, 0, m, i+1, end, nb,
            V_h, T_h, matrix_q, mpi);

        starpu_data_unregister_submit(V_h);
        starpu_data_unregister_submit(T_h);

        struct update *next = (*updates)->next;
        free(*updates);
        *updates = next;
    }
}

starneig_error_t starneig_hessenberg_insert_tasks(
    int panel_width, int begin, int end,
    int critical_prio, int update_prio, int misc_prio,
    starneig_matrix_t matrix_q, starneig_matrix_t matrix_a,
    bool limit_submitted, mpi_info_t mpi)
{

    // int total_worker_count = starpu_worker_get_count();


#ifdef STARNEIG_ENABLE_CUDA

    //
    // find a suitable GPU
    //

    unsigned gpu_memory_node = 0;
    ssize_t gpu_mem_size = 0;
    {
        int workers[STARPU_NMAXWORKERS];
        int worker_count = starpu_worker_get_ids_by_type(
            STARPU_CUDA_WORKER, workers, STARPU_NMAXWORKERS);

        ssize_t largest_size = 0;

        for (int i = 0; i < worker_count; i++) {
            unsigned node = starpu_worker_get_memory_node(workers[i]);
            ssize_t total = starpu_memory_get_total(node);
            starneig_verbose("GPU memory node %u contains %u MB of memory.",
                node, total/1000000);
            if (largest_size < total) {
                gpu_memory_node = node;
                gpu_mem_size = total;
            }
        }
    }
#endif

    // noncritical updates are added to a special update chain and are inserted
    // separately

    struct update *updates = NULL;
    struct update *tail = NULL;

    //
    // loop over panels
    //

    for (int i = begin; i < end-1; i += panel_width) {
        const int nb = MIN(panel_width, end-i-1);

        //
        // check whether the GPU has enough memory to store the remaining part
        // of the matrix
        //

#ifdef STARNEIG_ENABLE_CUDA
        if (0 < gpu_mem_size) {
            int rbegin = starneig_matrix_cut_ver_up(i+1, matrix_a);
            int rend = starneig_matrix_cut_ver_down(end, matrix_a);
            int cbegin = starneig_matrix_cut_hor_left(i+1, matrix_a);
            int cend = starneig_matrix_cut_hor_right(end, matrix_a);

            if ((rend-rbegin)*(cend-cbegin)*sizeof(double) < 0.75*gpu_mem_size){
                starneig_matrix_prefetch_section(
                    i+1, end, i+1, end, gpu_memory_node, 1, matrix_a);
            }
        }
#endif

        //
        // Register P (current panel), and V, T and Y from
        //
        //    (I - V T V^T)^T A (I - V T V^T)
        //  = (I - V T V^T)^T (A - Y V^T).
        //

        starpu_data_handle_t P_h, V_h, T_h, Y_h;
        starpu_matrix_data_register(
            &P_h, -1, 0, end-i-1, end-i-1, nb, sizeof(double));
        starpu_matrix_data_register(
            &V_h, -1, 0, end-i-1, end-i-1, nb, sizeof(double));
        starpu_matrix_data_register(
            &T_h, -1, 0, nb, nb, nb, sizeof(double));
        starpu_matrix_data_register(
            &Y_h, -1, 0, end-i-1, end-i-1, nb, sizeof(double));

#ifdef STARNEIG_ENABLE_MPI
        if (mpi != NULL) {
            starpu_mpi_data_register_comm(
                P_h, mpi->tag_offset++, 0, starneig_mpi_get_comm());
            starpu_mpi_data_register_comm(
                V_h, mpi->tag_offset++, 0, starneig_mpi_get_comm());
            starpu_mpi_data_register_comm(
                T_h, mpi->tag_offset++, 0, starneig_mpi_get_comm());
            starpu_mpi_data_register_comm(
                Y_h, mpi->tag_offset++, 0, starneig_mpi_get_comm());
        }
#endif

        starneig_insert_copy_matrix_to_handle(i+1, end, i, i+nb,
            critical_prio, matrix_a, P_h, mpi);


        starneig_insert_set_matrix_to_zero(critical_prio, V_h, mpi);

        //
        // loop over the columns in the panel
        //

        for (int j = 0; j < nb; j++) {

            // register v from (I - v \tau v^T)
            starneig_vector_t v = starneig_vector_init(
                STARNEIG_MATRIX_N(matrix_a), STARNEIG_MATRIX_BN(matrix_a),
                sizeof(double), starneig_vector_single_owner_func, (int[]){0},
                mpi);

            // register y from y = A v
            starneig_vector_t y = starneig_vector_init(
                STARNEIG_MATRIX_N(matrix_a), STARNEIG_MATRIX_BN(matrix_a),
                sizeof(double), starneig_vector_single_owner_func, (int[]){0},
                mpi);

            //
            // compute v and update V
            //

            starneig_hessenberg_insert_prepare_column(
                critical_prio, j, i+1, end, Y_h, V_h, T_h, P_h, v, mpi);

            //
            // compute y
            //

            {
                int _rbegin = i+1;
                while (_rbegin < end) {
                    int _rend = MIN(end,
                        starneig_matrix_cut_ver_down(_rbegin+1, matrix_a));

                    int _cbegin = i+j+1;
                    while (_cbegin < end) {
                        int _cend = MIN(end,
                            starneig_matrix_cut_hor_right(_cbegin+1, matrix_a));

                        starneig_hessenberg_insert_compute_column(
                            critical_prio, _rbegin, _rend, _cbegin, _cend,
                            matrix_a, v, y, mpi);

                        _cbegin = _cend;
                    }

                    _rbegin = _rend;
                }
            }

            //
            // update Y and T
            //

            starneig_hessenberg_insert_finish_column(
                critical_prio, j, i+1, end, V_h, T_h, Y_h, y, mpi);

            starneig_vector_free(v);
            starneig_vector_free(y);
        }

        //
        // update the trailing matrix from the right
        //

        {
            int _cbegin = i+nb;
            while (_cbegin < end) {
                int _cend = MIN(end,
                    starneig_matrix_cut_hor_right(_cbegin+1, matrix_a));
                int _rbegin = i+1;
                while (_rbegin < end) {
                    int _rend = MIN(end,
                        starneig_matrix_cut_ver_down(_rbegin+1, matrix_a));
                    starneig_hessenberg_insert_update_trail_right(
                        critical_prio, _rbegin, _rend, _cbegin, _cend, nb,
                        _rbegin-i-1, _cbegin-i-nb, V_h, T_h, Y_h, matrix_a,
                        mpi);
                    _rbegin = _rend;
                }
                _cbegin = _cend;
            }
        }

        //
        // update the trailing matrix from the left
        //

        insert_left_updates(critical_prio, i+1, end, i+nb, end, nb,
            V_h, T_h, matrix_a, mpi);

        starpu_data_unregister_submit(Y_h);

        //
        // store the update to the update chain
        //

        if (updates == NULL) {
            // set up the update chain
            updates = malloc(sizeof(struct update));
            tail = updates;
        }
        else {
            // add the new update to the end of the update chain
            tail->next = malloc(sizeof(struct update));
            tail = tail->next;
        }

        tail->i = i;
        tail->nb = nb;
        tail->P_h = P_h;
        tail->V_h = V_h;
        tail->T_h = T_h;
        tail->next = NULL;

        //
        // pause task insertion if necessary
        //
/*
        if (limit_submitted &&
        1000*total_worker_count < starpu_task_nsubmitted()) {
            starneig_scratch_unregister();
            insert_remaining(
                panel_width, begin, end, critical_prio, update_prio, misc_prio,
                matrix_q, matrix_a, &updates, mpi);
            starneig_scratch_unregister();
            starpu_task_wait_for_n_submitted(100*total_worker_count);
        }
*/
    }

    //
    // insert delayed update tasks
    //

    starneig_scratch_unregister();
    insert_remaining(
        panel_width, begin, end, critical_prio, update_prio, misc_prio,
        matrix_q, matrix_a, &updates, mpi);

    return STARNEIG_SUCCESS;
}
