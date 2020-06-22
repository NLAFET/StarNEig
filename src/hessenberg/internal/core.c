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
    starneig_matrix_descr_t matrix_q, starneig_matrix_descr_t matrix_a,
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

        int ver_part = STARNEIG_MATRIX_BM(matrix_a);
        int hor_part = STARNEIG_MATRIX_BN(matrix_a);
        int q_part = STARNEIG_MATRIX_BM(matrix_q);

        starneig_insert_copy_handle_to_matrix(i+1, end, i, i+nb,
            critical_prio, P_h, matrix_a, mpi);

        // update A from the right
        for (int j = 0; j < i+1; j += ver_part)
            starneig_hessenberg_insert_update_right(update_prio,
                j, MIN(i+1, j+ver_part), i+1, end, nb, V_h, T_h,
                matrix_a, mpi);

        // update A from the left
        for (int j = (end/hor_part)*hor_part; j < n; j += hor_part)
            starneig_hessenberg_insert_update_left(update_prio,
                i+1, end, MAX(end, j), MIN(n, j+hor_part), nb,
                V_h, T_h, matrix_a, mpi);

        // update Q from the right
        for (int j = 0; j < m; j += q_part)
            starneig_hessenberg_insert_update_right(misc_prio,
                j, MIN(m, j+q_part), i+1, end, nb, V_h, T_h,
                matrix_q, mpi);

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
    starneig_matrix_descr_t matrix_q, starneig_matrix_descr_t matrix_a,
    bool limit_submitted, mpi_info_t mpi)
{

    int total_worker_count = starpu_worker_get_count();

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
    // insert critical tasks (panels and trailing matrix updates)
    //

    starneig_vector_descr_t v = starneig_init_vector_descr(
        STARNEIG_MATRIX_N(matrix_a), STARNEIG_MATRIX_BN(matrix_a),
        sizeof(double), NULL, NULL, NULL);
    starneig_vector_descr_t y = starneig_init_vector_descr(
        STARNEIG_MATRIX_N(matrix_a), STARNEIG_MATRIX_BN(matrix_a),
        sizeof(double), NULL, NULL, NULL);

    for (int i = begin; i < end-1; i += panel_width) {
        const int nb = MIN(panel_width, end-i-1);

        //
        // check whether the GPU has enough memory
        //

        int try_gpu = 0;
#ifdef STARNEIG_ENABLE_CUDA
        if (0 < gpu_mem_size) {
            int rbegin = starneig_matrix_cut_vectically_up(i+1, matrix_a);
            int rend = starneig_matrix_cut_vectically_down(end, matrix_a);
            int cbegin = starneig_matrix_cut_horizontally_left(i+1, matrix_a);
            int cend = starneig_matrix_cut_horizontally_right(end, matrix_a);

            if ((rend-rbegin)*(cend-cbegin)*sizeof(double) < 0.75*gpu_mem_size){
                starneig_prefetch_section_matrix_descr(
                    i+1, end, i+1, end, gpu_memory_node, 1, matrix_a);
                try_gpu = 1;
            }
        }
#endif

        starpu_data_handle_t P_h, V_h, T_h, Y_h;
        starpu_matrix_data_register(
            &P_h, -1, 0, end-i-1, end-i-1, nb, sizeof(double));
        starpu_matrix_data_register(
            &V_h, -1, 0, end-i-1, end-i-1, nb, sizeof(double));
        starpu_matrix_data_register(
            &T_h, -1, 0, nb, nb, nb, sizeof(double));
        starpu_matrix_data_register(
            &Y_h, -1, 0, end-i-1, end-i-1, nb, sizeof(double));

        //
        // loop over the columns in the panel
        //

        starneig_insert_copy_matrix_to_handle(i+1, end, i, i+nb,
            critical_prio, matrix_a, P_h, NULL);

        for (int j = 0; j < nb; j++) {
            starneig_hessenberg_insert_prepare_column(
                critical_prio, j, i+1, end, Y_h, V_h, T_h, P_h, v);

            // trailing matrix operation
            if (try_gpu) {
                starneig_hessenberg_insert_compute_column(
                    critical_prio, i+1, end, i+j+1, end,
                    matrix_a, v, y);
            }
            else {
                int _begin = i+1;
                while (_begin < end) {
                    int _end = MIN(end, starneig_matrix_cut_vectically_down(
                        _begin+1, matrix_a));
                    starneig_hessenberg_insert_compute_column(
                        critical_prio, _begin, _end, i+j+1, end,
                        matrix_a, v, y);
                    _begin = _end;
                }
            }

            starneig_hessenberg_insert_finish_column(
                critical_prio, j, i+1, end, V_h, T_h, Y_h, y);
        }

        //
        // update the trailing matrix
        //

        if (try_gpu) {
            starneig_hessenberg_insert_update_trail(
                critical_prio, i+1, end, i+nb, end, nb, 0,
                V_h, T_h, Y_h, matrix_a, mpi);
        }
        else {
            int _begin = i+nb;
            while (_begin < end) {
                int _end = MIN(end,
                    starneig_matrix_cut_horizontally_right(_begin+1, matrix_a));
                starneig_hessenberg_insert_update_trail(
                    critical_prio, i+1, end, _begin, _end, nb, _begin-i-nb,
                    V_h, T_h, Y_h, matrix_a, mpi);
                _begin = _end;
            }
        }

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

        if (limit_submitted &&
        100*total_worker_count < starpu_task_nsubmitted()) {
            starneig_scratch_unregister();
            insert_remaining(
                panel_width, begin, end, critical_prio, update_prio, misc_prio,
                matrix_q, matrix_a, &updates, mpi);
            starneig_scratch_unregister();
            starpu_task_wait_for_n_submitted(10*total_worker_count);
        }
    }

    starneig_free_vector_descr(v);
    starneig_free_vector_descr(y);

    //
    // insert delayed update tasks
    //

    starneig_scratch_unregister();
    insert_remaining(
        panel_width, begin, end, critical_prio, update_prio, misc_prio,
        matrix_q, matrix_a, &updates, mpi);

    return STARNEIG_SUCCESS;
}
