///
/// @file This file contains the Hessenberg reduction task insertion function.
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
#include "core.h"
#include "tasks.h"
#include "../common/scratch.h"

///
/// @brief Calculates how much GPU memory can be used.
///
/// @param[in] ctx
///         Scheduling context.
///
/// @return Amount of available GPU memory.
///
static ssize_t get_gpu_mem_size(unsigned ctx)
{
    int *workers;
    starpu_sched_ctx_get_workers_list(ctx, &workers);
    int worker_count = starpu_sched_ctx_get_nworkers(ctx);

    ssize_t largest = 0;

    for (int i = 0; i < worker_count; i++) {
        if (starpu_worker_get_type(workers[i]) == STARPU_CUDA_WORKER) {
            unsigned node = starpu_worker_get_memory_node(workers[i]);
            ssize_t total = starpu_memory_get_total(node);
            starneig_verbose("GPU memory node %u contains %u MB of memory.",
                node, total/1000000);
            largest = MAX(largest, total);
        }
    }

    free(workers);

    return largest;
}

starneig_error_t starneig_hessenberg_ext_insert_tasks(
    int panel_width, int begin, int end,
    unsigned parallel_ctx, unsigned other_ctx,
    int critical_prio, int update_prio, int misc_prio,
    starneig_matrix_descr_t matrix_q, starneig_matrix_descr_t matrix_a,
    mpi_info_t mpi)
{
    ssize_t gpu_mem_size = get_gpu_mem_size(other_ctx);

    int parallel =
        parallel_ctx != STARPU_NMAX_SCHED_CTXS &&
        1 < starpu_sched_ctx_get_nworkers(parallel_ctx);

    int n = STARNEIG_MATRIX_N(matrix_a);
    int m = STARNEIG_MATRIX_M(matrix_a);

    // noncritical updates are added to a special update chain and are inserted
    // separately

    struct update {
        int i;
        int nb;
        starpu_data_handle_t V_h;
        starpu_data_handle_t T_h;
        struct update *next;
    };

    struct update *updates = NULL;
    struct update *tail = NULL;

    //
    // insert critical tasks (panels and trailing matrix updates)
    //

    unsigned ctx = parallel_ctx;
    if (!parallel)
        ctx = other_ctx;

    for (int i = begin; i < end-1; i += panel_width) {
        const int nb = MIN(panel_width, end-i-1);

        if (ctx != other_ctx &&
        (end-i)*(end-i)*sizeof(double) < 0.75*gpu_mem_size) {
            starneig_verbose("Switching to the other scheduling context.");
            starneig_scratch_unregister();
            ctx = other_ctx;
        }

        starpu_data_handle_t V_h, T_h, Y_h;

        // reduce panel
        starneig_hessenberg_ext_insert_process_panel(ctx, critical_prio,
            i+1, end, i, end, nb, matrix_a, &V_h, &T_h, &Y_h, parallel,
            mpi);

        // update trail
        starneig_hessenberg_ext_insert_update_trail(ctx, critical_prio,
            i+1, end, i+nb, end, nb, V_h, T_h, Y_h, matrix_a, parallel,
            mpi);

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
        tail->V_h = V_h;
        tail->T_h = T_h;
        tail->next = NULL;
    }

    starneig_scratch_unregister();

    //
    // insert delayed update tasks
    //

    while (updates != NULL) {
        int i = updates->i;
        int nb = updates->nb;
        starpu_data_handle_t V_h = updates->V_h;
        starpu_data_handle_t T_h = updates->T_h;

        int ver_part = STARNEIG_MATRIX_BM(matrix_a);
        int hor_part = STARNEIG_MATRIX_BN(matrix_a);
        int q_part = STARNEIG_MATRIX_BM(matrix_q);

        // update A from the right
        for (int j = 0; j < i+1; j += ver_part)
            starneig_hessenberg_ext_insert_update_right(other_ctx, update_prio,
                j, MIN(i+1, j+ver_part), i+1, end, nb, V_h, T_h,
                matrix_a, mpi);

        // update A from the left
        for (int j = (end/hor_part)*hor_part; j < n; j += hor_part)
            starneig_hessenberg_ext_insert_update_left(other_ctx, update_prio,
                i+1, end, MAX(end, j), MIN(n, j+hor_part), nb,
                V_h, T_h, matrix_a, mpi);

        // update Q from the right
        for (int j = 0; j < m; j += q_part)
            starneig_hessenberg_ext_insert_update_right(other_ctx, misc_prio,
                j, MIN(m, j+q_part), i+1, end, nb, V_h, T_h,
                matrix_q, mpi);

        starpu_data_unregister_submit(V_h);
        starpu_data_unregister_submit(T_h);

        struct update *next = updates->next;
        free(updates);
        updates = next;
    }

    return STARNEIG_SUCCESS;
}
