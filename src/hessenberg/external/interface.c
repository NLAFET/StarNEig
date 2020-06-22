///
/// @file This file contains the Hessenberg reduction interface functions.
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
#include <starneig/sep_sm.h>
#include "../common/node_internal.h"
#include "core.h"

static starneig_error_t hessenberg(
    struct starneig_hessenberg_conf const *_conf,
    int n, int begin, int end, int ldQ, int ldA, double *Q, double *A)
{
    // use default configuration if necessary
    struct starneig_hessenberg_conf *conf;
    struct starneig_hessenberg_conf local_conf;
    if (_conf == NULL)
        starneig_hessenberg_init_conf(&local_conf);
    else
        local_conf = *_conf;
    conf = &local_conf;

    //
    // check configuration
    //

    if (conf->tile_size == STARNEIG_HESSENBERG_DEFAULT_TILE_SIZE) {
        conf->tile_size = MAX(96, MIN(((n/5)/5)*8, 320));
        starneig_message("Setting tile size to %d.", conf->tile_size);
    }
    else {
        if (conf->tile_size < 8) {
            starneig_error("Invalid tile size. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    if (conf->panel_width == STARNEIG_HESSENBERG_DEFAULT_PANEL_WIDTH) {
        conf->panel_width = MAX(64, conf->tile_size/2);
        starneig_message("Setting panel width to %d.", conf->panel_width);
    }
    else {
        if (conf->panel_width < 8) {
            starneig_error("Invalid panel width. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    if (conf->parallel_worker_size !=
    STARNEIG_HESSENBERG_DEFAULT_PARALLEL_WORKER_SIZE) {
        if (conf->parallel_worker_size < 1) {
            starneig_error("Invalid parallel worker size. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    int worker_ids[STARPU_NMAXWORKERS];
    int worker_count = starpu_worker_get_ids_by_type(
        STARPU_CPU_WORKER, worker_ids, STARPU_NMAXWORKERS);

#ifdef STARNEIG_ENABLE_CUDA
    int gpu_worker_ids[STARPU_NMAXWORKERS];
    int gpu_worker_count = starpu_worker_get_ids_by_type(
        STARPU_CUDA_WORKER, gpu_worker_ids, STARPU_NMAXWORKERS);
#endif

    //
    // create parallel scheduler
    //

    int parallel_ctx_size;
    if (conf->parallel_worker_size ==
    STARNEIG_HESSENBERG_DEFAULT_PARALLEL_WORKER_SIZE) {
#ifdef STARNEIG_ENABLE_CUDA
        if (0 < gpu_worker_count)
            parallel_ctx_size = worker_count;
        else
#endif
            parallel_ctx_size = MAX(1, MIN(worker_count-1, 5*worker_count/6));
    } else {
        parallel_ctx_size = MIN(worker_count, conf->parallel_worker_size);
    }

    starneig_verbose(
        "Adding %d CPU workers to the parallel scheduling context.",
        parallel_ctx_size);

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
    unsigned parallel_ctx = starpu_sched_ctx_create(
        worker_ids, parallel_ctx_size, "parallel_cxt", 0);
#else
    unsigned parallel_ctx = starpu_sched_ctx_create(
        worker_ids, parallel_ctx_size, "parallel_cxt",
        STARPU_SCHED_CTX_POLICY_NAME, "peager", 0);
#endif

    //
    // create regular scheduler
    //

    char *other_ctx_sched = "prio";
#ifdef STARNEIG_ENABLE_CUDA
    if (0 < gpu_worker_count)
        other_ctx_sched = "dmdas";
#endif

    int other_ctx_size;
#ifdef STARNEIG_ENABLE_CUDA
        if (0 < gpu_worker_count)
            other_ctx_size = worker_count - parallel_ctx_size;
        else
#endif
            other_ctx_size = MAX(1, worker_count - parallel_ctx_size);

    starneig_verbose(
        "Adding %d CPU workers to the other scheduling context.",
        other_ctx_size);

    unsigned other_ctx = starpu_sched_ctx_create(
        worker_ids+worker_count-other_ctx_size, other_ctx_size, "other_ctx",
        STARPU_SCHED_CTX_POLICY_NAME, other_ctx_sched, 0);

#ifdef STARNEIG_ENABLE_CUDA
    // add GPUs to the regular scheduler
    if (0 < gpu_worker_count) {
        starneig_verbose(
            "Adding %d GPU workers to the other scheduling context.",
            gpu_worker_count);
        starpu_sched_ctx_add_workers(
            gpu_worker_ids, gpu_worker_count, other_ctx);
    }
#endif

    //starpu_sched_ctx_set_inheritor(parallel_ctx, other_ctx);

    //
    // register, partition and pack
    //

    starneig_matrix_descr_t matrix_a = starneig_register_matrix_descr(
        MATRIX_TYPE_FULL, n, n, conf->tile_size, conf->tile_size,
        -1, -1, ldA, sizeof(double), NULL, NULL, A, NULL);

    starneig_matrix_descr_t matrix_q = starneig_register_matrix_descr(
        MATRIX_TYPE_FULL, n, n, conf->tile_size, conf->tile_size,
        -1, -1, ldQ, sizeof(double), NULL, NULL, Q, NULL);

    //
    // insert tasks
    //

    starneig_error_t ret = starneig_hessenberg_ext_insert_tasks(
        conf->panel_width, begin, end, parallel_ctx, other_ctx,
        STARPU_MAX_PRIO, STARPU_DEFAULT_PRIO, STARPU_MIN_PRIO,
        matrix_q, matrix_a, NULL);

    //
    // finalize
    //

    starpu_sched_ctx_finished_submit(parallel_ctx);
    starpu_sched_ctx_finished_submit(other_ctx);

    // move workers manually (starpu_sched_ctx_set_inheritor does not work with
    // StarPU 1.2.6)
    {
        starpu_task_wait_for_all_in_ctx(parallel_ctx);
        int *workers;
        starpu_sched_ctx_get_workers_list(parallel_ctx, &workers);
        int worker_count = starpu_sched_ctx_get_nworkers(parallel_ctx);

        if (0 < worker_count) {
            starneig_verbose(
                "Moving %d CPU workers from the parallel scheduling context "
                "to the other scheduling context.", worker_count);
            starpu_sched_ctx_add_workers(workers, worker_count, other_ctx);
        }

        free(workers);
    }

    starneig_unregister_matrix_descr(matrix_a);
    starneig_unregister_matrix_descr(matrix_q);

    starneig_free_matrix_descr(matrix_a);
    starneig_free_matrix_descr(matrix_q);

    starpu_task_wait_for_all_in_ctx(parallel_ctx);
    starpu_task_wait_for_all_in_ctx(other_ctx);

    starpu_sched_ctx_delete(parallel_ctx);
    starpu_sched_ctx_delete(other_ctx);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__attribute__ ((visibility ("default")))
void starneig_hessenberg_init_conf(struct starneig_hessenberg_conf *conf) {
    conf->tile_size = STARNEIG_HESSENBERG_DEFAULT_TILE_SIZE;
    conf->panel_width = STARNEIG_HESSENBERG_DEFAULT_PANEL_WIDTH;
    conf->parallel_worker_size =
        STARNEIG_HESSENBERG_DEFAULT_PARALLEL_WORKER_SIZE;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_SM_Hessenberg_expert(
    struct starneig_hessenberg_conf *conf,
    int n, int begin, int end,
    double A[], int ldA,
    double Q[], int ldQ)
{
    if (n < 1)          return -2;
    if (begin < 0)      return -3;
    if (n < end)        return -4;
    if (A == NULL)      return -5;
    if (ldA < n)        return -6;
    if (Q == NULL)      return -7;
    if (ldQ < n)        return -8;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_SM);
    starneig_node_resume_starpu();

    starneig_error_t ret = hessenberg(conf, n, begin, end, ldQ, ldA, Q, A);

    starpu_task_wait_for_all();
    starneig_node_pause_starpu();
    starneig_node_set_mode(STARNEIG_MODE_OFF);
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_SM_Hessenberg(
    int n,
    double A[], int ldA,
    double Q[], int ldQ)
{
    if (n < 1)          return -1;
    if (A == NULL)      return -2;
    if (ldA < n)        return -3;
    if (Q == NULL)      return -4;
    if (ldQ < n)        return -5;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    return starneig_SEP_SM_Hessenberg_expert(NULL, n, 0, n, A, ldA, Q, ldQ);
}
