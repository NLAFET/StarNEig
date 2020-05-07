///
/// @file This file contains the Hessenberg reduction interface functions.
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
#include <starneig/sep_sm.h>
#include "../common/node_internal.h"
#include "../common/trace.h"
#include "core.h"
#include <math.h>

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
        int workers = starpu_worker_get_count();
        conf->tile_size = MAX(256, MIN(4096, divceil(n/sqrt(8*workers), 8)*8));
        starneig_message("Setting tile size to %d.", conf->tile_size);
    }
    else {
        if (conf->tile_size < 8) {
            starneig_error("Invalid tile size. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    if (conf->panel_width == STARNEIG_HESSENBERG_DEFAULT_PANEL_WIDTH) {
        conf->panel_width =
            MAX(64, divceil(0.001875596476 * n + 273.5908216, 8)*8);
        starneig_message("Setting panel width to %d.", conf->panel_width);
    }
    else {
        if (conf->panel_width < 8) {
            starneig_error("Invalid panel width. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    //
    // register, partition and pack
    //

    starneig_matrix_t matrix_a = starneig_matrix_register(
        MATRIX_TYPE_FULL, n, n, conf->tile_size, conf->tile_size,
        -1, -1, ldA, sizeof(double), NULL, NULL, A, NULL);
    STARNEIG_EVENT_SET_LABEL(matrix_a, 'A');

    starneig_matrix_t matrix_q = starneig_matrix_register(
        MATRIX_TYPE_FULL, n, n, conf->tile_size, conf->tile_size,
        -1, -1, ldQ, sizeof(double), NULL, NULL, Q, NULL);
    STARNEIG_EVENT_SET_LABEL(matrix_q, 'Q');

    //
    // insert tasks
    //

    STARNEIG_EVENT_INIT();

    starneig_error_t ret = starneig_hessenberg_insert_tasks(
        conf->panel_width, begin, end,
        STARPU_MAX_PRIO, STARPU_DEFAULT_PRIO, STARPU_MIN_PRIO,
        matrix_q, matrix_a, true, NULL);

    //
    // finalize
    //

    starneig_matrix_unregister(matrix_a);
    starneig_matrix_unregister(matrix_q);

    starneig_matrix_free(matrix_a);
    starneig_matrix_free(matrix_q);

    STARNEIG_EVENT_STORE(n, "trace.dat");
    STARNEIG_EVENT_FREE();

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__attribute__ ((visibility ("default")))
void starneig_hessenberg_init_conf(struct starneig_hessenberg_conf *conf) {
    conf->tile_size = STARNEIG_HESSENBERG_DEFAULT_TILE_SIZE;
    conf->panel_width = STARNEIG_HESSENBERG_DEFAULT_PANEL_WIDTH;
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
