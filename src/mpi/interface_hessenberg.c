///
/// @file
///
/// @brief
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
#include <starneig/distr_helpers.h>
#include <starneig/sep_dm.h>
#include <starneig/gep_dm.h>
#include "../common/node_internal.h"
#include "../common/utils.h"
#include "../mpi/utils.h"
#include "../mpi/node_internal.h"
#include "../hessenberg/core.h"
#include <starpu.h>
#include <starpu_mpi.h>

static starneig_error_t hessenberg_mpi(
    struct starneig_hessenberg_conf const *_conf, int begin, int end,
    struct starneig_distr_matrix *Q, struct starneig_distr_matrix *A)
{
    // use default configuration if necessary
    struct starneig_hessenberg_conf *conf;
    struct starneig_hessenberg_conf local_conf;
    if (_conf == NULL)
        starneig_hessenberg_init_conf(&local_conf);
    else
        local_conf = *_conf;
    conf = &local_conf;

    int n = A->rows;

    //
    // check configuration
    //

    int preferred_size;
    if (conf->tile_size == STARNEIG_HESSENBERG_DEFAULT_TILE_SIZE) {
        int workers = starpu_worker_get_count();
        preferred_size = MAX(256, MIN(4096, divceil(n/sqrt(8*workers), 8)*8));
    }
    else {
        if (conf->tile_size < 8) {
            starneig_error("Invalid tile size. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
        preferred_size = conf->tile_size;
    }

    starneig_message("Attempting to set tile size to %d.", preferred_size);

    conf->tile_size =
        starneig_mpi_find_valid_tile_size(preferred_size, A, NULL, Q, NULL);

    if (conf->tile_size < 8) {
        starneig_error("Cannot find a valid tile size. Exiting...");
        return STARNEIG_INVALID_DISTR_MATRIX;
    }

    starneig_message("Setting tile size to %d.", conf->tile_size);

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

    mpi_info_t mpi = starneig_mpi_get_info();

    //
    // register, partition and pack
    //

    starneig_matrix_t A_d =
        starneig_mpi_cache_convert_and_release(
            conf->tile_size, conf->tile_size,
            MATRIX_TYPE_FULL, A, mpi);

    starneig_matrix_t Q_d = NULL;
    if (Q != NULL)
        Q_d = starneig_mpi_cache_convert_and_release(
            conf->tile_size, conf->tile_size,
            MATRIX_TYPE_FULL, Q, mpi);

    //
    // insert tasks
    //

    starneig_error_t err = starneig_hessenberg_insert_tasks(
        conf->panel_width, begin, end,
        STARPU_MAX_PRIO, STARPU_DEFAULT_PRIO, STARPU_MIN_PRIO,
        Q_d, A_d, false, mpi);

    //
    // finalize
    //

    starneig_matrix_acquire(A_d);
    starneig_matrix_acquire(Q_d);

    return err;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Hessenberg_expert(
    struct starneig_hessenberg_conf *conf,
    int begin, int end,
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t Q)
{
    if (A == NULL)      return -2;
    if (Q == NULL)      return -3;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    starneig_error_t ret = hessenberg_mpi(
        conf, begin, end, Q, A);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Hessenberg(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t Q)
{
    if (A == NULL)      return -1;
    if (Q == NULL)      return -2;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int m = starneig_distr_matrix_get_rows(A);

    return starneig_SEP_DM_Hessenberg_expert(NULL, 0, m, A, Q);
}
