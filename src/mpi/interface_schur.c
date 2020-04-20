///
/// @file
///
/// @brief
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
#include <starneig/distr_helpers.h>
#include <starneig/sep_dm.h>
#include <starneig/gep_dm.h>
#include "../common/node_internal.h"
#include "../common/utils.h"
#include "../mpi/utils.h"
#include "../mpi/node_internal.h"
#include "../schur/core.h"
#include "../schur/process_args.h"
#include <starpu.h>
#include <starpu_mpi.h>

static starneig_error_t schur_mpi(
    struct starneig_schur_conf const *_conf, struct starneig_distr_matrix *Q,
    struct starneig_distr_matrix *Z, struct starneig_distr_matrix *A,
    struct starneig_distr_matrix *B, double *real, double *imag, double *beta,
    mpi_info_t mpi)
{
    // use default configuration if necessary
    struct starneig_schur_conf *conf;
    struct starneig_schur_conf local_conf;
    if (_conf == NULL)
        starneig_schur_init_conf(&local_conf);
    else
        local_conf = *_conf;
    conf = &local_conf;

    int n = A->rows;

    //
    // select tile size
    //

    int world_size = starneig_mpi_get_comm_size();

    int preferred_size;
    if (conf->tile_size == STARNEIG_SCHUR_DEFAULT_TILE_SIZE) {
        int worker_count = starpu_worker_get_count();
        int optimal;
        if (B != NULL)
            optimal = divceil(
                0.8*starneig_schur_get_optimal_tile_size(n, worker_count), 8)*8;
        else
            optimal = starneig_schur_get_optimal_tile_size(n, worker_count);
        int naive = divceil(n/(world_size*worker_count), 8)*8;

        preferred_size = MAX(64, MIN(512,
            MAX(divceil(optimal/2, 8)*8, MIN(optimal, naive))));
    }
    else {
        preferred_size = conf->tile_size;
    }

    starneig_message("Attempting to set tile size to %d.", preferred_size);

    int tile_size =
        starneig_mpi_find_valid_tile_size(preferred_size, A, B, Q, Z);

    if (tile_size < 8) {
        starneig_error("Cannot find a valid tile size. Exiting...");
        return STARNEIG_INVALID_DISTR_MATRIX;
    }

    starneig_message("Setting tile size to %d.", tile_size);
    conf->tile_size = tile_size;

    //
    // register, partition and pack
    //

    starneig_matrix_descr_t A_d =
        starneig_mpi_cache_convert_and_release(
            conf->tile_size, conf->tile_size,
            MATRIX_TYPE_FULL, A, mpi);

    starneig_matrix_descr_t B_d = NULL;
    if (B != NULL)
        B_d = starneig_mpi_cache_convert_and_release(
            conf->tile_size, conf->tile_size,
            MATRIX_TYPE_FULL, B, mpi);

    starneig_matrix_descr_t Q_d = NULL;
    if (Q != NULL)
        Q_d = starneig_mpi_cache_convert_and_release(
            conf->tile_size, conf->tile_size,
            MATRIX_TYPE_FULL, Q, mpi);

    starneig_matrix_descr_t Z_d = NULL;
    if (Z != NULL)
        Z_d = starneig_mpi_cache_convert_and_release(
            conf->tile_size, conf->tile_size,
            MATRIX_TYPE_FULL, Z, mpi);

    starneig_vector_descr_t real_d = NULL;
    if (real != NULL)
        real_d = starneig_init_matching_vector_descr(
            A_d, sizeof(double), real, mpi);

    starneig_vector_descr_t imag_d = NULL;
    if (imag != NULL)
        imag_d = starneig_init_matching_vector_descr(
            A_d, sizeof(double), imag, mpi);

    starneig_vector_descr_t beta_d = NULL;
    if (beta != NULL)
        beta_d = starneig_init_matching_vector_descr(
            A_d, sizeof(double), beta, mpi);

    //
    // insert tasks
    //

    starneig_error_t err = starneig_schur_insert_tasks(
        conf, Q_d, Z_d, A_d, B_d, real_d, imag_d, beta_d, mpi);

    //
    // finalize
    //

    for (int i = 0; real_d != NULL && i < world_size; i++)
        starneig_gather_vector_descr(i, real_d);

    for (int i = 0; imag_d != NULL && i < world_size; i++)
        starneig_gather_vector_descr(i, imag_d);

    for (int i = 0; beta_d != NULL && i < world_size; i++)
        starneig_gather_vector_descr(i, beta_d);

    starneig_acquire_matrix_descr(A_d);
    starneig_acquire_matrix_descr(B_d);
    starneig_acquire_matrix_descr(Q_d);
    starneig_acquire_matrix_descr(Z_d);

    starneig_unregister_vector_descr(real_d);
    starneig_unregister_vector_descr(imag_d);
    starneig_unregister_vector_descr(beta_d);

    starneig_free_vector_descr(real_d);
    starneig_free_vector_descr(imag_d);
    starneig_free_vector_descr(beta_d);

    return err;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Schur_expert(
    struct starneig_schur_conf *conf,
    starneig_distr_matrix_t H,
    starneig_distr_matrix_t Q,
    double real[], double imag[])
{
    if (H == NULL)      return -2;
    if (Q == NULL)      return -3;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    mpi_info_t mpi = starneig_mpi_get_info();

    starneig_error_t ret = schur_mpi(
        conf, Q, NULL, H, NULL, real, imag, NULL, mpi);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Schur(
    starneig_distr_matrix_t H,
    starneig_distr_matrix_t Q,
    double real[], double imag[])
{
    if (H == NULL)      return -1;
    if (Q == NULL)      return -2;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    return starneig_SEP_DM_Schur_expert(NULL, H, Q, real, imag);
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_DM_Schur_expert(
    struct starneig_schur_conf *conf,
    starneig_distr_matrix_t H,
    starneig_distr_matrix_t T,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[])
{
    if (H == NULL)      return -2;
    if (T == NULL)      return -3;
    if (Q == NULL)      return -4;
    if (Z == NULL)      return -5;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    mpi_info_t mpi = starneig_mpi_get_info();

    starneig_error_t ret = schur_mpi(
        conf, Q, Z, H, T, real, imag, beta, mpi);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_DM_Schur(
    starneig_distr_matrix_t H,
    starneig_distr_matrix_t T,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[])
{
    if (H == NULL)      return -1;
    if (T == NULL)      return -2;
    if (Q == NULL)      return -3;
    if (Z == NULL)      return -4;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    return starneig_GEP_DM_Schur_expert(NULL, H, T, Q, Z, real, imag, beta);
}
