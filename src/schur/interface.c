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
#include "core.h"
#include "process_args.h"
#include "../common/utils.h"
#include "../common/node_internal.h"
#include "../common/trace.h"
#include <starneig/sep_sm.h>
#include <starneig/gep_sm.h>

static starneig_error_t schur(
    struct starneig_schur_conf const *_conf, int n, int ldQ, int ldZ, int ldA,
    int ldB, double *Q, double *Z, double *A, double *B, double *real,
    double *imag, double *beta)
{
    // use default configuration if necessary
    struct starneig_schur_conf *conf;
    struct starneig_schur_conf local_conf;
    if (_conf == NULL)
        starneig_schur_init_conf(&local_conf);
    else
        local_conf = *_conf;
    conf = &local_conf;

    starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

    //
    // select tile size
    //

    if (conf->tile_size == STARNEIG_SCHUR_DEFAULT_TILE_SIZE) {
        conf->tile_size =
            starneig_schur_get_optimal_tile_size(n, starpu_worker_get_count());
        starneig_message("Setting tile size to %d.", conf->tile_size);
    }

    //
    // register, partition and pack
    //

    starneig_matrix_descr_t A_d = starneig_register_matrix_descr(
        MATRIX_TYPE_FULL, n, n,
        conf->tile_size, conf->tile_size, -1, -1, ldA, sizeof(double),
        NULL, NULL, A, NULL);
    STARNEIG_EVENT_SET_LABEL(A_d, 'A');

    starneig_matrix_descr_t B_d = NULL;
    if (B != NULL) {
        B_d = starneig_register_matrix_descr(
            MATRIX_TYPE_FULL, n, n,
            conf->tile_size, conf->tile_size, -1, -1, ldB, sizeof(double),
            NULL, NULL, B, NULL);
        STARNEIG_EVENT_SET_LABEL(B_d, 'B');
    }

    starneig_matrix_descr_t Q_d = NULL;
    if (Q != NULL) {
        Q_d = starneig_register_matrix_descr(
            MATRIX_TYPE_FULL, n, n,
            conf->tile_size, conf->tile_size, -1, -1, ldQ, sizeof(double),
            NULL, NULL, Q, NULL);
        STARNEIG_EVENT_SET_LABEL(Q_d, 'Q');
    }

    starneig_matrix_descr_t Z_d = NULL;
    if (Z != NULL) {
        Z_d = starneig_register_matrix_descr(
            MATRIX_TYPE_FULL, n, n,
            conf->tile_size, conf->tile_size, -1, -1, ldZ, sizeof(double),
            NULL, NULL, Z, NULL);
        STARNEIG_EVENT_SET_LABEL(Z_d, 'Z');
    }

    starneig_vector_descr_t real_d = NULL;
    if (real != NULL)
        real_d = starneig_init_matching_vector_descr(
            A_d, sizeof(double), real, NULL);

    starneig_vector_descr_t imag_d = NULL;
    if (imag != NULL)
        imag_d = starneig_init_matching_vector_descr(
            A_d, sizeof(double), imag, NULL);

    starneig_vector_descr_t beta_d = NULL;
    if (beta != NULL)
        beta_d = starneig_init_matching_vector_descr(
            A_d, sizeof(double), beta, NULL);

    //
    // insert tasks
    //

    STARNEIG_EVENT_INIT();

    starneig_error_t ret = starneig_schur_insert_tasks(
        conf, Q_d, Z_d, A_d, B_d, real_d, imag_d, beta_d, NULL);

    //
    // finalize
    //

    starneig_unregister_matrix_descr(A_d);
    starneig_unregister_matrix_descr(B_d);
    starneig_unregister_matrix_descr(Q_d);
    starneig_unregister_matrix_descr(Z_d);
    starneig_unregister_vector_descr(real_d);
    starneig_unregister_vector_descr(imag_d);
    starneig_unregister_vector_descr(beta_d);

    starneig_free_matrix_descr(A_d);
    starneig_free_matrix_descr(B_d);
    starneig_free_matrix_descr(Q_d);
    starneig_free_matrix_descr(Z_d);
    starneig_free_vector_descr(real_d);
    starneig_free_vector_descr(imag_d);
    starneig_free_vector_descr(beta_d);

    STARNEIG_EVENT_STORE(n, "trace.dat");
    STARNEIG_EVENT_FREE();

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__attribute__ ((visibility ("default")))
void starneig_schur_init_conf(struct starneig_schur_conf *conf)
{
    CHECK_INIT();
    conf->iteration_limit = STARNEIG_SCHUR_DEFAULT_INTERATION_LIMIT;
    conf->tile_size = STARNEIG_SCHUR_DEFAULT_TILE_SIZE;
    conf->small_limit = STARNEIG_SCHUR_DEFAULT_SMALL_LIMIT;
    conf->aed_window_size = STARNEIG_SCHUR_DEFAULT_AED_WINDOW_SIZE;
    conf->aed_nibble = STARNEIG_SCHUR_DEFAULT_AED_NIBBLE;
    conf->aed_parallel_soft_limit =
        STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_SOFT_LIMIT;
    conf->aed_parallel_hard_limit =
        STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_HARD_LIMIT;
    conf->shift_count = STARNEIG_SCHUR_DEFAULT_SHIFT_COUNT;
    conf->window_size = STARNEIG_SCHUR_DEFAULT_WINDOW_SIZE;
    conf->shifts_per_window = STARNEIG_SCHUR_DEFAULT_SHIFTS_PER_WINDOW;
    conf->update_width = STARNEIG_SCHUR_DEFAULT_UPDATE_WIDTH;
    conf->update_height = STARNEIG_SCHUR_DEFAULT_UPDATE_HEIGHT;
    conf->left_threshold = STARNEIG_SCHUR_DEFAULT_THRESHOLD;
    conf->right_threshold = STARNEIG_SCHUR_DEFAULT_THRESHOLD;
    conf->inf_threshold = STARNEIG_SCHUR_DEFAULT_THRESHOLD;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_SM_Schur_expert(
    struct starneig_schur_conf *conf,
    int n,
    double H[], int ldH,
    double Q[], int ldQ,
    double real[], double imag[])
{
    if (n < 1)          return -2;
    if (H == NULL)      return -3;
    if (ldH < n)        return -4;
    if (Q == NULL)      return -5;
    if (ldQ < n)        return -6;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_SM);
    starneig_node_resume_starpu();

    starneig_error_t ret = schur(
        conf, n, ldQ, 0, ldH, 0, Q, NULL, H, NULL, real, imag, NULL);

    starpu_task_wait_for_all();
    starneig_node_pause_starpu();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_SM_Schur(
    int n,
    double H[], int ldH,
    double Q[], int ldQ,
    double real[], double imag[])
{
    if (n < 1)          return -1;
    if (H == NULL)      return -2;
    if (ldH < n)        return -3;
    if (Q == NULL)      return -4;
    if (ldQ < n)        return -5;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    return starneig_SEP_SM_Schur_expert(NULL, n, H, ldH, Q, ldQ, real, imag);
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_Schur_expert(
    struct starneig_schur_conf *conf,
    int n,
    double H[], int ldH,
    double T[], int ldT,
    double Q[], int ldQ,
    double Z[], int ldZ,
    double real[], double imag[], double beta[])
{
    if (n < 1)          return -2;
    if (H == NULL)      return -3;
    if (ldH < n)        return -4;
    if (T == NULL)      return -5;
    if (ldT < n)        return -6;
    if (Q == NULL)      return -7;
    if (ldQ < n)        return -8;
    if (Z == NULL)      return -9;
    if (ldZ < n)        return -10;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_SM);
    starneig_node_resume_starpu();

    starneig_error_t ret = schur(
        conf, n, ldQ, ldZ, ldH, ldT, Q, Z, H, T, real, imag, beta);

    starpu_task_wait_for_all();
    starneig_node_pause_starpu();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_Schur(
    int n,
    double H[], int ldH,
    double T[], int ldT,
    double Q[], int ldQ,
    double Z[], int ldZ,
    double real[], double imag[], double beta[])
{
    if (n < 1)          return -1;
    if (H == NULL)      return -2;
    if (ldH < n)        return -3;
    if (T == NULL)      return -4;
    if (ldT < n)        return -5;
    if (Q == NULL)      return -6;
    if (ldQ < n)        return -7;
    if (Z == NULL)      return -8;
    if (ldZ < n)        return -9;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    return starneig_GEP_SM_Schur_expert(
        NULL, n, H, ldH, T, ldT, Q, ldQ, Z, ldZ, real, imag, beta);
}
