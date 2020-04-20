///
/// @file
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
#include "utils.h"
#include "node_internal.h"
#include "distr_matrix_internal.h"
#include "../common/utils.h"
#include "../common/tasks.h"
#include "../common/node_internal.h"
#include <starpu_mpi.h>
#include <stddef.h>
#include <math.h>

#define SELECT(val,i) { if (0 <= (i) && (i) < size) selected[i] = val; }

typedef int (*sep_func_t)(double real, double imag, void *arg);

struct sep_args {
    sep_func_t predicate;
    void *arg;
};

static void apply_predicate_sep(
    int size, int rbegin, int cbegin, int m, int n, int ldS, int ldT,
    void const *_arg, void const *_S, void const *_T, void **masks)
{
    struct sep_args const *predicate_arg = _arg;
    sep_func_t predicate = predicate_arg->predicate;
    void *arg = predicate_arg->arg;

    double const *S = _S;
    int *selected = masks[0];

    int begin = 0 < rbegin && 0 < cbegin ? -1 : 0;
    int end = rbegin+size < m && cbegin+size < n ? size+1 : size;

    for (int i = begin; i < end; i++) {

        //
        // 2-by-2 block
        //
        if (i+1 < m && i+1 < n && S[(size_t)(cbegin+i)*ldS+rbegin+i+1] != 0.0) {
            double real1, imag1, real2, imag2;
            starneig_compute_complex_eigenvalue(
                ldS, 0, &S[(cbegin+i)*ldS + rbegin+i], NULL,
                &real1, &imag1, &real2, &imag2,
                NULL, NULL);

            if (predicate(real1, imag1, arg)) {
                SELECT(1, i);
                SELECT(1, i+1);
            }
            else {
                SELECT(0, i);
                SELECT(0, i+1);
            }

            i++;
        }

        //
        // 1-by-1 block
        //
        else {
            double __S = S[(size_t)(cbegin+i)*ldS+rbegin+i];
            if (predicate(__S, 0.0, arg)) {
                SELECT(1, i);
            }
            else {
                SELECT(0, i);
            }
        }
    }
}

typedef int (*gep_func_t)(double real, double imag, double beta, void *arg);

struct gep_args {
    gep_func_t predicate;
    void *arg;
};

static void apply_predicate_gep(
    int size, int rbegin, int cbegin, int m, int n, int ldS, int ldT,
    void const *_arg, void const *_S, void const *_T, void **masks)
{
    struct gep_args const *predicate_arg = _arg;
    gep_func_t predicate = predicate_arg->predicate;
    void *arg = predicate_arg->arg;

    double const *S = _S;
    double const *T = _T;
    int *selected = masks[0];

    int begin = 0 < rbegin && 0 < cbegin ? -1 : 0;
    int end = rbegin+size < m && cbegin+size < n ? size+1 : size;

    for (int i = begin; i < end; i++) {

        //
        // 2-by-2 block
        //
        if (i+1 < m && i+1 < n && S[(size_t)(cbegin+i)*ldS+rbegin+i+1] != 0.0) {
            double real1, imag1, real2, imag2, beta1, beta2;
            starneig_compute_complex_eigenvalue(ldS, ldT,
                &S[(cbegin+i)*ldS + rbegin+i], &T[(cbegin+i)*ldT + rbegin+i],
                &real1, &imag1, &real2, &imag2, &beta1, &beta2);

            if (predicate(real1, imag1, beta1, arg)) {
                SELECT(1, i);
                SELECT(1, i+1);
            }
            else {
                SELECT(0, i);
                SELECT(0, i+1);
            }

            i++;
        }

        //
        // 1-by-1 block
        //
        else {
            double __S = S[(size_t)(cbegin+i)*ldS+rbegin+i];
            double __T = T[(size_t)(cbegin+i)*ldT+rbegin+i];
            if (predicate(__S, 0.0, __T, arg)) {
                SELECT(1, i);
            }
            else {
                SELECT(0, i);
            }
        }
    }
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Select(
    starneig_distr_matrix_t S,
    int (*predicate)(double real, double imag, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (S == NULL)          return -1;
    if (predicate == NULL)  return -2;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    int m = starneig_distr_matrix_get_rows(S);

    mpi_info_t mpi = starneig_mpi_get_info();

    int tile_size = starneig_mpi_find_valid_tile_size(128, S, NULL, NULL, NULL);
    starneig_matrix_descr_t S_d = starneig_mpi_cache_convert_and_release(
        tile_size, tile_size, MATRIX_TYPE_UPPER_HESSENBERG, S, mpi);

    starneig_vector_descr_t selected_d = starneig_init_matching_vector_descr(
        S_d, sizeof(int), selected, mpi);

    struct sep_args args = {
        .predicate = predicate,
        .arg = arg,
    };

    starneig_insert_scan_diagonal(
        0, m, 0, 1, 1, 1, 1, STARPU_MAX_PRIO, apply_predicate_sep,
        &args, S_d, NULL, mpi, selected_d, NULL);

    int world_size = starneig_mpi_get_comm_size();
    for (int i = 0; i < world_size; i++)
        starneig_gather_vector_descr(i, selected_d);

    starneig_acquire_matrix_descr(S_d);

    starneig_unregister_vector_descr(selected_d);
    starneig_free_vector_descr(selected_d);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    if (num_selected != NULL) {
        *num_selected = 0;
        for (int i = 0; i < m; i++)
            if (selected[i]) (*num_selected)++;
    }

    return STARNEIG_SUCCESS;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_DM_Select(
    starneig_distr_matrix_t S,
    starneig_distr_matrix_t T,
    int (*predicate)(double real, double imag, double beta, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (S == NULL)          return -1;
    if (T == NULL)          return -2;
    if (predicate == NULL)  return -3;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    int m = starneig_distr_matrix_get_rows(S);

    mpi_info_t mpi = starneig_mpi_get_info();

    int tile_size = starneig_mpi_find_valid_tile_size(128, S, T, NULL, NULL);

    starneig_matrix_descr_t S_d = starneig_mpi_cache_convert_and_release(
        tile_size, tile_size, MATRIX_TYPE_UPPER_HESSENBERG, S, mpi);

    starneig_matrix_descr_t T_d = starneig_mpi_cache_convert_and_release(
        tile_size, tile_size, MATRIX_TYPE_UPPER_TRIANGULAR, T, mpi);

    starneig_vector_descr_t selected_d = starneig_init_matching_vector_descr(
        S_d, sizeof(int), selected, mpi);

    struct gep_args args = {
        .predicate = predicate,
        .arg = arg,
    };

    starneig_insert_scan_diagonal(
        0, m, 0, 1, 1, 1, 1, STARPU_MAX_PRIO, apply_predicate_gep,
        &args, S_d, T_d, mpi, selected_d, NULL);

    int world_size = starneig_mpi_get_comm_size();
    for (int i = 0; i < world_size; i++)
        starneig_gather_vector_descr(i, selected_d);

    starneig_acquire_matrix_descr(S_d);
    starneig_acquire_matrix_descr(T_d);

    starneig_unregister_vector_descr(selected_d);
    starneig_free_vector_descr(selected_d);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    if (num_selected != NULL) {
        *num_selected = 0;
        for (int i = 0; i < m; i++)
            if (selected[i]) (*num_selected)++;
    }

    return STARNEIG_SUCCESS;
}
