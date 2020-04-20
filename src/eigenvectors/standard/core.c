///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
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
#include "../../common/tasks.h"
#include "core.h"
#include "cpu.h"

// TODO: move codelets
static struct starpu_codelet bound_cl = {
    .name = "bound",
    .cpu_funcs = {starneig_eigvec_std_cpu_bound},
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W}
};

static struct starpu_codelet backsolve_cl = {
    .name = "backsolve",
    .cpu_funcs = {starneig_eigvec_std_cpu_backsolve},
    .nbuffers = 8,
    .modes = {STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W,
              STARPU_R, STARPU_R, STARPU_W}
};

static struct starpu_codelet solve_cl = {
    .name = "solve",
    .cpu_funcs = {starneig_eigvec_std_cpu_solve},
    .nbuffers = 10,
    .dyn_modes = (enum starpu_data_access_mode[])
    { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW, STARPU_RW,
      STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_W }
};

static struct starpu_codelet update_cl = {
    .name = "update",
    .cpu_funcs = {starneig_eigvec_std_cpu_update},
    .nbuffers = 9,
    .dyn_modes = (enum starpu_data_access_mode[])
    { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
      STARPU_RW, STARPU_RW, STARPU_RW, STARPU_R }
};

static struct starpu_codelet backtransform_cl = {
    .name = "backtransform",
    .cpu_funcs = {starneig_eigvec_std_cpu_backtransform},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W}
};





starneig_error_t starneig_eigvec_std_insert_backsolve_tasks(
    int num_tiles,
    starpu_data_handle_t **S_tiles,
    starpu_data_handle_t **S_tiles_norms,
    starpu_data_handle_t *lambda_tiles,
    starpu_data_handle_t *lambda_type_tiles,
    starpu_data_handle_t **X_tiles,
    starpu_data_handle_t **scales_tiles,
    starpu_data_handle_t **Xnorms_tiles,
    starpu_data_handle_t *selected_tiles,
    starpu_data_handle_t *selected_lambda_type_tiles,
    starpu_data_handle_t *info_tiles,
    double smlnum,
    int critical_prio, int update_prio)
{
    //
    // compute the column majorants
    //

    for (int i = 0; i < num_tiles; i++) {
        starpu_task_insert(
            &bound_cl,
            STARPU_PRIORITY, critical_prio,
            STARPU_R, S_tiles[i][i],
            STARPU_W, S_tiles_norms[i][i], 0);
        for (int j = i+1; j < num_tiles; j++) {
            starpu_task_insert(
                &bound_cl,
                STARPU_PRIORITY, critical_prio,
                STARPU_R, S_tiles[i][j],
                STARPU_W, S_tiles_norms[i][j], 0);

            starneig_insert_set_to_zero(critical_prio, X_tiles[i][j]);

        }
    }

    //
    // Compute the eigenvectors of S.
    //

    for (int k = num_tiles - 1; k >= 0; k--) {
        for (int j = k; j >= 0; j--) {
            if (k == j) {
                // Form initial right-hand sides and backsolve.
                starpu_task_insert(
                    &backsolve_cl,
                    STARPU_PRIORITY, critical_prio,
                    STARPU_R, S_tiles[k][k],
                    STARPU_R, S_tiles_norms[k][k],
                    STARPU_W, X_tiles[k][k],
                    STARPU_W, scales_tiles[k][k],
                    STARPU_W, Xnorms_tiles[k][k],
                    STARPU_R, lambda_type_tiles[k],
                    STARPU_R, selected_tiles[k],
                    STARPU_W, info_tiles[k],
                    STARPU_VALUE, &smlnum, sizeof(double), 0);
            }
            else { // k != j
                // Multi-shift solve.
                starpu_task_insert(
                    &solve_cl,
                    STARPU_PRIORITY, critical_prio,
                    STARPU_R, S_tiles[j][j],
                    STARPU_R, S_tiles_norms[j][j],
                    STARPU_RW, X_tiles[j][k],
                    STARPU_RW, scales_tiles[j][k],
                    STARPU_RW, Xnorms_tiles[j][k],
                    STARPU_R, lambda_tiles[k],
                    STARPU_R, lambda_type_tiles[k],
                    STARPU_R, selected_tiles[k],
                    STARPU_R, lambda_type_tiles[j],
                    STARPU_W, info_tiles[k],
                    STARPU_VALUE, &smlnum, sizeof(double), 0);
            }

            for (int i = j-1; i >= 0; i--) {
                // Linear update.
                starpu_task_insert(
                    &update_cl,
                    STARPU_PRIORITY, update_prio,
                    STARPU_R, S_tiles[i][j],
                    STARPU_R, S_tiles_norms[i][j],
                    STARPU_R, X_tiles[j][k],
                    STARPU_R, scales_tiles[j][k],
                    STARPU_R, Xnorms_tiles[j][k],
                    STARPU_RW, X_tiles[i][k],
                    STARPU_RW, scales_tiles[i][k],
                    STARPU_RW, Xnorms_tiles[i][k],
                    STARPU_R, selected_lambda_type_tiles[k], 0);
            }
        }
    }

    return STARNEIG_SUCCESS;
}


starneig_error_t starneig_eigvec_std_insert_backtransform_tasks(
    int *first_row, int num_tiles,
    starpu_data_handle_t **Q_tiles,
    starpu_data_handle_t **X_tiles,
    starpu_data_handle_t **Y_tiles)
{
    for (int j = num_tiles-1; j >= 0; j--) {
        for (int i = num_tiles-1; i >= 0; i--) {
            int num_inner = first_row[j+1]-first_row[0];
            starpu_task_insert(
                &backtransform_cl,
                STARPU_R, Q_tiles[i][0],
                STARPU_R, X_tiles[0][j],
                STARPU_W, Y_tiles[i][j],
                STARPU_VALUE, &num_inner, sizeof(int),
                0);
        }
    }

    return STARNEIG_SUCCESS;
}
