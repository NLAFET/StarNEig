///
/// @file
///
/// @brief This file contains code which is related segment processing
/// arguments.
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

#ifndef STARNEIG_SCHUR_PROCESS_ARGS_H
#define STARNEIG_SCHUR_PROCESS_ARGS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "../common/common.h"
#include "../common/vector.h"
#include "../common/matrix.h"
#include <starneig/expert.h>
#include <starneig/error.h>
#include <starpu.h>

///
/// @brief Linear parameter function.
///
///  f(x) = alpha * x + beta
///
typedef struct {
    double alpha;   ///< alpha
    double beta;    ///< beta
} parameter_t;

///
/// @brief Evaluates a linear parameter function.
///
/// @brief[in] x
///         evaluation point
///
/// @brief[in] parameter
///         linear parameter function
///
/// @return parameter function value at the evaluation point
///
static inline double evaluate_parameter(double x, parameter_t parameter)
{
     return parameter.alpha * x + parameter.beta;
}

///
/// @brief Segment processing arguments.
///
struct process_args {
    mpi_info_t mpi;                       ///< MPI info
    int min_prio;                         ///< minimum priority for other_ctx
    int max_prio;                         ///< maximum priority for other_ctx
    int default_prio;                     ///< default priority for other_ctx
    int iteration_limit;                  ///< iteration limit
    parameter_t small_limit;              ///< small QR limit
    parameter_t aed_window_size;          ///< AED window size
    parameter_t shift_count;          ///< AED shift count
    parameter_t aed_nibble;               ///< nibble point
    parameter_t aed_parallel_soft_limit;  ///< soft parallel AED limit
    parameter_t aed_parallel_hard_limit;  ///< soft parallel AED limit
    parameter_t bulges_window_size;       ///< bulge chasing window size
    parameter_t bulges_shifts_per_window; ///< bulge chasing shifts per window
    enum {
        BULGES_WINDOW_PLACEMENT_FIXED,    ///< fixed window size
        BULGES_WINDOW_PLACEMENT_ROUNDED   ///< rounded to the nearest tile
    } bulges_window_placement;            ///< bulge chasing window placement
    int q_height;                         ///< height of a Q matrix update task
    int z_height;                         ///< height of a Z matrix update task
    int a_width;                          ///< width of an A matrix update task
    int a_height;                         ///< height of an A matrix update task
    int b_width;                          ///< width of an B matrix update task
    int b_height;                         ///< height of an B matrix update task
    starneig_matrix_descr_t matrix_a;     ///< matrix A descriptor
    starneig_matrix_descr_t matrix_b;     ///< matrix B descriptor
    starneig_matrix_descr_t matrix_q;     ///< matrix Q descriptor
    starneig_matrix_descr_t matrix_z;     ///< matrix Z descriptor
    double thres_a;                       ///< threshold for matrix A
    double thres_b;                       ///< threshold for off-diagonal
                                          ///< entries of matrix B
    double thres_inf;                     ///< threshold for diagonal entries
                                          ///< of matrix B
};

///
/// @brief Returns "optimal" tile size for given problem size.
///
/// @param[in] n
///         matrix dimension
///
/// @param[in] workers
///         number of workers
///
/// @return optimal tile size
///
int starneig_schur_get_optimal_tile_size(int n, int workers);

///
/// @brief Returns "optimal" AED window size for given problem size.
///
/// @param[in] n
///         matrix dimension
///
/// @param[in] workers
///         number of workers
///
/// @return optimal AED window size
///
int starneig_get_optimal_aed_size(int n, int workers);

///
/// @brief Returns "optimal" shift count for given problem size.
///
/// @param[in] n
///         matrix dimension
///
/// @param[in] workers
///         number of workers
///
/// @return optimal shift count
///
int starneig_get_optimal_shift_count(int n, int workers);

///
/// @brief Builds a segment processing argument structure from an existing
/// segment processing argument structure.
///
/// @param[in] source
///         source segment processing argument structure
///
/// @param[in] matrix_q
///         matrix Q descriptor
///
/// @param[in] matrix_z
///         matrix Z descriptor
///
/// @param[in] matrix_a
///         matrix A descriptor
///
/// @param[in] matrix_q
///         matrix Q descriptor
///
/// @param[out] args
///         segment processing argument structure
///
/// @return error code
///
starneig_error_t starneig_build_process_args_from(
    struct process_args const *source,
    const starneig_matrix_descr_t matrix_q,
    const starneig_matrix_descr_t matrix_z,
    const starneig_matrix_descr_t matrix_a,
    const starneig_matrix_descr_t matrix_b,
    struct process_args *args);

///
/// @brief Builds a segment processing argument structure.
///
/// @param[in] conf
///         The configuration structure.
///
/// @param[in] matrix_q
///         The matrix Q descriptor.
///
/// @param[in] matrix_z
///         The matrix Z descriptor.
///
/// @param[in] matrix_a
///         The matrix A descriptor.
///
/// @param[in] matrix_b
///         The matrix B descriptor.
///
/// @param[in] thres_a
///         Those entries of the matrix A that are smaller in magnitudes than
///         this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] mpi
///         The MPI info.
///
/// @param[out] args
///         The segment processing argument structure.
///
/// @return Error code.
///
starneig_error_t starneig_build_process_args(
    struct starneig_schur_conf const *conf,
    const starneig_matrix_descr_t matrix_q,
    const starneig_matrix_descr_t matrix_z,
    const starneig_matrix_descr_t matrix_a,
    const starneig_matrix_descr_t matrix_b,
    double thres_a, double thres_b, double thres_inf,
    mpi_info_t mpi, struct process_args *args);

#endif
