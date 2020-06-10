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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "process_args.h"
#include "tasks.h"
#include "../common/utils.h"
#include <math.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starpu_mpi.h>
#endif

int starneig_schur_get_optimal_tile_size(int n, int workers)
{
/*
Fitting Function: ax + b
initial a= 0
initial b= 0.005
________ Input Data (Fit values from inital guesses)________
   x 	             y 	            y(fit) 	 residual
 3.0    	   0.01206	   0.005  	 0.00706
 7.0    	   0.01136	   0.005  	 0.00636
11.0    	   0.01087	   0.005  	 0.00587
19.0    	   0.01018	   0.005  	 0.00518
27.0    	   0.00838	   0.005  	 0.00338
======================   Results   =========================
a= -0.0001437 +- 0.0000148
b= 0.012497 +- 0.00024
______________________ Output Data _________________________
   x 	             y 	            y(fit) 	 residual
 3.0    	   0.01206	   0.01207	-0.0
 7.0    	   0.01136	   0.01149	-0.00013
11.0    	   0.01087	   0.01092	-0.00004
19.0    	   0.01018	   0.00977	 0.00041
27.0    	   0.00838	   0.00862	-0.00024
------------------------------------------------------------
sum of squared residuals= 2.423e-7
stand. dev. y values= 0.0002842
correlation between a & b= -0.8411

Fitting Function: ax + b
initial a= 7.5
initial b= 0.4
________ Input Data (Fit values from inital guesses)________
   x 	             y 	            y(fit) 	 residual
 1.0    	  62.16729	   7.9    	54.26729
 3.0    	 119.3067 	  22.9    	96.4067
 7.0    	 102.04446	  52.9    	49.14446
11.0    	  88.59951	  82.9    	 5.69951
19.0    	 105.31472	 142.9    	-37.58528
27.0    	  41.56999	 202.9    	-161.33001
======================   Results   =========================
a= -1.3 +- 1.31
b= 101.2 +- 19
______________________ Output Data _________________________
   x 	             y 	            y(fit) 	 residual
 1.0    	  62.16729	  99.94483	-37.77754
 3.0    	 119.3067 	  97.34269	21.964
 7.0    	 102.04446	  92.13841	 9.90604
11.0    	  88.59951	  86.93413	 1.66537
19.0    	 105.31472	  76.52557	28.78914
27.0    	  41.56999	  66.11701	-24.54702
------------------------------------------------------------
sum of squared residuals= 3442
stand. dev. y values= 29.33
correlation between a & b= -0.779



    double A = -0.0001437 * workers + 0.012497;
    double B = -1.3 * workers + 101.2;

    return MAX(32, divceil(A * n + B, 8)*8);
*/

    return MAX(32, divceil(0.02*n, 8)*8);
}

int starneig_get_optimal_aed_size(int n, int workers)
{
    // these are from LAPACK
    int min_val;
    if (n < 30)
        min_val = 2;
    else if (n < 60)
        min_val = 4;
    else if (n < 150)
        min_val = 10;
    else if (n < 590) {
        double x = (n-150)/(590-150);
        min_val = (1-x)*10 + x*64;
    }
    else if (n < 3000)
        min_val = 64;
    else if (n < 6000)
        min_val = 128;
    else
        min_val = 256;

    return MAX(min_val/0.7, 0.08*n);
}

int starneig_get_optimal_shift_count(int n, int workers)
{
    // these are from LAPACK
    int min_val;
    if (n < 30)
        min_val = 2;
    else if (n < 60)
        min_val = 4;
    else if (n < 150)
        min_val = 10;
    else if (n < 590) {
        double x = (n-150)/(590-150);
        min_val = (1-x)*10 + x*64;
    }
    else if (n < 3000)
        min_val = 64;
    else if (n < 6000)
        min_val = 128;
    else
        min_val = 256;

    return MAX(min_val, 0.06*n);
}

starneig_error_t starneig_build_process_args_from(
    struct process_args const *source,
    const starneig_matrix_descr_t matrix_q,
    const starneig_matrix_descr_t matrix_z,
    const starneig_matrix_descr_t matrix_a,
    const starneig_matrix_descr_t matrix_b,
    struct process_args *args)
{
    args->mpi = source->mpi;

    args->max_prio = source->max_prio;
    args->min_prio = source->min_prio;
    args->default_prio = source->default_prio;

    args->iteration_limit = source->iteration_limit;

    args->small_limit =
        (parameter_t) { .alpha = 0.0, .beta =
            MAX(300, 2*STARNEIG_MATRIX_BM(matrix_a)) };

    args->aed_window_size = (parameter_t) {
        .alpha = 0.0,
        .beta =
            starneig_get_optimal_aed_size(STARNEIG_MATRIX_M(matrix_a)-1, 6)
    };

    args->shift_count = (parameter_t) {
        .alpha = 0.0,
        .beta =
            starneig_get_optimal_shift_count(STARNEIG_MATRIX_M(matrix_a)-1, 6)
    };

    args->aed_nibble = source->aed_nibble;

    args->aed_parallel_soft_limit = source->aed_parallel_soft_limit;
    args->aed_parallel_hard_limit = source->aed_parallel_hard_limit;

    // bulges_window_placement is set to BULGES_WINDOW_PLACEMENT_ROUNDED
    // so bulges_window_size and bulges_shifts_per_window do not have an effect.
    // However, it is probably a  good idea to set them to some reasonable
    // values.
    args->bulges_window_size = (parameter_t)
        { .alpha = 0.0, .beta = 2*STARNEIG_MATRIX_BM(matrix_a) };
    args->bulges_shifts_per_window = (parameter_t)
        { .alpha = 0.0, .beta = args->bulges_window_size.beta/3-2 };

    args->bulges_window_placement =  BULGES_WINDOW_PLACEMENT_ROUNDED;

    args->a_width = 0;
    if (matrix_a != NULL)
        args->a_width = 6*STARNEIG_MATRIX_BN(matrix_a);

    args->a_height = 0;
    if (matrix_a != NULL)
        args->a_height = 6*STARNEIG_MATRIX_BM(matrix_a);

    args->b_width = 0;
    if (matrix_b != NULL)
        args->b_width = 6*STARNEIG_MATRIX_BN(matrix_b);

    args->b_height = 0;
    if (matrix_b != NULL)
        args->b_height = 6*STARNEIG_MATRIX_BM(matrix_b);

    args->q_height = 0;
    if (matrix_q != NULL)
        args->q_height = 6*STARNEIG_MATRIX_BM(matrix_q);

    args->z_height = 0;
    if (matrix_z != NULL)
        args->z_height = 6*STARNEIG_MATRIX_BM(matrix_z);

    args->matrix_a = matrix_a;
    args->matrix_b = matrix_b;
    args->matrix_q = matrix_q;
    args->matrix_z = matrix_z;

    args->thres_a = source->thres_a;
    args->thres_b = source->thres_b;
    args->thres_inf = source->thres_inf;

    return STARNEIG_SUCCESS;
}

starneig_error_t starneig_build_process_args(
    struct starneig_schur_conf const *conf,
    const starneig_matrix_descr_t matrix_q,
    const starneig_matrix_descr_t matrix_z,
    const starneig_matrix_descr_t matrix_a,
    const starneig_matrix_descr_t matrix_b,
    double thres_a, double thres_b, double thres_inf,
    mpi_info_t mpi, struct process_args *args)
{
    int n = STARNEIG_MATRIX_N(matrix_a);

    int world_size = starneig_mpi_get_comm_size();
    int worker_count = starpu_cpu_worker_get_count();

    args->mpi = mpi;

    args->max_prio = STARPU_MAX_PRIO;
    args->min_prio = STARPU_MIN_PRIO;
    args->default_prio = STARPU_DEFAULT_PRIO;

    // iteration limit
    if (conf == NULL ||
    conf->iteration_limit == STARNEIG_SCHUR_DEFAULT_INTERATION_LIMIT) {
        args->iteration_limit = 300;
    }
    else {
        if (0 < conf->iteration_limit) {
            args->iteration_limit = conf->iteration_limit;
        }
        else {
            starneig_error("Invalid iteration count limit. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    // other limits

    if (conf->small_limit == STARNEIG_SCHUR_DEFAULT_SMALL_LIMIT) {
        args->small_limit = (parameter_t) { .alpha = 0.0, .beta =
            MAX(300, 2*STARNEIG_MATRIX_BM(matrix_a)) };
    }
    else {
        if (2 < conf->small_limit) {
            args->small_limit =
                (parameter_t) { .alpha = 0.0, .beta = conf->small_limit };
        }
        else {
            starneig_error("Invalid small limit. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    if (conf == NULL ||
    (conf->aed_window_size == STARNEIG_SCHUR_DEFAULT_AED_WINDOW_SIZE &&
    conf->shift_count == STARNEIG_SCHUR_DEFAULT_SHIFT_COUNT)) {
        args->aed_window_size = (parameter_t) {
            .alpha = 0.0,
            .beta = starneig_get_optimal_aed_size(
                n, starpu_worker_get_count())
        };
        args->shift_count = (parameter_t) {
            .alpha = 0.0,
            .beta = starneig_get_optimal_shift_count(
                n, starpu_worker_get_count())
        };
    }
    else if (conf->aed_window_size ==
    STARNEIG_SCHUR_DEFAULT_AED_WINDOW_SIZE) {
        if (2 <= conf->shift_count) {
            args->aed_window_size = (parameter_t)
                { .alpha = 0.0, .beta = 2*conf->shift_count };
            args->shift_count = (parameter_t)
                { .alpha = 0.0, .beta = conf->shift_count };
        }
        else {
            starneig_error("Invalid number of AED shifts. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }
    else if (conf->shift_count ==
    STARNEIG_SCHUR_DEFAULT_SHIFT_COUNT) {
        if (4 < conf->aed_window_size) {
            args->aed_window_size = (parameter_t)
                { .alpha = 0.0, .beta = conf->aed_window_size };
            args->shift_count = (parameter_t)
                { .alpha = 0.0, .beta = conf->aed_window_size/2 };
        }
        else {
            starneig_error("Invalid AED window size. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }
    else {
        if (conf->shift_count <= conf->aed_window_size) {
             args->aed_window_size = (parameter_t)
                { .alpha = 0.0, .beta = conf->aed_window_size };
            args->shift_count = (parameter_t)
                { .alpha = 0.0, .beta = MIN(
                    9*conf->aed_window_size/10, conf->shift_count) };
        }
        else {
            starneig_error(
                "Invalid AED window size or shift count. Exiting...");
                return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    if (conf->aed_nibble == STARNEIG_SCHUR_DEFAULT_AED_NIBBLE) {
        args->aed_nibble = (parameter_t) { .alpha = 0.0, .beta = 40 };
    }
    else {
        if (0 < conf->aed_nibble && conf->aed_nibble < 100) {
            args->aed_nibble =
                (parameter_t) { .alpha = 0.0, .beta = conf->aed_nibble };
        }
        else {
            starneig_error("Invalid nibble point. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    if (conf->aed_parallel_soft_limit ==
    STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_SOFT_LIMIT) {
        args->aed_parallel_soft_limit =
            (parameter_t) { .alpha = 0.0, .beta = 600 };
    }
    else {
        if (0 < conf->aed_parallel_soft_limit) {
            args->aed_parallel_soft_limit = (parameter_t)
                { .alpha = 0.0, .beta = conf->aed_parallel_soft_limit };
        }
        else {
            starneig_error("Invalid soft parallel AED limit. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    if (conf->aed_parallel_hard_limit ==
    STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_HARD_LIMIT) {
        args->aed_parallel_hard_limit =
            (parameter_t) { .alpha = 0.0, .beta = 300 };
    }
    else {
        if (0 < conf->aed_parallel_hard_limit) {
            args->aed_parallel_hard_limit = (parameter_t)
                { .alpha = 0.0, .beta = conf->aed_parallel_hard_limit };
        }
        else {
            starneig_error("Invalid hard parallel AED limit. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    if (conf == NULL || conf->window_size ==
    STARNEIG_SCHUR_DEFAULT_WINDOW_SIZE ||
    conf->window_size == STARNEIG_SCHUR_ROUNDED_WINDOW_SIZE) {
        args->bulges_window_size = (parameter_t)
            { .alpha = 0.0, .beta = 2*STARNEIG_MATRIX_BM(matrix_a) };
        args->bulges_window_placement= BULGES_WINDOW_PLACEMENT_ROUNDED;
    }
    else {
        if (5 <= conf->window_size) {
            args->bulges_window_size = (parameter_t)
                { .alpha = 0.0, .beta = conf->window_size };
        }
        else {
            starneig_error("Invalid window size. Exiting...");
            return STARNEIG_INVALID_ARGUMENTS;
        }
        args->bulges_window_placement = BULGES_WINDOW_PLACEMENT_FIXED;
    }

    // shifts per window
    if (conf == NULL ||
    conf->shifts_per_window == STARNEIG_SCHUR_DEFAULT_SHIFTS_PER_WINDOW) {
        args->bulges_shifts_per_window = (parameter_t)
            { .alpha = 0.0, .beta = args->bulges_window_size.beta/3-2 };
    }
    else {
        if (2 <= conf->shifts_per_window) {
            args->bulges_shifts_per_window = (parameter_t)
                { .alpha = 0.0,
                    .beta = MIN(args->bulges_window_size.beta/3-2,
                        conf->shifts_per_window) };
        }
        else {
            starneig_error(
                "Invalid number of shifts per window. Exiting...");
                return STARNEIG_INVALID_ARGUMENTS;
        }
    }

    #define set_width(x_width, matrix_x) \
        args->x_width = 0; \
        if (matrix_x != NULL) \
            args->x_width = starneig_calc_update_size( \
                STARNEIG_MATRIX_N(matrix_x), STARNEIG_MATRIX_BN(matrix_x), \
                STARNEIG_MATRIX_SBN(matrix_x), world_size, worker_count)

    #define set_width_warn(X, x_width, matrix_x) \
        set_width(x_width, matrix_x); \
        starneig_warning( \
            "Invalid matrix " #X " update width. Using %d.", \
            args->x_width)

    // update task width
    if (conf == NULL ||
    conf->update_width == STARNEIG_SCHUR_DEFAULT_UPDATE_WIDTH) {
        set_width(a_width, matrix_a);
        set_width(b_width, matrix_b);
    }
    else {
        if (0 < conf->update_width) {
            args->a_width = conf->update_width;
            args->b_width = conf->update_width;
        }
        else {
            set_width_warn(A, a_width, matrix_a);
            set_width_warn(B, b_width, matrix_b);
        }
    }

    #undef set_width
    #undef set_width_warn

    #define set_height(x_height, matrix_x) \
        args->x_height = 0; \
        if (matrix_x != NULL) \
            args->x_height = starneig_calc_update_size( \
                STARNEIG_MATRIX_M(matrix_x), STARNEIG_MATRIX_BM(matrix_x), \
                STARNEIG_MATRIX_SBM(matrix_x), world_size, worker_count)

    #define set_height_warn(X, x_height, matrix_x) \
        set_height(x_height, matrix_x); \
        starneig_warning( \
            "Invalid matrix " #X " update height. Using %d.", \
            args->x_height)

    // update task height
    if (conf == NULL ||
    conf->update_height == STARNEIG_SCHUR_DEFAULT_UPDATE_HEIGHT) {
        set_height(a_height, matrix_a);
        set_height(b_height, matrix_b);
        set_height(q_height, matrix_q);
        set_height(z_height, matrix_z);
    }
    else {
        if (0 < conf->update_height) {
            args->a_height = conf->update_height;
            args->b_height = conf->update_height;
            args->q_height = conf->update_height;
            args->z_height = conf->update_height;
        }
        else {
            set_height_warn(A, a_height, matrix_a);
            set_height_warn(B, b_height, matrix_b);
            set_height_warn(Q, q_height, matrix_q);
            set_height_warn(Z, z_height, matrix_z);
        }
    }

    #undef set_height
    #undef set_height_warn

    args->matrix_a = matrix_a;
    args->matrix_b = matrix_b;
    args->matrix_q = matrix_q;
    args->matrix_z = matrix_z;

    args->thres_a = thres_a;
    args->thres_b = thres_b;
    args->thres_inf = thres_inf;

    return STARNEIG_SUCCESS;
}
