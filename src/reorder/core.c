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
#include "core.h"
#include "plan.h"
#include "insert_engine.h"
#include "../common/common.h"
#include "../common/utils.h"
#include "../common/tasks.h"
#include <math.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starpu_mpi.h>
#endif

///
/// @brief Task insertion plan descriptor.
///
struct plan_descr {
    starneig_reorder_plan_t type;                     ///< plan enumerator
    char *name;                                     ///< plan name
    starneig_reorder_blueprint_t preferred_blueprint; ///< preferred blueprint
    plan_interface_t func;                          ///< interface function
};

///
/// @brief Task insert blueprint descriptor.
///
struct blueprint_descr {
    starneig_reorder_blueprint_t type;        ///< blueprint enumerator
    char *name;                             ///< blueprint name
    starneig_reorder_plan_t *valid_plans;     ///< valid plans
    starneig_reorder_plan_t preferred_plan;   ///< preferred plan
    blueprint_step_t *blueprint;            ///< blueprint
};

///
/// @brief Task insertion plan descriptors.
///
static const struct plan_descr plans[] = {
    { .type = STARNEIG_REORDER_ONE_PART_PLAN,
        .name = "one-part task insertion plan",
        .preferred_blueprint = STARNEIG_REORDER_CHAIN_INSERT_A,
        .func = &starneig_formulate_plan },
    { .type = STARNEIG_REORDER_MULTI_PART_PLAN,
        .name = "multi-part task insertion plan",
        .preferred_blueprint = STARNEIG_REORDER_DUMMY_INSERT_B,
        .func = &starneig_formulate_multiplan}
};

///
/// @brief Task insertion blueprint descriptors.
///
static const struct blueprint_descr blueprints[] = {
    { .type = STARNEIG_REORDER_DUMMY_INSERT_A,
        .name = "one-pass forward dummy blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_ONE_PART_PLAN,
            STARNEIG_REORDER_MULTI_PART_PLAN,
            0 },
        .preferred_plan = STARNEIG_REORDER_ONE_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_FORWARD,
                    WINDOW_BEGIN,
                        DUMMY_WINDOW,
                        DUMMY_RIGHT_UPDATE,
                        DUMMY_LEFT_UPDATE,
                        DUMMY_Q_UPDATE,
                        UNREGISTER,
                    WINDOW_END,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_DUMMY_INSERT_B,
        .name = "two-pass backward dummy blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_MULTI_PART_PLAN, 0 },
        .preferred_plan = STARNEIG_REORDER_MULTI_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    WINDOW_BEGIN,
                        DUMMY_WINDOW,
                        DUMMY_RIGHT_UPDATE,
                    WINDOW_END,
                CHAIN_END,
                CHAIN_BACKWARD,
                    WINDOW_BEGIN,
                        DUMMY_LEFT_UPDATE,
                        DUMMY_Q_UPDATE,
                        UNREGISTER,
                    WINDOW_END,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_CHAIN_INSERT_A,
        .name = "one-pass forward chain blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_ONE_PART_PLAN,
            STARNEIG_REORDER_MULTI_PART_PLAN,
            0 },
        .preferred_plan = STARNEIG_REORDER_ONE_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_FORWARD,
                    WINDOWS,
                    LEFT_UPDATES,
                    REMAINING_RIGHT_UPDATES,
                    Q_UPDATES,
                    UNREGISTER,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_CHAIN_INSERT_B,
        .name = "two-pass forward chain blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_ONE_PART_PLAN,
            STARNEIG_REORDER_MULTI_PART_PLAN,
            0 },
        .preferred_plan = STARNEIG_REORDER_ONE_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_FORWARD,
                    WINDOWS,
                    LEFT_UPDATES,
                CHAIN_END,
                CHAIN_FORWARD,
                    REMAINING_RIGHT_UPDATES,
                    Q_UPDATES,
                    UNREGISTER,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_CHAIN_INSERT_C,
        .name = "one-pass backward chain blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_MULTI_PART_PLAN, 0 },
        .preferred_plan = STARNEIG_REORDER_MULTI_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    WINDOWS,
                    RIGHT_UPDATES,
                    LEFT_UPDATES,
                    Q_UPDATES,
                    UNREGISTER,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_CHAIN_INSERT_D,
        .name = "two-pass backward chain blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_MULTI_PART_PLAN, 0 },
        .preferred_plan = STARNEIG_REORDER_MULTI_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    WINDOWS,
                    RIGHT_UPDATES,
                    LEFT_UPDATES,
                CHAIN_END,
                CHAIN_BACKWARD,
                    Q_UPDATES,
                    UNREGISTER,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_CHAIN_INSERT_E,
        .name = "two-pass delayed backward chain blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_MULTI_PART_PLAN, 0 },
        .preferred_plan = STARNEIG_REORDER_MULTI_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    WINDOWS,
                    RIGHT_UPDATES,
                    LEFT_UPDATES,
                CHAIN_END,
            CHAIN_LIST_END,
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    Q_UPDATES,
                    UNREGISTER,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    },
    { .type = STARNEIG_REORDER_CHAIN_INSERT_F,
        .name = "three-pass delayed backward chain blueprint",
        .valid_plans = (starneig_reorder_plan_t[]) {
            STARNEIG_REORDER_MULTI_PART_PLAN, 0 },
        .preferred_plan = STARNEIG_REORDER_MULTI_PART_PLAN,
        .blueprint = (blueprint_step_t[]) {
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    WINDOWS,
                CHAIN_END,
                CHAIN_BACKWARD,
                    RIGHT_UPDATES,
                    LEFT_UPDATES,
                CHAIN_END,
            CHAIN_LIST_END,
            CHAIN_LIST_BEGIN,
                CHAIN_BACKWARD,
                    Q_UPDATES,
                    UNREGISTER,
                CHAIN_END,
            CHAIN_LIST_END,
            END
        }
    }
};

///
/// @brief Locates a task insertion plan descriptor that corresponds to a given
/// enumerator value.
///
///  If the given enumerator is invalid, then the function returns a NULL
///  pointer.
///
/// @param[in] type - enumerator value
///
/// @return matching task insertion plan descriptor
///
static struct plan_descr const * get_plan(starneig_reorder_plan_t type)
{
    for (int i = 0; i < sizeof(plans)/sizeof(plans[0]); i++)
        if (plans[i].type == type)
            return &plans[i];

    return NULL;
}

///
/// @brief Locates a task insertion blueprint descriptor that corresponds to a
/// given enumerator value.
///
///  If the given enumerator is invalid, then the function returns a NULL
///  pointer.
///
/// @param[in] type - enumerator value
///
/// @return matching task insertion blueprint descriptor
///
static struct blueprint_descr const * get_blueprint(
    starneig_reorder_blueprint_t type)
{
    for (int i = 0; i < sizeof(blueprints)/sizeof(blueprints[0]); i++)
        if (blueprints[i].type == type)
            return &blueprints[i];

    return NULL;
}

///
/// @brief Check if the given plan is valid with the given blueprint.
///
/// @param[in] plan - plan enumerator
/// @param[in] blueprint - blueprint descriptor
///
/// @return 1 if the plan is valid, 0 otherwise
///
static int is_valid_plan(
    starneig_reorder_plan_t plan, struct blueprint_descr const *blueprint)
{
    for (int i = 0; blueprint->valid_plans[i] != 0; i++)
        if (blueprint->valid_plans[i] == plan)
            return 1;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int starneig_reorder_get_optimal_tile_size(int n, double select_ratio)
{
    /*

    When the selection ratio (sr) is fixed, the optimal tile size seems to be
    a linear function of the matrix size. Therefore, it is assumed that
    the optimal tile size is of the form 1.0E-2 * A(sr) * n + B(sr).

    For A(sr) we have:

    Fitting Function: a(1-exp(-bx)) + c
    initial a= 2.5
    initial b= 1
    initial c= 1
    with Convergence Damping
    ________ Input Data (Fit values from inital guesses)________
       x 	             y 	            y(fit) 	 residual
     0.05   	   1.08019	   1.12193	-0.04174
     0.15   	   1.71557	   1.34823	 0.36734
     0.25   	   2.04984	   1.553  	 0.49684
     0.35   	   2.14621	   1.73828	 0.40793
     0.45   	   2.28273	   1.90593	 0.3768
    ======================   Results   =========================
    a= 1.7863 +- 0.0692
    b= 7.063 +- 0.82
    c= 0.5495 +- 0.087
    ______________________ Output Data _________________________
       x 	             y 	            y(fit) 	 residual
     0.05   	   1.08019	   1.08099	-0.0008
     0.15   	   1.71557	   1.71662	-0.00105
     0.25   	   2.04984	   2.03028	 0.01956
     0.35   	   2.14621	   2.18505	-0.03885
     0.45   	   2.28273	   2.26143	 0.0213
    ------------------------------------------------------------
    sum of squared residuals= 0.002347
    stand. dev. y values= 0.03426
    correlation between a & b= 0.3903
    correlation between b & c= -0.827
    correlation between a & c= -0.8144

    For B(sr) we have:

    Fitting Function: ax + b
    initial a= 40
    initial b= 0.25
    ________ Input Data (Fit values from inital guesses)________
      x 	             y 	            y(fit) 	 residual
     0.05   	  85.58544	   2.25   	83.33544
     0.15   	  72.41609	   6.25   	66.16609
     0.25   	  75.99583	  10.25   	65.74583
     0.35   	 108.87949	  14.25   	94.62949
     0.45   	 117.91548	  18.25   	99.66548
    ======================   Results   =========================
     a= 101.1 +- 45.3
     b= 66.9 +- 13
    ______________________ Output Data _________________________
       x 	             y 	            y(fit) 	 residual
     0.05   	  85.58544	  71.93377	13.65167
     0.15   	  72.41609	  82.04612	-9.63003
     0.25   	  75.99583	  92.15847	-16.16264
     0.35   	 108.87949	 102.27081	 6.60868
     0.45   	 117.91548	 112.38316	 5.53232
    ------------------------------------------------------------
    sum of squared residuals= 614.6
    stand. dev. y values= 14.31
    correlation between a & b= -0.8704

    */

    if (0.5 < select_ratio)
        select_ratio = 1.0 - select_ratio;

    double A = 1.7863*(1.0-exp(-7.063*select_ratio)) + 0.5495;
    double B = 101.1 * select_ratio + 66.9;

    return MAX(32, divceil(1.0E-2*A * n + B, 8)*8);
}

starneig_error_t starneig_reorder_insert_tasks(
    struct starneig_reorder_conf const *conf,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t Q, starneig_matrix_descr_t Z,
    starneig_matrix_descr_t A, starneig_matrix_descr_t B,
    starneig_vector_descr_t real, starneig_vector_descr_t imag,
    starneig_vector_descr_t beta,
    mpi_info_t mpi)
{
    // use default configuration if necessary
    struct starneig_reorder_conf _conf;
    if (conf == NULL) {
        starneig_reorder_init_conf(&_conf);
        conf = &_conf;
    }

    //
    // check mandatory arguments
    //

    if (selected == NULL) {
        starneig_error("Eigenvalue selection bitmap is NULL. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (A == NULL) {
        starneig_error("Matrix A is NULL. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    //
    // check matrix dimension and tile sizes
    //

    int n = STARNEIG_MATRIX_N(A);
    int tile_size = STARNEIG_MATRIX_BN(A);

    if (!starneig_is_valid_matrix(n, tile_size, Q)) {
        starneig_error("Matrix Q has invalid dimension. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (!starneig_is_valid_matrix(n, tile_size, Z)) {
        starneig_error("Matrix Z has invalid dimension. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (!starneig_is_valid_matrix(n, tile_size, A)) {
        starneig_error("Matrix A has invalid dimension. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (!starneig_is_valid_matrix(n, tile_size, B)) {
        starneig_error("Matrix B has invalid dimension. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (STARNEIG_VECTOR_BM(selected) != tile_size) {
        starneig_error(
            "Eigenvalue selection bitmap has invalid dimensions. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    //
    // setup plan and blueprint
    //

    // locate reordering plan descriptor
    struct plan_descr const *plan_desc = NULL;
    if (conf->plan != STARNEIG_REORDER_DEFAULT_PLAN) {
        plan_desc = get_plan(conf->plan);
        if (plan_desc == NULL) {
            starneig_error("Invalid plan. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    // locate insert engine descriptor
    struct blueprint_descr const *blueprint_desc = NULL;
    if (conf->blueprint != STARNEIG_REORDER_DEFAULT_BLUEPRINT) {
        blueprint_desc = get_blueprint(conf->blueprint);
        if (blueprint_desc == NULL) {
            starneig_error("Invalid engine. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }
    }

    // select default plan and engine if necessary

    if (plan_desc == NULL && blueprint_desc == NULL) {
        plan_desc = get_plan(STARNEIG_REORDER_MULTI_PART_PLAN);
        starneig_message("Using %s.", plan_desc->name);
    }

    if (plan_desc == NULL) {
        plan_desc = get_plan(blueprint_desc->preferred_plan);
        starneig_message("Using %s.", plan_desc->name);
    }

    if (blueprint_desc == NULL) {
        blueprint_desc = get_blueprint(plan_desc->preferred_blueprint);
        starneig_message("Using %s.", blueprint_desc->name);
    }

    // make sure that the selected plan-engine combination is valid
    if (!is_valid_plan(plan_desc->type, blueprint_desc)) {
        starneig_error("Invalid plan. Exiting...");
        return STARNEIG_INVALID_CONFIGURATION;
    }

    //
    // construct task insertion engine configuration structure
    //

    struct starneig_engine_conf_t engine_conf;

    // set small window size
    if (conf->small_window_size == STARNEIG_REORDER_DEFAULT_SMALL_WINDOW_SIZE) {
        if (B != NULL)
            engine_conf.small_window_size = 32;
        else
            engine_conf.small_window_size = 64;
    }
    else if (conf->small_window_size < 4) {
        starneig_error("Invalid small window size. Exiting...");
        return STARNEIG_INVALID_CONFIGURATION;
    }
    else {
        engine_conf.small_window_size = conf->small_window_size;
    }

    // check small window threshold
    if (conf->small_window_threshold ==
    STARNEIG_REORDER_DEFAULT_SMALL_WINDOW_THRESHOLD){
        if (B != NULL)
            engine_conf.small_window_threshold = 64;
        else
            engine_conf.small_window_threshold = 128;
    }
    else if (conf->small_window_threshold < 4) {
        starneig_error("Invalid small window size. Exiting...");
        return STARNEIG_INVALID_CONFIGURATION;
    }
    else {
        engine_conf.small_window_threshold = conf->small_window_threshold;
    }

    // figure out how many workers we have in total

    int world_size = starneig_mpi_get_comm_size();
    int worker_count = starpu_worker_get_count();

    // set update task widths
    if (conf->update_width < 1 ||
    conf->update_width == STARNEIG_REORDER_DEFAULT_UPDATE_WIDTH) {
        engine_conf.a_width = starneig_calc_update_size(
            STARNEIG_MATRIX_N(A), STARNEIG_MATRIX_BN(A),
            STARNEIG_MATRIX_SBN(A), world_size, worker_count);
        if (B != NULL)
            engine_conf.b_width = starneig_calc_update_size(
                STARNEIG_MATRIX_N(B), STARNEIG_MATRIX_BN(B),
                STARNEIG_MATRIX_SBN(B), world_size, worker_count);
    }
    else if (conf->update_width % tile_size != 0) {
        int valid_width = (conf->update_width/tile_size)*tile_size;
        starneig_warning(
            "Invalid update width. Setting update width to %d.", valid_width);

        engine_conf.a_width = valid_width;
        engine_conf.b_width = valid_width;;
    }
    else {
        engine_conf.a_width = conf->update_width;
        engine_conf.b_width = conf->update_width;
    }

    // set update task heights
    if (conf->update_height < 1 ||
    conf->update_height == STARNEIG_REORDER_DEFAULT_UPDATE_HEIGHT) {
        engine_conf.a_height = starneig_calc_update_size(
            STARNEIG_MATRIX_M(A), STARNEIG_MATRIX_BM(A),
            STARNEIG_MATRIX_SBM(A), world_size, worker_count);
        if (B != NULL)
            engine_conf.b_height = starneig_calc_update_size(
                STARNEIG_MATRIX_M(B), STARNEIG_MATRIX_BM(B),
                STARNEIG_MATRIX_SBM(B), world_size, worker_count);
        if (Q != NULL)
            engine_conf.q_height = starneig_calc_update_size(
                STARNEIG_MATRIX_M(Q), STARNEIG_MATRIX_BM(Q),
                STARNEIG_MATRIX_SBM(Q), world_size, worker_count);
        if (Z != NULL)
            engine_conf.z_height = starneig_calc_update_size(
                STARNEIG_MATRIX_M(Z), STARNEIG_MATRIX_BM(Z),
                STARNEIG_MATRIX_SBM(Z), world_size, worker_count);

    }
    else if (conf->update_height % tile_size != 0) {
        int valid_height = (conf->update_height/tile_size)*tile_size;
        starneig_warning(
            "Invalid update height. Setting update height to %d.",
            valid_height);

        engine_conf.a_height = valid_height;
        engine_conf.b_height = valid_height;
        engine_conf.q_height = valid_height;
        engine_conf.z_height = valid_height;

    }
    else {
        engine_conf.a_height = conf->update_height;
        engine_conf.b_height = conf->update_height;
        engine_conf.q_height = conf->update_height;
        engine_conf.z_height = conf->update_height;
    }

    //
    // check window size and values per chain arguments
    //

    int window_size = conf->window_size;
    int values_per_chain = conf->values_per_chain;

    if (window_size == STARNEIG_REORDER_DEFAULT_WINDOW_SIZE) {

        if (values_per_chain != STARNEIG_REORDER_DEFAULT_VALUES_PER_CHAIN)
            starneig_warning("Ignoring parameter values_per_chain.");

        window_size = -1;
        values_per_chain = -1;
        starneig_message("Using \"rounded\" window size.");
    }
    else if (window_size != STARNEIG_REORDER_ROUNDED_WINDOW_SIZE) {
        if (window_size < 4) {
            starneig_error("Invalid window size. Exiting...");
            return STARNEIG_INVALID_CONFIGURATION;
        }

        if (values_per_chain == STARNEIG_REORDER_DEFAULT_VALUES_PER_CHAIN) {
            values_per_chain = window_size/2;
            starneig_message(
                "Setting values per chain to %d.", conf->values_per_chain);
        }
        else {
            if (window_size-2 < values_per_chain) {
                starneig_error(
                    "Invalid number of selected eigenvalues per chain. "
                    "Exiting...");
                return STARNEIG_INVALID_CONFIGURATION;
            }
        }
    }

    //
    // initialize plan
    //

    int *host_selected = starneig_acquire_vector_descr(selected);

    starneig_vector_descr_t complex_distr_d =
        starneig_extract_subdiagonals(A, mpi);
    int *complex_distr = starneig_acquire_vector_descr(complex_distr_d);
    starneig_free_vector_descr(complex_distr_d);

    struct plan *plan = plan_desc->func(n, window_size, values_per_chain,
        tile_size, host_selected, complex_distr);

    free(host_selected);
    free(complex_distr);

    //
    // insert tasks
    //

    starneig_process_plan(&engine_conf, blueprint_desc->blueprint, selected,
        Q, Z, A, B, plan, mpi);

    //
    // finalize
    //

    starneig_free_plan(plan);

    if (real != NULL && imag != NULL)
        starneig_insert_extract_eigenvalues(
            STARPU_MAX_PRIO, A, B, real, imag, beta, mpi);

    return STARNEIG_SUCCESS;
}
