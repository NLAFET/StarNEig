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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "full_chain.h"
#include "../common/parse.h"
#include "../common/init.h"
#include "../common/hooks.h"
#include "../common/local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "../common/starneig_pencil.h"
#endif
#include "../common/io.h"
#include <starneig/starneig.h>
#include <stdlib.h>
#include <math.h>

static void default_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n (num) -- Problem dimension\n"
        "  --generalized -- Generate a generalized problem\n"
    );
    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void default_initializer_print_args(int argc, char * const *argv)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));
    if (read_opt("--generalized", argc, argv, NULL))
        printf(" --generalized");
    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int default_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    if (read_int("--n", argc, argv, argr, -1) < 1)
        return 1;

    read_opt("--generalized", argc, argv, argr);

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static struct hook_data_env* default_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);
    int generalized = read_opt("--generalized", argc, argv, NULL);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = init_pencil();

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    data->mat_a = generate_random_fullpos(n, n, helper);
    data->mat_q = generate_identity(n, n, helper);
    if (generalized) {
        data->mat_b = generate_random_fullpos(n, n, helper);
        data->mat_z = generate_identity(n, n, helper);
    }

    init_helper_free(helper);

    return env;
}

static const struct hook_initializer_t default_initializer = {
    .name = "default",
    .desc = "Default initializer",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &default_initializer_print_usage,
    .print_args = &default_initializer_print_args,
    .check_args = &default_initializer_check_args,
    .init = &default_initializer_init
};

////////////////////////////////////////////////////////////////////////////////

struct starpu_state {
    int argc;
    char * const *argv;
    struct hook_data_env *env;
};

static void starpu_print_usage(int argc, char * const *argv)
{
    printf(
        "  --cores [default,(num)} -- Number of CPU cores\n"
        "  --gpus [default,(num)} -- Number of GPUS\n"
    );
}

static void starpu_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
}

static int starpu_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    return 0;
}

static hook_solver_state_t starpu_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, NULL, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, NULL, "default", NULL);

    int cores = STARNEIG_USE_ALL;
    if (arg_cores.type == MULTIARG_INT)
        cores = arg_cores.int_value;

    int gpus = STARNEIG_USE_ALL;
    if (arg_gpus.type == MULTIARG_INT)
        gpus = arg_gpus.int_value;

#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS)
        starneig_node_init(cores, gpus, STARNEIG_FAST_DM);
    else
#endif
        starneig_node_init(
            cores, gpus, STARNEIG_HINT_SM | STARNEIG_AWAKE_WORKERS);

    return env;
}

static int starpu_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    if (state == NULL)
        return 0;

    starneig_node_finalize();
    return 0;
}

static int sep_predicate(double real, double imag, void *arg)
{
    if (0 < real)
        return 1;
    return 0;
}

static int gep_predicate(double real, double imag, double beta, void *arg)
{
    if (0 < real/beta && beta != 0.0)
        return 1;
    return 0;
}

static int starpu_run(hook_solver_state_t state)
{
    struct hook_data_env *env = state;

    int ret = 0;
    int num_selected;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil_t pencil = (pencil_t) env->data;
        int n = LOCAL_MATRIX_N(pencil->mat_a);

        double *real, *imag, *beta;
        init_supplementary_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

        if (pencil->mat_b != NULL) {
            ret = starneig_GEP_SM_Reduce(n,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                real, imag, beta, &gep_predicate, NULL, NULL, &num_selected);
        }
        else {
            ret = starneig_SEP_SM_Reduce(n,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                real, imag, &sep_predicate, NULL, NULL, &num_selected);
        }
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        pencil_t pencil = (pencil_t) env->data;
        int n = starneig_distr_matrix_get_rows(
            STARNEIG_MATRIX_HANDLE(pencil->mat_a));

        double *real, *imag, *beta;
        init_supplementary_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

        if (pencil->mat_b != NULL) {
#ifdef STARNEIG_GEP_DM_REDUCE
            ret = starneig_GEP_DM_Reduce(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
                real, imag, beta, &gep_predicate, NULL, NULL, &num_selected);
#else
            fprintf(stderr, "starneig_GEP_DM_Reduce() does not exists.\n");
            ret = -1;
#endif
        }
        else {
#ifdef STARNEIG_SEP_DM_REDUCE
            ret = starneig_SEP_DM_Reduce(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                real, imag, &sep_predicate, NULL, NULL, &num_selected);
#else
            fprintf(stderr, "starneig_SEP_DM_Reduce() does not exists.\n");
            ret = -1;
#endif
        }
    }
#endif

    return ret;
}

static const struct hook_solver starpu_solver = {
    .name = "starneig-simple",
    .desc = "StarPU based subroutines",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &starpu_print_usage,
    .print_args = &starpu_print_args,
    .check_args = &starpu_check_args,
    .prepare = &starpu_prepare,
    .finalize = &starpu_finalize,
    .run = &starpu_run
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


const struct hook_experiment_descr full_chain_experiment = {
    .initializers = (struct hook_initializer_t const *[])
    {
        &default_initializer,
        0
    },
    .supplementers = (struct hook_supplementer_t const *[])
    {
        0
    },
    .solvers = (struct hook_solver const *[])
    {
        &starpu_solver,
        0
    },
    .hook_descrs = (struct hook_descr_t const *[])
    {
        &default_schur_test_descr,
        &default_residual_test_descr,
        &default_print_pencil_descr,
        &default_print_input_pencil_descr,
        &default_store_raw_pencil_descr,
        &default_store_raw_input_pencil_descr,
        0
    }
};
