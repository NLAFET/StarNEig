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
#include "solvers.h"
#include "../common/common.h"
#include "../common/threads.h"
#include "../common/parse.h"
#include "../common/local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "../common/starneig_pencil.h"
#endif
#include <starneig/starneig.h>
#include <stdlib.h>

static hook_solver_state_t lapack_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return (hook_solver_state_t) env->data;
}

static int lapack_finalize(hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int lapack_run(hook_solver_state_t state)
{
    extern void dtrevc_(char const *, char const *, int const *, int const *,
        double const *, int const *, double *, int const *, double *,
        int const *, int const *, int *, double *, int *);

    extern void dtgevc_(char const *, char const *, int const *, int const *,
        double const *, int const *, double const *, int const *, double *,
        int const *, double *, int const *, int const *, int *, double *,
        int *);

    threads_set_mode(THREADS_MODE_LAPACK);

    pencil_t data = (pencil_t) state;
    int const *selected = get_supplementaty_selected(data->supp);
    int n = LOCAL_MATRIX_N(data->mat_a);

    int selected_count = 0;
    for (int i = 0; i < n; i++)
        if (selected[i]) selected_count++;

    int ldA = LOCAL_MATRIX_LD(data->mat_a);
    double *A = LOCAL_MATRIX_PTR(data->mat_a);

    data->mat_x =
        init_local_matrix(n, selected_count, NUM_REAL | PREC_DOUBLE);
    int ldX = LOCAL_MATRIX_LD(data->mat_x);
    double *X = LOCAL_MATRIX_PTR(data->mat_x);

    int ld_X;
    double *_X;
    {
        size_t ld;
        _X = alloc_matrix(n, selected_count, sizeof(double), &ld);
        ld_X = ld;
    }

    int info, selected_count2;
    double *work = NULL;

    if (data->mat_b != NULL) {
        int ldB = LOCAL_MATRIX_LD(data->mat_b);
        double *B = LOCAL_MATRIX_PTR(data->mat_b);

        work = malloc(6*n*sizeof(double));
        dtgevc_(
            "Right", "Selected", selected, &n, A, &ldA, B, &ldB, _X, &ld_X,
            _X, &ld_X, &selected_count, &selected_count2, work, &info);
    }
    else {
        work = malloc(3*n*sizeof(double));
        dtrevc_(
            "Right", "Selected", selected, &n, A, &ldA, _X, &ld_X, _X, &ld_X,
            &selected_count, &selected_count2, work, &info);
    }

    if (selected_count != selected_count2)
        info = 1;

    if (info != 0)
        goto cleanup;

    if (data->mat_b != NULL) {
        int ldZ = LOCAL_MATRIX_LD(data->mat_z);
        double *Z = LOCAL_MATRIX_PTR(data->mat_z);

        dgemm("No transpose", "No transpose",
            n, selected_count, n, 1.0, Z, ldZ, _X, ld_X, 0.0, X, ldX);
    }
    else {
        int ldQ = LOCAL_MATRIX_LD(data->mat_q);
        double *Q = LOCAL_MATRIX_PTR(data->mat_q);

        dgemm("No transpose", "No transpose",
            n, selected_count, n, 1.0, Q, ldQ, _X, ld_X, 0.0, X, ldX);
    }

cleanup:

    threads_set_mode(THREADS_MODE_DEFAULT);

    free(_X);
    free(work);

    return info;
}

const struct hook_solver eigenvectors_lapack_solver = {
    .name = "lapack",
    .desc = "LAPACK's dtrevc/dtgevc subroutine",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .prepare = &lapack_prepare,
    .finalize = &lapack_finalize,
    .run = &lapack_run
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
        "  --tile-size [default,(num)] -- tile size\n"
    );
}

static void starpu_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
    print_multiarg("--tile-size", argc, argv, "default", NULL);
}

static int starpu_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);
    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, argr, "default", NULL);

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    if (tile_size.type == MULTIARG_INVALID ||
    (tile_size.type == MULTIARG_INT && tile_size.int_value < 1)) {
        fprintf(stderr, "Invalid tile size.\n");
        return -1;
    }

    return 0;
}

static hook_solver_state_t starpu_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    struct starpu_state *state = malloc(sizeof(struct starpu_state));

    state->argc = argc;
    state->argv = argv;
    state->env = env;

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
        starneig_node_init(cores, gpus, STARNEIG_HINT_DM);
    else
#endif
        starneig_node_init(cores, gpus, STARNEIG_HINT_SM);

    return state;
}

static int starpu_finalize(hook_solver_state_t state, struct hook_data_env *env)
{
    if (state == NULL)
        return 0;

    starneig_node_finalize();

    free(state);
    return 0;
}

static int starpu_run(hook_solver_state_t state)
{
    int argc = ((struct starpu_state *) state)->argc;
    char * const *argv = ((struct starpu_state *) state)->argv;
    struct hook_data_env *env = ((struct starpu_state *) state)->env;

    struct starneig_eigenvectors_conf conf;
    starneig_eigenvectors_init_conf(&conf);

    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, NULL, "default", NULL);

    if (tile_size.type == MULTIARG_INT)
        conf.tile_size = tile_size.int_value;

    int ret = 0;

    pencil_t pencil = (pencil_t) env->data;

    int n = GENERIC_MATRIX_N(pencil->mat_a);
    int *selected = get_supplementaty_selected(pencil->supp);

    int selected_count = 0;
    for (int i = 0; i < n; i++)
        if (selected[i]) selected_count++;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil->mat_x =
            init_local_matrix(n, selected_count, NUM_REAL | PREC_DOUBLE);

        if (pencil->mat_b != NULL)
            ret = starneig_GEP_SM_Eigenvectors_expert(&conf, n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                LOCAL_MATRIX_PTR(pencil->mat_x), LOCAL_MATRIX_LD(pencil->mat_x)
            );
        else
            ret = starneig_SEP_SM_Eigenvectors_expert(&conf, n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_x), LOCAL_MATRIX_LD(pencil->mat_x)
            );
    }

//#ifdef STARNEIG_ENABLE_MPI
//    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
//    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
//        pencil->mat_x = init_starneig_matrix(
//            n, selected_count,
//            STARNEIG_MATRIX_BM(pencil->mat_a),
//            STARNEIG_MATRIX_BN(pencil->mat_a),
//            NUM_REAL | PREC_DOUBLE,
//            STARNEIG_MATRIX_DISTR(pencil->mat_a));
//
//        if (pencil->mat_b != NULL)
//            ret = starneig_GEP_DM_Eigenvectors_expert(&conf, selected,
//                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_x)
//            );
//        else
//            ret = starneig_SEP_DM_Eigenvectors_expert(&conf, selected,
//                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_x)
//            );
//    }
//#endif

    return ret;
}

const struct hook_solver eigenvectors_starpu_solver = {
    .name = "starneig",
    .desc = "StarPU based subroutine",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
//#ifdef STARNEIG_ENABLE_MPI
//        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
//#endif
//#ifdef STARNEIG_ENABLE_BLACS
//        HOOK_DATA_FORMAT_PENCIL_BLACS,
//#endif
        0 },
    .print_usage = &starpu_print_usage,
    .print_args = &starpu_print_args,
    .check_args = &starpu_check_args,
    .prepare = &starpu_prepare,
    .finalize = &starpu_finalize,
    .run = &starpu_run
};

////////////////////////////////////////////////////////////////////////////////

static void starpu_simple_print_usage(int argc, char * const *argv)
{
    printf(
        "  --cores [default,(num)} -- Number of CPU cores\n"
        "  --gpus [default,(num)} -- Number of GPUS\n"
    );
}

static void starpu_simple_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
}

static int starpu_simple_check_args(int argc, char * const *argv, int *argr)
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

static hook_solver_state_t starpu_simple_prepare(
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
        starneig_node_init(cores, gpus, STARNEIG_HINT_DM);
    else
#endif
        starneig_node_init(
            cores, gpus, STARNEIG_HINT_SM);

    return env;
}

static int starpu_simple_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    if (state == NULL)
        return 0;

    starneig_node_finalize();

    return 0;
}

static int starpu_simple_run(hook_solver_state_t state)
{
    struct hook_data_env *env = state;

    int ret = 0;

    pencil_t pencil = (pencil_t) env->data;

    int n = GENERIC_MATRIX_N(pencil->mat_a);
    int *selected = get_supplementaty_selected(pencil->supp);

    int selected_count = 0;
    for (int i = 0; i < n; i++)
        if (selected[i]) selected_count++;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil->mat_x =
            init_local_matrix(n, selected_count, NUM_REAL | PREC_DOUBLE);

        if (pencil->mat_b != NULL)
            ret = starneig_GEP_SM_Eigenvectors(n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                LOCAL_MATRIX_PTR(pencil->mat_x), LOCAL_MATRIX_LD(pencil->mat_x)
            );
        else
            ret = starneig_SEP_SM_Eigenvectors(n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_x), LOCAL_MATRIX_LD(pencil->mat_x)
            );
    }

//#ifdef STARNEIG_ENABLE_MPI
//    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
//    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
//        pencil->mat_x = init_starneig_matrix(
//            n, selected_count,
//            STARNEIG_MATRIX_BM(pencil->mat_a),
//            STARNEIG_MATRIX_BN(pencil->mat_a),
//            NUM_REAL | PREC_DOUBLE,
//            STARNEIG_MATRIX_DISTR(pencil->mat_a));
//
//        if (pencil->mat_b != NULL)
//            ret = starneig_GEP_DM_Eigenvectors(selected,
//                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_x)
//            );
//        else
//            ret = starneig_SEP_DM_Eigenvectors(selected,
//                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
//                STARNEIG_MATRIX_HANDLE(pencil->mat_x)
//            );
//    }
//#endif

    return ret;
}

const struct hook_solver eigenvectors_starpu_simple_solver = {
    .name = "starneig-simple",
    .desc = "StarPU based subroutine (simplified interface)",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
//#ifdef STARNEIG_ENABLE_MPI
//        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
//#endif
//#ifdef STARNEIG_ENABLE_BLACS
//        HOOK_DATA_FORMAT_PENCIL_BLACS,
//#endif
        0 },
    .print_usage = &starpu_simple_print_usage,
    .print_args = &starpu_simple_print_args,
    .check_args = &starpu_simple_check_args,
    .prepare = &starpu_simple_prepare,
    .finalize = &starpu_simple_finalize,
    .run = &starpu_simple_run
};
