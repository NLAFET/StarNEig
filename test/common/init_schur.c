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
#include "init_schur.h"
#include "complex_distr.h"
#include "select_distr.h"
#include "hook_experiment.h"
#include "local_pencil.h"
#include "threads.h"
#ifdef STARNEIG_ENABLE_MPI
#include "starneig_pencil.h"
#endif
#include "init.h"
#include "parse.h"
#include <starneig/starneig.h>
#include <stdlib.h>
#include <stdio.h>

static void default_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n -- Problem dimension\n"
        "  --generalized -- Form a generalized problem\n"
        "  --complex-distr (complex distribution) -- 2-by-2 block "
        "distribution module\n"
        "  --qI -- Initialize Q to identity\n"
        "  --zI -- Initialize Z to identity\n"
    );

    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void default_initializer_print_args(int argc, char * const *argv)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));

    int generalized = read_opt("--generalized", argc, argv, NULL);

    if (generalized)
        printf(" --generalized");

    struct complex_distr const *complex_distr =
        read_complex_distr("--complex-distr", argc, argv, NULL);

    printf(" --complex-distr %s", complex_distr->name);

    if (read_opt("--qI", argc, argv, NULL))
        printf(" --qI");
    if (generalized && read_opt("--zI", argc, argv, NULL))
        printf(" --zI");

    if (complex_distr->print_args != NULL)
        complex_distr->print_args(argc, argv);

    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int default_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    if (read_int("--n", argc, argv, argr, -1) < 1)
        return 1;

    int generalized = read_opt("--generalized", argc, argv, argr);

    struct complex_distr const *complex_distr =
        read_complex_distr("--complex-distr", argc, argv, argr);
    if (complex_distr == NULL) {
        fprintf(stderr, "Invalid 2-by-2 block distribution module.\n");
        return -1;
    }

    read_opt("--qI", argc, argv, argr);

    if (generalized)
        read_opt("--zI", argc, argv, argr);

    if (complex_distr->check_args != NULL) {
        int ret = complex_distr->check_args(argc, argv, argr);
        if (ret)
            return ret;
    }

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static struct hook_data_env* default_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);

    int generalized = read_opt("--generalized", argc, argv, NULL);

    struct complex_distr const *complex_distr =
        read_complex_distr("--complex-distr", argc, argv, NULL);

    int qI = read_opt("--qI", argc, argv, NULL);
    int zI = read_opt("--zI", argc, argv, NULL);

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t pencil = env->data = init_pencil();

    //
    // generate A matrix
    //

    pencil->mat_a = generate_random_uptriag(n, n, helper);
    if (generalized)
        pencil->mat_b = generate_random_uptriag(n, n, helper);

    if (complex_distr->init != NULL)
        complex_distr->init(argc, argv, pencil->mat_a, pencil->mat_b);

    //
    // generate Q matrix
    //

    if (qI)
        pencil->mat_q = generate_identity(n, n, helper);
    else
        pencil->mat_q = generate_random_householder(n, helper);



    if (generalized) {

        //
        // generate Z matrix
        //

        if (zI)
            pencil->mat_z = generate_identity(n, n, helper);
        else
            pencil->mat_z = generate_random_householder(n, helper);

    }

    init_helper_free(helper);

    return env;
}

////////////////////////////////////////////////////////////////////////////////

static void starpu_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n -- Problem dimension\n"
        "  --generalized -- Form a generalized problem\n"
    );

    init_helper_print_usage("", INIT_HELPER_BLACS_PENCIL, argc, argv);
}

static void starpu_initializer_print_args(int argc, char * const *argv)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));

    if (read_opt("--generalized", argc, argv, NULL))
        printf(" --generalized");

    init_helper_print_args("", INIT_HELPER_BLACS_PENCIL, argc, argv);
}

static int starpu_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    if (read_int("--n", argc, argv, argr, -1) < 1)
        return 1;

    read_opt("--generalized", argc, argv, argr);

    return init_helper_check_args(
        "", INIT_HELPER_BLACS_PENCIL, argc, argv, argr);
}

static struct hook_data_env* starpu_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT... \n");

    int n = read_int("--n", argc, argv, NULL, -1);

    int generalized = read_opt("--generalized", argc, argv, NULL);

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t pencil = env->data = init_pencil();

    //
    // generate matrices
    //

    pencil->mat_a = generate_random_fullpos(n, n, helper);
    pencil->mat_q = generate_identity(n, n, helper);
    pencil->mat_ca = copy_matrix_descr(pencil->mat_a);

    if (generalized) {
        pencil->mat_b = generate_random_fullpos(n, n, helper);
        pencil->mat_z = generate_identity(n, n, helper);
        pencil->mat_cb = copy_matrix_descr(pencil->mat_b);
    }

#ifdef STARNEIG_ENABLE_BLACS
    if (format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        starneig_node_init(threads_get_workers(), STARNEIG_USE_ALL,
            STARNEIG_HINT_DM | STARNEIG_FXT_DISABLE);

        if (generalized) {
            starneig_GEP_DM_HessenbergTriangular(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z));

            starneig_GEP_DM_Schur(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
                NULL, NULL, NULL);
        }
        else {
            starneig_SEP_DM_Hessenberg(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q));

            starneig_SEP_DM_Schur(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                NULL, NULL);
        }

        starneig_node_finalize();
    }
#endif

    if (format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        starneig_node_init(threads_get_workers(), STARNEIG_USE_ALL,
            STARNEIG_HINT_SM | STARNEIG_FXT_DISABLE);

        if (generalized) {
            starneig_GEP_SM_HessenbergTriangular(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z)
            );

            starneig_GEP_SM_Schur(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                NULL, NULL, NULL);
        }

        else {
            starneig_SEP_SM_Hessenberg(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q));

            starneig_SEP_SM_Schur(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                NULL, NULL);
        }

        starneig_node_finalize();
    }

    init_helper_free(helper);

    return env;
}

void schur_initializer_print_usage(int argc, char * const *argv)
{
    print_avail_complex_distr();
    print_opt_complex_distr();
}

const struct hook_initializer_t default_schur_initializer = {
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

const struct hook_initializer_t starpu_schur_initializer = {
    .name = "starneig",
    .desc = "StarPU based initializer",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &starpu_initializer_print_usage,
    .print_args = &starpu_initializer_print_args,
    .check_args = &starpu_initializer_check_args,
    .init = &starpu_initializer_init
};
