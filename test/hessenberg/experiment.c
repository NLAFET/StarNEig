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
#include "experiment.h"
#include "solvers.h"
#include "../common/parse.h"
#include "../common/local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "../common/starneig_pencil.h"
#endif
#include "../common/crawler.h"
#include "../common/complex_distr.h"
#include "../common/init.h"
#include "../common/checks.h"
#include "../common/hooks.h"
#include "../common/io.h"
#include <starneig/starneig.h>
#include <stdlib.h>
#include <stdio.h>

static void default_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n -- Problem dimension\n"
        "  --generalized -- Generalized problem\n"
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

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = init_pencil();

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
    .desc = "Default Hessenberg initializer",
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

static void known_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n (num) -- Problem dimension\n"
        "  --generalized -- Generalized problem\n"
        "  --complex-distr (complex distribution) -- 2-by-2 block "
        "distribution module\n"
    );

    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void known_initializer_print_args(int argc, char * const *argv)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));

    int generalized = read_opt("--generalized", argc, argv, NULL);

    if (generalized)
        printf(" --generalized");

    struct complex_distr const *complex_distr =
        read_complex_distr("--complex-distr", argc, argv, NULL);

    printf(" --complex-distr %s", complex_distr->name);

    if (complex_distr->print_args != NULL)
        complex_distr->print_args(argc, argv);

    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int known_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    if (read_int("--n", argc, argv, argr, -1) < 1)
        return 1;

    read_opt("--generalized", argc, argv, argr);

    struct complex_distr const *complex_distr =
        read_complex_distr("--complex-distr", argc, argv, argr);
    if (complex_distr == NULL) {
        fprintf(stderr, "Invalid 2-by-2 block distribution module.\n");
        return -1;
    }

    if (complex_distr->check_args != NULL) {
        int ret = complex_distr->check_args(argc, argv, argr);
        if (ret)
            return ret;
    }

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static struct hook_data_env* known_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);

    int generalized = read_opt("--generalized", argc, argv, NULL);

    struct complex_distr const *complex_distr =
        read_complex_distr("--complex-distr", argc, argv, NULL);

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t pencil = env->data = init_pencil();

    double *real, *imag, *beta;
    init_supplementary_known_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

    // generate (generalized) Schur form and multiply with Householder
    // reflectors from both sides

    if (generalized) {
        matrix_t mat_s = generate_random_uptriag(n, n, helper);
        matrix_t mat_t = generate_identity(n, n, helper);

        complex_distr->init(argc, argv, mat_s, mat_t);
        extract_eigenvalues(mat_s, mat_t, real, imag, beta);

        matrix_t mat_q = generate_random_householder(n, helper);
        matrix_t mat_z = generate_random_householder(n, helper);

        mul_QAZT(mat_q, mat_s, mat_z, &pencil->mat_a);
        mul_QAZT(mat_q, mat_t, mat_z, &pencil->mat_b);

        free_matrix_descr(mat_s);
        free_matrix_descr(mat_t);
        free_matrix_descr(mat_q);
        free_matrix_descr(mat_z);
    }
    else {
        matrix_t mat_s = generate_random_uptriag(n, n, helper);

        complex_distr->init(argc, argv, mat_s, NULL);
        extract_eigenvalues(mat_s, NULL, real, imag, beta);

        matrix_t mat_q = generate_random_householder(n, helper);

        mul_QAZT(mat_q, mat_s, mat_q, &pencil->mat_a);

        free_matrix_descr(mat_s);
        free_matrix_descr(mat_q);
    }

    pencil->mat_q = generate_identity(n, n, helper);
    if (generalized)

    init_helper_free(helper);

    return env;
}

static const struct hook_initializer_t known_initializer = {
    .name = "known",
    .desc = "Generates an upper Hessenberg matrix with known eigenvalues",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &known_initializer_print_usage,
    .print_args = &known_initializer_print_args,
    .check_args = &known_initializer_check_args,
    .init = &known_initializer_init
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void print_usage(int argc, char * const *argv)
{
    print_avail_complex_distr();
    print_opt_complex_distr();
}

const struct hook_experiment_descr hessenberg_experiment = {
    .print_usage = &print_usage,
    .initializers = (struct hook_initializer_t const *[])
    {
        &default_initializer,
        &known_initializer,
        &mtx_initializer,
        &raw_initializer,
        0
    },
    .supplementers = (struct hook_supplementer_t const *[])
    {
        0
    },
    .solvers = (struct hook_solver const *[])
    {
        &hessenberg_starpu_solver,
        &hessenberg_starpu_simple_solver,
        &hessenberg_lapack_solver,
#ifdef PDGEHRD_FOUND
        &hessenberg_scalapack_solver,
#endif
#ifdef MAGMA_FOUND
        &hessenberg_magma_solver,
#endif
        0
    },
    .hook_descrs = (struct hook_descr_t const *[])
    {
        &default_hessenberg_test_descr,
        &default_residual_test_descr,
        &default_print_pencil_descr,
        &default_print_input_pencil_descr,
        &default_store_raw_pencil_descr,
        &default_store_raw_input_pencil_descr, 0
    }
};
