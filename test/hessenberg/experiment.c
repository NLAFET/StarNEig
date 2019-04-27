///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
///
/// Copyright (c) 2019, Umeå Universitet
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
#include "../common/init.h"
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
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const struct hook_experiment_descr hessenberg_experiment = {
    .initializers = (struct hook_initializer_t const *[])
    {
        &default_initializer,
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
