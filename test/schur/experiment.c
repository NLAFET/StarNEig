///
/// @file This file contains the Schur reduction experiment.
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
#include "../common/common.h"
#include "../common/threads.h"
#include "../common/parse.h"
#include "../common/init.h"
#include "../common/checks.h"
#include "../common/hooks.h"
#include "../common/local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "../common/starneig_pencil.h"
#endif
#include "../common/io.h"
#include "../common/crawler.h"
#include "../common/complex_distr.h"
#include "../hessenberg/solvers.h"
#include <starneig/starneig.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#ifdef GSL_FOUND
#include <gsl/gsl_randist.h>
#endif

#define DEFLATE         0x1
#define SET_TO_INF      0x2

static int deflate_and_place_infinities_crawler(
    int offset, int size, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    int const *sub = arg;

    double *A = ptrs[0];
    size_t ldA = lds[0];

    double *B = NULL; size_t ldB = 0;
    if (ptrs[1] != NULL) {
        B = ptrs[1];
        ldB = lds[1];
    }

    for (int i = 1; i < size; i++) {
        if (sub[offset+i] & DEFLATE)
            A[(i-1)*ldA+i] = 0.0;
        if (B != NULL && sub[offset+i] & SET_TO_INF)
            B[i*ldB+i] = 0.0;
    }

    return size;
}

///
/// @brief Modifies a matrix pencil such that it is decoupled to multiple
/// independent subproblems and adds
///
/// @param[in] cuts
///         The desired number of deflations.
///
/// @param[in] infinities
///         The desired number of infinities.
///
/// @param[in,out] pencil
///         The matrix pencil.
///
static void deflate_and_place_infinities(
    int cuts, int infinities, pencil_t pencil)
{
    int n = GENERIC_MATRIX_N(pencil->mat_a);
    int *sub = malloc(n*sizeof(int));
    memset(sub, 0, n*sizeof(int));

    cuts = MIN(cuts, n-1);
    for (int i = 0; i < cuts; i++) {
        int p = prand() % (n-1) + 1;
        while (sub[p] & DEFLATE)
            p = prand() % (n-1) + 1;
        sub[p] |= DEFLATE;
    }

    for (int i = 0; i < infinities; i++) {
        int p = prand() % (n-1) + 1;
        while (sub[p] & SET_TO_INF)
            p = prand() % (n-1) + 1;
        sub[p] |= SET_TO_INF;
    }

    crawl_matrices(CRAWLER_RW, CRAWLER_DIAG_WINDOW,
        &deflate_and_place_infinities_crawler, sub, 0,
        pencil->mat_a, pencil->mat_b, NULL);

    free(sub);

    free_matrix_descr(pencil->mat_ca);
    pencil->mat_ca = NULL;

    free_matrix_descr(pencil->mat_cb);
    pencil->mat_cb = NULL;

    fill_pencil(pencil);
}

////////////////////////////////////////////////////////////////////////////////

static void random_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n (num) -- Problem dimension\n"
        "  --generalized -- Generalized problem\n"
        "  --decouple (num) -- Decouple the problem\n"
        "  --set-to-inf (num) -- Place infinities\n"
    );

    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void random_initializer_print_args(int argc, char * const *argv)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));
    if (read_opt("--generalized", argc, argv, NULL))
        printf(" --generalized");

    printf(" --decouple %d", read_int("--decouple", argc, argv, NULL, 0));
    printf(" --set-to-inf %d", read_int("--set-to-inf", argc, argv, NULL, 0));

    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int random_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    if (read_int("--n", argc, argv, argr, -1) < 1)
        return 1;

    read_opt("--generalized", argc, argv, argr);

    if (read_int("--decouple", argc, argv, argr, 0) < 0)
        return 1;

    if (read_int("--set-to-inf", argc, argv, argr, 0) < 0)
        return 1;

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static struct hook_data_env* random_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);
    int generalized = read_opt("--generalized", argc, argv, NULL);
    int decouple = read_int("--decouple", argc, argv, NULL, 0);
    int set_to_inf = read_int("--set-to-inf", argc, argv, NULL, 0);

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = init_pencil();

    data->mat_a = generate_random_hessenberg(n, n, helper);
    data->mat_q = generate_random_householder(n, helper);

    if (generalized) {
        data->mat_b = generate_random_uptriag(n, n, helper);
        data->mat_z = generate_random_householder(n, helper);
    }

    if (0 < decouple || 0 < set_to_inf)
        deflate_and_place_infinities(decouple, set_to_inf, data);

    init_helper_free(helper);

    return env;
}

static const struct hook_initializer_t random_initializer = {
    .name = "random",
    .desc = "Generates a random upper Hessenberg matrix",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &random_initializer_print_usage,
    .print_args = &random_initializer_print_args,
    .check_args = &random_initializer_check_args,
    .init = &random_initializer_init
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

    // reduce the dense matrix (pencil) to Hessenberg(-triangular) form

    pencil->mat_q = generate_identity(n, n, helper);
    pencil->mat_ca = copy_matrix_descr(pencil->mat_a);
    if (generalized) {
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
        }
        else {
            starneig_SEP_DM_Hessenberg(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q));
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
        }

        else {
            starneig_SEP_SM_Hessenberg(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q));
        }

        starneig_node_finalize();
    }

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

#ifdef GSL_FOUND

static void hessrand_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --n (num) -- Problem dimension\n"
        "  --generalized -- Generalized problem\n"
        "  --decouple (num) -- Decouple the problem\n"
        "  --set-to-inf (num) -- Place infinities\n"
    );

    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void hessrand_initializer_print_args(int argc, char * const *argv)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));
    if (read_opt("--generalized", argc, argv, NULL))
        printf(" --generalized");

    printf(" --decouple %d", read_int("--decouple", argc, argv, NULL, 0));
    printf(" --set-to-inf %d", read_int("--set-to-inf", argc, argv, NULL, 0));

    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int hessrand_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    if (read_int("--n", argc, argv, argr, -1) < 1)
        return 1;

    read_opt("--generalized", argc, argv, argr);

    if (read_int("--decouple", argc, argv, argr, 0) < 0)
        return 1;

    if (read_int("--set-to-inf", argc, argv, argr, 0) < 0)
        return 1;

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static int hessrand_crawler(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    gsl_rng *r = arg;

    {
        double *A = ptrs[0];
        size_t ldA = lds[0];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < offset+i+1; j++)
                A[i*ldA+j] = gsl_ran_gaussian(r, 1.0);
            if (offset+i+1 < m)
                A[i*ldA+offset+i+1] = sqrt(gsl_ran_chisq(r, n-offset-i-1));
            for (int j = offset+i+2; j < m; j++)
                A[i*ldA+j] = 0.0;
        }
    }

    if (1 < count) {
        double *B = ptrs[1];
        size_t ldB = lds[1];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < offset+i; j++)
                B[i*ldB+j] = gsl_ran_gaussian(r, 1.0);
            B[i*ldB+offset+i] = sqrt(gsl_ran_chisq(r, offset+i));
            for (int j = offset+i+1; j < m; j++)
                B[i*ldB+j] = 0.0;
        }

        if (offset == 0)
            B[0] = sqrt(gsl_ran_chisq(r, n));
    }

    return width;
}

static struct hook_data_env* hessrand_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);
    int generalized = read_opt("--generalized", argc, argv, NULL);
    int decouple = read_int("--decouple", argc, argv, NULL, 0);
    int set_to_inf = read_int("--set-to-inf", argc, argv, NULL, 0);

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = init_pencil();

    data->mat_a = init_matrix(n, n, helper);
    if (generalized)
        data->mat_b = init_matrix(n, n, helper);

    gsl_rng_env_setup();

    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r = gsl_rng_alloc(T);
    gsl_rng_set(r, prand());

    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &hessrand_crawler, r, 0,
        data->mat_a, data->mat_b, NULL);

    gsl_rng_free(r);

    data->mat_q = generate_random_householder(n, helper);
    if (generalized)
        data->mat_z = generate_random_householder(n, helper);

    if (0 < decouple || 0 < set_to_inf)
        deflate_and_place_infinities(decouple, set_to_inf, data);

    init_helper_free(helper);

    return env;
}

static const struct hook_initializer_t hessrand_initializer = {
    .name = "hessrand",
    .desc = "Generates a hessrand upper Hessenberg matrix",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &hessrand_initializer_print_usage,
    .print_args = &hessrand_initializer_print_args,
    .check_args = &hessrand_initializer_check_args,
    .init = &hessrand_initializer_init
};

#endif

////////////////////////////////////////////////////////////////////////////////

static void generic_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --left-input (mtx filename) -- Left-hand side matrix input "
        "file name\n"
        "  --right-input (mtx filename) -- Right-hand side matrix input "
        "file name\n"
        "  --input-begin (num) -- First matrix row/column to be read\n"
        "  --input-end (num) -- Last matrix row/column to be read + 1\n"
        "  --n (num) -- Problem dimension\n"
        "  --generalized -- Generate a generalized problem\n"
        "  --decouple (num) -- Decouple the problem\n"
        "  --set-to-inf (num) -- Place infinities\n"
    );
    init_helper_print_usage("", INIT_HELPER_BLACS_PENCIL, argc, argv);
}

static void generic_initializer_print_args(int argc, char * const *argv)
{
    char const *left_input = read_str("--left-input", argc, argv, NULL, NULL);
    char const *right_input = read_str("--right-input", argc, argv, NULL, NULL);

    if (left_input != NULL) {
        printf(" --left-input %s", left_input);
        if (right_input)
            printf(" --right-input %s", right_input);
        int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
        int input_end = read_int("--input-end", argc, argv, NULL, -1);
        if (0 <= input_begin && 0 <= input_end)
            printf(" --input-begin %d --input-end %d", input_begin, input_end);
    }
    else {
        printf(" --n %d", read_int("--n", argc, argv, NULL, -1));
        if (read_opt("--generalized", argc, argv, NULL))
            printf(" --generalized");
    }

    printf(" --decouple %d", read_int("--decouple", argc, argv, NULL, 0));
    printf(" --set-to-inf %d", read_int("--set-to-inf", argc, argv, NULL, 0));

    init_helper_print_args("", INIT_HELPER_BLACS_PENCIL, argc, argv);
}

static int generic_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    char const *left_input = read_str("--left-input", argc, argv, argr, NULL);
    char const *right_input = read_str("--right-input", argc, argv, argr, NULL);
    if (left_input != NULL) {
        if (access(left_input, R_OK) != 0) {
            fprintf(stderr, "Left-hand side input file does not exists.\n");
            return 1;
        }
        int input_begin = read_int("--input-begin", argc, argv, argr, -1);
        int input_end = read_int("--input-end", argc, argv, argr, -1);
        if (input_begin < 0 && input_end < input_begin)
            return 1;
    }
    else if (right_input != NULL) {
        fprintf(stderr, "Left-hand side input filename is missing.\n");
        return 1;
    }
    else {
        if (read_int("--n", argc, argv, argr, -1) < 1)
            return 1;
        read_opt("--generalized", argc, argv, argr);
    }

    if (read_int("--decouple", argc, argv, argr, 0) < 0)
        return 1;

    if (read_int("--set-to-inf", argc, argv, argr, 0) < 0)
        return 1;

    return init_helper_check_args(
        "", INIT_HELPER_BLACS_PENCIL, argc, argv, argr);
}

static struct hook_data_env* lapack_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);
    char const *left_input = read_str("--left-input", argc, argv, NULL, NULL);
    char const *right_input = read_str("--right-input", argc, argv, NULL, NULL);
    int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
    int input_end = read_int("--input-end", argc, argv, NULL, -1);
    int generalized = read_opt("--generalized", argc, argv, NULL);
    int decouple = read_int("--decouple", argc, argv, NULL, 0);
    int set_to_inf = read_int("--set-to-inf", argc, argv, NULL, 0);

    if (left_input != NULL) {
        int m;
        read_mtx_dimensions_from_file(left_input, &m, &n);
    }

    if (input_begin == -1)
        input_begin = 0;
    if (input_end == -1)
        input_end = n;

    init_helper_t helper = init_helper_init_hook(
        "", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = HOOK_DATA_FORMAT_PENCIL_LOCAL;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = data = init_pencil();

    // initialize A

    if (left_input != NULL)
        data->mat_a = read_mtx_sub_matrix_from_file(
            input_begin, input_end, left_input, helper);
    else
        data->mat_a = generate_random_fullpos(n, n, helper);

    // initialize B

    if (right_input != NULL) {
        generalized = 1;
        data->mat_b = read_mtx_sub_matrix_from_file(
            input_begin, input_end, right_input, helper);
    }
    else if (generalized) {
        data->mat_b = generate_random_fullpos(n, n, helper);
    }

    // initialize Q and Z

    data->mat_q = generate_identity(n, n, helper);
    if (generalized)
        data->mat_z = generate_identity(n, n, helper);

    // prepare for decoupling

    if (decouple < 1) {
        data->mat_ca = copy_matrix_descr(data->mat_a);
        data->mat_cb = copy_matrix_descr(data->mat_b);
    }

    // reduce

    hook_solver_state_t state =
        hessenberg_lapack_solver.prepare(argc, argv, env);
    hessenberg_lapack_solver.run(state);
    hessenberg_lapack_solver.finalize(state, env);

    if (0 < decouple || 0 < set_to_inf)
        deflate_and_place_infinities(decouple, set_to_inf, data);

    init_helper_free(helper);

    printf("INIT FINISHED.\n");

    return env;
}

static struct hook_data_env* starpu_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    int n = read_int("--n", argc, argv, NULL, -1);
    char const *left_input = read_str("--left-input", argc, argv, NULL, NULL);
    char const *right_input = read_str("--right-input", argc, argv, NULL, NULL);
    int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
    int input_end = read_int("--input-end", argc, argv, NULL, -1);
    int generalized = read_opt("--generalized", argc, argv, NULL);
    int decouple = read_int("--decouple", argc, argv, NULL, 0);
    int set_to_inf = read_int("--set-to-inf", argc, argv, NULL, 0);

    if (left_input != NULL) {
        int m;
        read_mtx_dimensions_from_file(left_input, &m, &n);
    }

    init_helper_t helper =
        init_helper_init_hook("", format, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = data = init_pencil();

    // initialize A

    if (left_input != NULL) {
        if (0 <= input_begin)
            data->mat_a = read_mtx_sub_matrix_from_file(
                input_begin, input_end, left_input, helper);
        else
            data->mat_a = read_mtx_matrix_from_file(left_input, helper);
    }
    else {
        data->mat_a = generate_random_fullpos(n, n, helper);
    }

    // initialize B

    if (right_input != NULL) {
        generalized = 1;
        if (0 <= input_begin)
            data->mat_b = read_mtx_sub_matrix_from_file(
                input_begin, input_end, right_input, helper);
        else
            data->mat_b = read_mtx_matrix_from_file(right_input, helper);
    }
    else if (generalized) {
        data->mat_b = generate_random_fullpos(n, n, helper);
    }

    // initialize Q and Z

    data->mat_q = generate_identity(n, n, helper);
    if (generalized)
        data->mat_z = generate_identity(n, n, helper);

    // prepare for decoupling

    if (decouple < 1 && set_to_inf < 1) {
        data->mat_ca = copy_matrix_descr(data->mat_a);
        data->mat_cb = copy_matrix_descr(data->mat_b);
    }

    // reduce

    if (format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {

        starneig_node_init(threads_get_workers(), STARNEIG_USE_ALL,
            STARNEIG_HINT_SM | STARNEIG_FXT_DISABLE);

        if (generalized)
            starneig_GEP_SM_HessenbergTriangular(LOCAL_MATRIX_N(data->mat_a),
                LOCAL_MATRIX_PTR(data->mat_a), LOCAL_MATRIX_LD(data->mat_a),
                LOCAL_MATRIX_PTR(data->mat_b), LOCAL_MATRIX_LD(data->mat_b),
                LOCAL_MATRIX_PTR(data->mat_q), LOCAL_MATRIX_LD(data->mat_q),
                LOCAL_MATRIX_PTR(data->mat_z), LOCAL_MATRIX_LD(data->mat_z));
        else
            starneig_SEP_SM_Hessenberg(LOCAL_MATRIX_N(data->mat_a),
                LOCAL_MATRIX_PTR(data->mat_a), LOCAL_MATRIX_LD(data->mat_a),
                LOCAL_MATRIX_PTR(data->mat_q), LOCAL_MATRIX_LD(data->mat_q));

        starneig_node_finalize();

    }

#ifdef STARNEIG_ENABLE_BLACS
    if (format == HOOK_DATA_FORMAT_PENCIL_BLACS) {

        starneig_node_init(threads_get_workers(), STARNEIG_USE_ALL,
            STARNEIG_HINT_DM | STARNEIG_FXT_DISABLE);

        if (generalized)
            starneig_GEP_DM_HessenbergTriangular(
                STARNEIG_MATRIX_HANDLE(data->mat_a),
                STARNEIG_MATRIX_HANDLE(data->mat_b),
                STARNEIG_MATRIX_HANDLE(data->mat_q),
                STARNEIG_MATRIX_HANDLE(data->mat_z));
        else
            starneig_SEP_DM_Hessenberg(
                STARNEIG_MATRIX_HANDLE(data->mat_a),
                STARNEIG_MATRIX_HANDLE(data->mat_q));

        starneig_node_finalize();
    }
#endif

    if (0 < decouple || 0 < set_to_inf)
        deflate_and_place_infinities(decouple, set_to_inf, data);

    init_helper_free(helper);

    printf("INIT FINISHED.\n");

    return env;
}

static const struct hook_initializer_t lapack_initializer = {
    .name = "lapack",
    .desc =
        "Reduces a matrix pencil to upper Hessenberg / Hessenberg-triangular "
        "form using a LAPACK algorithm",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .print_usage = &generic_initializer_print_usage,
    .print_args = &generic_initializer_print_args,
    .check_args = &generic_initializer_check_args,
    .init = &lapack_initializer_init
};

static const struct hook_initializer_t starpu_initializer = {
    .name = "starneig",
    .desc =
        "Reduces a matrix pencil to upper Hessenberg / Hessenberg-triangular "
        "form using a StarPU based parallel algorithm",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &generic_initializer_print_usage,
    .print_args = &generic_initializer_print_args,
    .check_args = &generic_initializer_check_args,
    .init = &starpu_initializer_init
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void print_usage(int argc, char * const *argv)
{
    print_avail_complex_distr();
    print_opt_complex_distr();
}

const struct hook_experiment_descr schur_experiment = {
    .print_usage = &print_usage,
    .initializers = (struct hook_initializer_t const *[])
    {
#ifdef GSL_FOUND
        &hessrand_initializer,
#endif
        &starpu_initializer,
        &lapack_initializer,
        &random_initializer,
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
        &schur_starpu_solver,
        &schur_starpu_simple_solver,
        &schur_lapack_solver,
#ifdef PDLAHQR_FOUND
        &schur_pdlahqr_solver,
#endif
#ifdef PDHSEQR_FOUND
        &schur_pdhseqr_solver,
#endif
#ifdef CUSTOM_PDHSEQR
        &schur_custom_pdhseqr_solver,
#endif
#ifdef PDHGEQZ_FOUND
        &schur_pdhgeqz_solver,
#endif
        0
    },
    .hook_descrs = (struct hook_descr_t const *[])
    {
        &default_schur_test_descr,
        &default_eigenvalues_descr,
        &default_known_eigenvalues_descr,
        &default_analysis_descr,
        &default_residual_test_descr,
        &default_print_pencil_descr,
        &default_print_input_pencil_descr,
        &default_store_raw_pencil_descr,
        &default_store_raw_input_pencil_descr,
        0
    }
};
