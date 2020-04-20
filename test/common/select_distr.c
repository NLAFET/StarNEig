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
#include "select_distr.h"
#include "common.h"
#include "parse.h"
#include "checks.h"
#include "crawler.h"
#include "hook_experiment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double const default_ratio = 0.35;
static double const default_cluster_min_size = 5;
static double const default_cluster_max_size = 45;

static int select_block(int i, int n, int const *blocks, int *select)
{
    if (select[i])
        return 0;
    if (blocks[i])
        return 0;

    select[i] = 1;
    if (i+1 < n && blocks[i+1]) {
        select[i+1] = 1;
        return 2;
    }

    return 1;
}

static int find_blocks_crawler(
    int offset, int size, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    int *blocks = arg;

    double const *A = ptrs[0];
    size_t ldA = lds[0];

    if (offset == 0)
        blocks[0] = 0;

    int i = 1;
    while (i < size) {
        blocks[offset+i] = A[(i-1)*ldA+i] != 0.0;
        i++;
    }

    return offset+size == n ? size : i-1;
}

static int * find_blocks(const pencil_t pencil)
{
    int n = GENERIC_MATRIX_M(pencil->mat_a);
    int *blocks = malloc(n*sizeof(int));
    crawl_matrices(CRAWLER_R, CRAWLER_DIAG_WINDOW,
        &find_blocks_crawler, blocks, n*sizeof(int), pencil->mat_a, NULL);

    return blocks;
}

////////////////////////////////////////////////////////////////////////////////

static void uniform_select_print_usage()
{
    printf(
        "  --select-ratio (0.0-1.0) -- Selection ratio\n");
}

static int uniform_select_check_args(
    int argc, char * const *argv, int *argr)
{
    double select_ratio = read_double(
        "--select-ratio", argc, argv, argr, default_ratio);
    if (select_ratio < 0.0 || 1.0 < select_ratio)
        return -1;
    return 0;
}

static void uniform_select_print_args(int argc, char * const *argv)
{
    double select_ratio = read_double(
        "--select-ratio", argc, argv, NULL, default_ratio);
    printf(" --select-ratio %f", select_ratio);
}

static int uniform_select_distr_init(
    int argc, char * const * argv, const pencil_t pencil, int *select)
{
    double select_ratio = read_double(
        "--select-ratio", argc, argv, NULL, default_ratio);

    int n = GENERIC_MATRIX_N(pencil->mat_a);

    for (int i = 0; i < n; i++)
        select[i] = 0;

    int *blocks = find_blocks(pencil);

    int selected = 0;
    while(selected < select_ratio*n)
        selected += select_block(prand() % n, n, blocks, select);

    free(blocks);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////

static void bulk_select_print_usage()
{
    printf(
        "  --bulk-select-begin [middle,(num)] -- Select all diagonal "
        "entries between begin and end\n"
        "  --bulk-select-end [bottom,(num)] -- Select all diagonal "
        "entries between begin and end\n"
    );
}

static int bulk_select_check_args(
    int argc, char * const *argv, int *argr)
{
    struct multiarg_t begin = read_multiarg(
        "--bulk-select-begin", argc, argv, argr, "middle", NULL);
    struct multiarg_t end = read_multiarg(
        "--bulk-select-end", argc, argv, argr, "bottom", NULL);

    if (begin.type == MULTIARG_INVALID ||
    (begin.type == MULTIARG_INT && begin.int_value < 0)) {
        fprintf(stderr, "Invalid --bulk-select-begin value.\n");
        return -1;
    }

    if (end.type == MULTIARG_INVALID ||
    (end.type == MULTIARG_INT && end.int_value < 0)) {
        fprintf(stderr, "Invalid --bulk-select-end value.\n");
        return -1;
    }

    return 0;
}

static void bulk_select_print_args(int argc, char * const *argv)
{
    print_multiarg("--bulk-select-begin", argc, argv, "middle", NULL);
    print_multiarg("--bulk-select-end", argc, argv, "bottom", NULL);
}

static int bulk_select_init(
    int argc, char * const * argv, const pencil_t pencil, int *select)
{
    int n = GENERIC_MATRIX_N(pencil->mat_a);

    int begin = 0, end = n;

    struct multiarg_t begin_arg = read_multiarg(
        "--bulk-select-begin", argc, argv, NULL, "middle", NULL);
    struct multiarg_t end_arg = read_multiarg(
        "--bulk-select-end", argc, argv, NULL, "bottom", NULL);

    if (begin_arg.type == MULTIARG_STR &&
    strcmp("middle", begin_arg.str_value) == 0)
        begin = n/2;
    if (begin_arg.type == MULTIARG_INT)
        begin = MIN(n, begin_arg.int_value);

    if (end_arg.type == MULTIARG_STR &&
    strcmp("bottom", end_arg.str_value) == 0)
        end = n;
    if (end_arg.type == MULTIARG_INT)
        end = MIN(n, MAX(begin, end_arg.int_value));

    for (int i = 0; i < n; i++)
        select[i] = 0;

    int *blocks = find_blocks(pencil);

    int i = begin;
    while (i < end)
        i += select_block(i, n, blocks, select);

    free(blocks);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////

static void cluster_select_print_usage()
{
    printf(
        "  --select-ratio (0.0-1.0) -- Selection ratio\n"
        "  --cluster-min-size (num) -- Minimum cluster size\n"
        "  --cluster-max-size (num) -- Maximum cluster size\n");
}

static int cluster_select_check_args(
    int argc, char * const *argv, int *argr)
{
    double select_ratio = read_double("--select-ratio", argc, argv, argr,
        default_ratio);
    int min_size = read_int("--cluster-min-size", argc, argv, argr,
        default_cluster_min_size);
    int max_size = read_int("--cluster-max-size", argc, argv, argr,
        default_cluster_max_size);

    if (select_ratio < 0.0 || 1.0 < select_ratio)
        return -1;

    if (min_size < 0 || max_size <= min_size)
        return -1;

    return 0;
}

static void cluster_select_print_args(int argc, char * const *argv)
{
    printf(" --select-ratio %f --cluster-min-size %d --cluster-max-size %d",
        read_double("--select-ratio", argc, argv, NULL, default_ratio),
        read_int("--cluster-min-size", argc, argv, NULL,
            default_cluster_min_size),
        read_int("--cluster-max-size", argc, argv, NULL,
            default_cluster_max_size));
}

static int cluster_select_init(
    int argc, char * const * argv, const pencil_t pencil, int *select)
{
    double select_ratio = read_double("--select-ratio", argc, argv, NULL,
        default_ratio);
    int min_size = read_int("--cluster-min-size", argc, argv, NULL,
        default_cluster_min_size);
    int max_size = read_int("--cluster-max-size", argc, argv, NULL,
        default_cluster_max_size);

    int n = GENERIC_MATRIX_N(pencil->mat_a);

    for (int i = 0; i < n; i++)
        select[i] = 0;

    int *blocks = find_blocks(pencil);

    int selected = 0;
    while (selected < select_ratio*n) {
        int size = MIN(select_ratio*n - selected,
            prand() % (max_size-min_size) + min_size);
        int begin = prand() % (n-size);
        int end = MIN(n, begin+size);

        if (0 < begin && select[begin-1])
            continue;

        if (end < n && select[end])
            continue;

        for (int i = begin; i < end && selected < select_ratio*n; i++)
            selected += select_block(i, n, blocks, select);
    }

    free(blocks);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////

static void find_pos_real_blocks(const pencil_t pencil, int *select)
{
    int n = GENERIC_MATRIX_M(pencil->mat_a);

    double *real = malloc(n*sizeof(double));
    double *imag = malloc(n*sizeof(double));
    double *beta = malloc(n*sizeof(double));

    extract_eigenvalues(pencil->mat_a, pencil->mat_b, real, imag, beta);

    for (int i = 0; i < n; i++) {
        if (imag[i] != 0.0) {
            if (beta[i] != 0.0 && 0.0 < real[i]) {
                select[i] = 1;
                select[i+1] = 1;
            }
            else {
                select[i] = 0;
                select[i+1] = 0;
            }
            i++;
        }
        else {
            if (beta[i] != 0.0 && 0.0 < real[i])
                select[i] = 1;
            else
                select[i] = 0;
        }
    }

    free(real);
    free(imag);
    free(beta);
}

static int pos_real_select_init(
    int argc, char * const * argv, const pencil_t pencil, int *select)
{
    find_pos_real_blocks(pencil, select);
    return 0;
}

static const struct select_distr {
    char const *name;   ///< module name
    char const *desc;   ///< module description

    ///
    /// @brief Prints usage information.
    ///
    void (*print_usage)();

    ///
    /// @brief Checks the command line arguments.
    ///
    /// @param[in] argc     command line argument count
    /// @param[in] argv     command line arguments
    /// @param[inout] argr  array that tracks which command line arguments have
    ///                     been processed
    ///
    /// @return 0 if the command line arguments are valid, non-zero otherwise
    ///
    int (*check_args)(int argc, char * const *argv, int *argr);

    ///
    /// @brief Prints active command line arguments.
    ///
    /// @param[in] argc     command line argument count
    /// @param[in] argv     command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Generates an eigenvalue selection vector.
    ///
    /// @param[in] argc         command line argument count
    /// @param[in] argv         command line arguments
    /// @param[in] pencil       matrix pencil
    /// @param[inout] selected  eigenvalue selection vector
    ///
    /// @return 0 when the function call was successful, non-zero otherwise
    ///
    int (*init)(
        int argc, char * const * argv, const pencil_t pencil, int *select);
} select_distrs[] = {
    { .name = "uniform",
        .desc = "Uniform eigenvalue selection module",
        .print_usage = &uniform_select_print_usage,
        .check_args = &uniform_select_check_args,
        .print_args = &uniform_select_print_args,
        .init = &uniform_select_distr_init
    },
    { . name = "cluster",
        .desc = "Cluster eigenvalue selection module",
        .print_usage = &cluster_select_print_usage,
        .check_args = &cluster_select_check_args,
        .print_args = &cluster_select_print_args,
        .init = &cluster_select_init
    },
    { .name = "bulk",
        .desc = "Bulk eigenvalue selection module",
        .print_usage = &bulk_select_print_usage,
        .check_args = &bulk_select_check_args,
        .print_args = &bulk_select_print_args,
        .init = &bulk_select_init
    },
    { .name = "positive",
        .desc = "Positive real part eigenvalue selection module",
        .init = &pos_real_select_init
    },
};

static int select_destr_default = 0;
static int select_distr_count = sizeof(select_distrs)/sizeof(select_distrs[0]);

static struct select_distr const * read_select_distr(
    char const *name, int argc, char * const *argv, int *argr)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            for (int j = 0; j < select_distr_count; j++) {
                if (strcmp(select_distrs[j].name, argv[i+1]) == 0) {
                    if (argr != NULL)
                        argr[i] = argr[i+1] = 1;
                    return &select_distrs[j];
                }
            }
            return NULL;
        }
    }

    return &select_distrs[select_destr_default];
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void print_usage(int argc, char * const *argv)
{
    printf(
        "  --select-distr (select distribution) -- Eigenvalue selection "
        "module\n"
    );

    printf(
        "\n"
        "Available eigenvalue selection modules:\n");
    for (int i = 0; i < select_distr_count; i++)
        printf("    [%s]: %s%s\n",
            select_distrs[i].name, select_distrs[i].desc,
            i == select_destr_default ? " (default)" : "");

    for (int i = 0; i < select_distr_count; i++) {
        if (select_distrs[i].print_usage != NULL) {
            printf("\n%s eigenvalue selection module specific options:\n",
                select_distrs[i].name);
            select_distrs[i].print_usage();
        }
    }
}

static void print_args(int argc, char * const *argv)
{
    struct select_distr const *select_distr =
        read_select_distr("--select-distr", argc, argv, NULL);
    if (select_distr->print_args != NULL)
        select_distr->print_args(argc, argv);
}

static int check_args(int argc, char * const *argv, int *argr)
{
    struct select_distr const *select_distr =
        read_select_distr("--select-distr", argc, argv, argr);
    if (select_distr == NULL) {
        fprintf(stderr, "Invalid eigenvalue selection module.\n");
        return -1;
    }

    if (select_distr->check_args != NULL) {
        int ret = select_distr->check_args(argc, argv, argr);
        if (ret)
            return ret;
    }

    return 0;
}

static void supplement(struct hook_data_env *env, int argc, char * const *argv)
{
    pencil_t pencil = env->data;
    int n = GENERIC_MATRIX_N(pencil->mat_a);

    struct select_distr const *select_distr =
        read_select_distr("--select-distr", argc, argv, NULL);

    int *selected;
    init_supplementary_selected(n, &selected, &pencil->supp);

    memset(selected, 0, n*sizeof(int));
    if (select_distr->init != NULL)
        select_distr->init(argc, argv, pencil, selected);
}

const struct hook_supplementer_t selection_supplementer = {
    .name = "select",
    .desc = "Eigenvalue selection supplementer",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0
    },
    .print_usage = &print_usage,
    .print_args = &print_args,
    .check_args = &check_args,
    .supplement = &supplement
};
