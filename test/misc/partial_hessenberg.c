///
/// @file
///
/// @brief This file contains an experiment for partial Hessenberg reduction.
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
#include "partial_hessenberg.h"
#include "../common/parse.h"
#include "../common/local_pencil.h"
#include "../common/init.h"
#include "../common/checks.h"
#include <starneig/starneig.h>
#include <stdio.h>

static const int fail_threshold = 1000;

void partial_hessenberg_print_usage(
    int argc, char * const *argv, experiment_info_t const info)
{
    printf(
        "  --n (num) -- Problem dimension\n"
        "  --begin (num) -- First unreduced row/column\n"
        "  --end (num)  -- Last unreduced row/column + 1\n"
        "  --cores [default,(num)] -- Number of CPU cores\n"
        "  --gpus [default,(num)] -- Number of GPUS\n"
        "  --tile-size [default,(num)] -- Tile size\n"
        "  --panel-width [default,(num)] -- Panel width\n"
    );
}

void partial_hessenberg_print_args(
    int argc, char * const *argv, experiment_info_t const info)
{
    printf(" --n %d", read_int("--n", argc, argv, NULL, -1));
    printf(" --begin %d", read_int("--begin", argc, argv, NULL, -1));
    printf(" --end %d", read_int("--end", argc, argv, NULL, -1));
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
    print_multiarg("--tile-size", argc, argv, "default", NULL);
    print_multiarg("--panel-width", argc, argv, "default", NULL);
}

int partial_hessenberg_check_args(
    int argc, char * const *argv, int *argr, experiment_info_t const info)
{
    int n = read_int("--n", argc, argv, argr, -1);
    int begin = read_int("--begin", argc, argv, argr, -1);
    int end = read_int("--end", argc, argv, argr, -1);

    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);
    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, argr, "default", NULL);
    struct multiarg_t panel_width = read_multiarg(
        "--panel-width", argc, argv, argr, "default", NULL);

    if (n < 1 || begin < 0 || end < begin || n < end)
        return 1;

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    if (tile_size.type == MULTIARG_INVALID ||
    (tile_size.type == MULTIARG_INT && tile_size.int_value < 1)) {
        fprintf(stderr, "Invalid tile size.\n");
        return -1;
    }

    if (panel_width.type == MULTIARG_INVALID ||
    (panel_width.type == MULTIARG_INT && panel_width.int_value < 1)) {
        fprintf(stderr, "Invalid panel width.\n");
        return -1;
    }

    return 0;
}

int partial_hessenberg_run(
    int argc, char * const *argv, experiment_info_t const info)
{
    int n = read_int("--n", argc, argv, NULL, -1);
    int begin = read_int("--begin", argc, argv, NULL, -1);
    int end = read_int("--end", argc, argv, NULL, -1);
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, NULL, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, NULL, "default", NULL);
    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, NULL, "default", NULL);
    struct multiarg_t panel_width = read_multiarg(
        "--panel-width", argc, argv, NULL, "default", NULL);

    int cores = STARNEIG_USE_ALL;
    if (arg_cores.type == MULTIARG_INT)
        cores = arg_cores.int_value;

    int gpus = STARNEIG_USE_ALL;
    if (arg_gpus.type == MULTIARG_INT)
        gpus = arg_gpus.int_value;

    init_helper_t helper = init_helper_init(
        "", LOCAL_MATRIX, n, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    pencil_t pencil = init_pencil();
    pencil->mat_a = generate_random_uptriag(n, n, helper);
    pencil->mat_q = generate_identity(n, n, helper);

    init_helper_free(helper);

    {
        double *A = (double *) LOCAL_MATRIX_PTR(pencil->mat_a);
        int ldA = LOCAL_MATRIX_LD(pencil->mat_a);

        for (int i = begin; i < end-1; i++)
            for (int j = i+1; j < end; j++)
                A[(size_t)i*ldA+j] = 2.0*(1.0*prand()/PRAND_MAX)-1.0;
    }

    fill_pencil(pencil);

    starneig_node_init(cores, gpus, STARNEIG_HINT_SM);

    struct starneig_hessenberg_conf conf;
    starneig_hessenberg_init_conf(&conf);

    if (tile_size.type == MULTIARG_INT)
        conf.tile_size = tile_size.int_value;
    if (panel_width.type == MULTIARG_INT)
        conf.panel_width = panel_width.int_value;

    starneig_SEP_SM_Hessenberg_expert(&conf,
        LOCAL_MATRIX_N(pencil->mat_a), begin, end,
        LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
        LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q));

    starneig_node_finalize();

    int failed = 0;
    {
        double *A = (double *) LOCAL_MATRIX_PTR(pencil->mat_a);
        int ldA = LOCAL_MATRIX_LD(pencil->mat_a);

        for (int i = 0; i < n-1; i++) {
            int k = begin <= i && i < end-1 ? 2 : 1;
            int first_failed = n;
            for (int j = i+k; j < n; j++) {
                if (A[(size_t)i*ldA+j] != 0.0) {
                    if (failed == 0)
                        printf("FAILED COLUMNS AT:");
                    if (j < first_failed) {
                        printf(" (%d,%d", i, j);
                        first_failed = j;
                    }
                    failed++;
                }
                else if (first_failed < j-1) {
                    printf("-%d)", j-1);
                    first_failed = n;
                }
                else if (first_failed < j) {
                    printf(")");
                    first_failed = n;
                }
            }
            if (first_failed == n-1)
                printf("-%d]", n-1);
            else if (first_failed < n)
                printf("]");
        }
        if (0 < failed)
            printf("\n");
        else
            printf("NO FAILED COLUMNS\n");
    }

    printf("|Q ~A Q^T - A| / |A|");
    fflush(stdout);

    double res_a = compute_qazt_c_norm(
        pencil->mat_q, pencil->mat_a, pencil->mat_q, pencil->mat_ca);

    printf(" = %.0f u\n", res_a);

    printf("|Q Q^T - I| / |I|");
    fflush(stdout);

    double res_q = compute_qqt_norm(pencil->mat_q);

    printf(" = %.0f u\n", res_q);

    free_pencil(pencil);

    return 0 < failed || fail_threshold < res_a || fail_threshold < res_q;
}
