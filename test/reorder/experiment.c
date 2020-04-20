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
#include "../common/common.h"
#include "../common/parse.h"
#include "../common/local_pencil.h"
#include "../common/init_schur.h"
#include "../common/select_distr.h"
#include "../common/checks.h"
#include "../common/hooks.h"
#include "../common/io.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static double check_1x1_block(
    int i, int j, double const *real, double const *imag, double const *beta,
    double const *new_real, double const *new_imag, double const *new_beta,
    double fail_thres, double warning_thres, int *fail,
    int *warning, double *mean, double *min, double *max)
{
    double diff = 0.0;

    // 2-by-2 block in a place of a 1-by-1 block
    if (new_imag[j] != 0.0) {
        diff = ((long long)1<<52) *
            sqrt(
                squ(real[i]/beta[i]-new_real[j]/new_beta[j]) +
                squ(new_imag[j]/new_beta[j])
            ) / fabs(real[i]/beta[i]);

        printf(
            "REORDERING CHECK (FAILURE): Encountered a 2-by-2 block in a "
            "place of a 1-by-1 block at %d. The eigenvalue should have "
            "been %E (diff = %.0f u).\n",
            j, real[i]/beta[i], diff);
    }
    else {
        if (real[i] == 0.0 && beta[i] == 0.0) {
            diff = 0.0;
        }
        else if (real[i] == 0.0 && new_real[j] == 0.0) {
            diff = 0.0;
        }
        else if (beta[i] == 0.0 && new_beta[j] == 0.0) {
            diff = 0.0;
        }
        else {
            if (real[i] == 0.0)
                diff = ((long long)1<<52) * fabs(new_real[j]);
            else if (beta[i] == 0.0)
                diff = ((long long)1<<52) * fabs(new_beta[j]);
            else
                diff = ((long long)1<<52) *
                    fabs(
                        real[i]/beta[i]-new_real[j]/new_beta[j]
                    ) / fabs(real[i]/beta[i]);
        }

        if (warning_thres < diff || fail_thres < diff || isnan(diff)) {
            if (fail_thres < diff || isnan(diff)) {
                printf(
                    "REORDERING CHECK (FAILURE): An incorrect eigenvalue "
                    "at %d. The eigenvalue should have been %E "
                    "(diff = %.0f u)\n",
                    j, real[i]/beta[i], diff);
                (*fail)++;
            }
            else {
                (*warning)++;
                printf(
                    "REORDERING CHECK (WARNING): An incorrect eigenvalue "
                    "at %d. The eigenvalue should have been %E "
                    "(diff = %.0f u)\n", j,
                    real[i]/beta[i], diff);
            }
        }
    }

    return diff;
}

static double check_2x2_block(
    int i, int j, double const *real, double const *imag, double const *beta,
    double const *new_real, double const *new_imag, double const *new_beta,
    double fail_thres, double warning_thres, int *fail,
    int *warning, double *mean, double *min, double *max)
{
    double diff = 0.0;

    // 1-by-1 block in a place of a 2-by-2 block
    if (new_imag[j] == 0.0) {
        diff = ((long long)1<<52) *
            sqrt(
                squ(real[i]/beta[i]-new_real[j]/new_beta[j]) +
                squ(imag[i]/beta[i])
            ) / sqrt(squ(real[i]/beta[i])+squ(imag[i]/beta[i]));
        printf(
            "REORDERING CHECK (FAILURE): Encountered a 1-by-1 block in a "
            "place of a 2-by-2 block at %d. The eigenvalue should have been "
            "(%E,+-%E) (diff = %.0f u).\n",
            j, real[i]/beta[i], imag[i]/beta[i], diff);
    }
    else {
        diff = ((long long)1<<52) *
            sqrt(
                squ(real[i]/beta[i]-new_real[j]/new_beta[j]) +
                squ(imag[i]/beta[i]-new_imag[j]/new_beta[j])) /
            sqrt(squ(real[i]/beta[i])+squ(imag[i]/beta[i]));

        if (warning_thres < diff || fail_thres < diff || isnan(diff)) {
            if (fail_thres < diff || isnan(diff)) {
                (*fail)++;
                printf(
                    "REORDERING CHECK (FAILURE): An incorrect eigenvalue "
                    "pair at %d. The eigenvalue should have been "
                    "(%E,+-%E) (diff = %.0f u)\n",
                    j, real[i]/beta[i], imag[i]/beta[i], diff);
            }
            else {
                (*warning)++;
                printf(
                    "REORDERING CHECK (WARNING): An incorrect eigenvalue "
                    "pair at %d. The eigenvalue should have been "
                    "(%E,+-%E) (diff = %.0f u)\n",
                    j, real[i]/beta[i], imag[i]/beta[i], diff);
            }
        }
    }

    return diff;
}

static void check_eigenvalues(
    const pencil_t pencil, double const *real, double const *imag,
    double const *beta, double fail_thres, double warning_thres, int *fail,
    int *warning, double *mean, double *min, double *max)
{
    int n = GENERIC_MATRIX_N(pencil->mat_a);

    double *new_real = malloc(n*sizeof(double));
    double *new_imag = malloc(n*sizeof(double));
    double *new_beta = malloc(n*sizeof(double));

    extract_eigenvalues(
        pencil->mat_a, pencil->mat_b, new_real, new_imag, new_beta);

    int const *selected = get_supplementaty_selected(pencil->supp);

    *warning = 0;
    *fail = 0;
    double res[n];

    printf("REORDERING CHECK: Checking selected eigenvalues...\n");

    int m = 0, top = 0;
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            if (imag[i] != 0.0) {
                double diff = check_2x2_block(i, top, real, imag, beta,
                    new_real, new_imag, new_beta, fail_thres, warning_thres,
                    fail, warning, mean, min, max);
                res[m++] = diff;
                res[m++] = diff;
                top += 2;
                i++;
            } else {
                res[m++] = check_1x1_block(i, top, real, imag, beta,
                    new_real, new_imag, new_beta, fail_thres, warning_thres,
                    fail, warning, mean, min, max);
                top++;
            }
        }
    }

    printf("REORDERING CHECK: Checking other eigenvalues...\n");

    for (int i = 0; i < n; i++) {
        if (!selected[i]) {
            if (imag[i] != 0.0) {
                double diff = check_2x2_block(i, top, real, imag, beta,
                    new_real, new_imag, new_beta, fail_thres, warning_thres,
                    fail, warning, mean, min, max);
                res[m++] = diff;
                res[m++] = diff;
                top += 2;
                i++;
            } else {
                res[m++] = check_1x1_block(i, top, real, imag, beta,
                    new_real, new_imag, new_beta, fail_thres, warning_thres,
                    fail, warning, mean, min, max);
                top++;
            }
        }
    }


    qsort(res, m, sizeof(double), &double_compare);

    *mean = double_mean(m, res);
    *min = res[0];
    *max = res[m-1];

    printf("REORDERING CHECK: mean = %.0f u, min = %.0f u, max = %.0f u\n",
        *mean, *min, *max);

    free(new_real);
    free(new_imag);
    free(new_beta);
}

static const int reordering_test_default_fail_threshold = 10000;
static const int reordering_test_default_warn_threshold = 1000;

struct reordering_test_state {
    int repeat;             ///< experiment repetition count
    int fail_threshold;     ///< norm failure threshold
    int warn_threshold;     ///< norm warning threshold
    int *warning;
    int *fail;
    double *mean;
    double *min;
    double *max;
    double **real;
    double **imag;
    double **beta;
};

static void reordering_test_print_usage(int argc, char * const *argv)
{
    printf(
        "  --reordering-fail-threshold (num) -- Failure threshold\n"
        "  --reordering-warn-threshold (num) -- Warning threshold\n"
    );
}

static void reordering_test_print_args(int argc, char * const *argv)
{
    printf(" --reordering-fail-threshold %d",
        read_int("--reordering-fail-threshold", argc, argv, NULL,
            reordering_test_default_fail_threshold));
    printf(" --reordering-warn-threshold %d",
        read_int("--reordering-warn-threshold", argc, argv, NULL,
            reordering_test_default_warn_threshold));
}

static int reordering_test_check_args(int argc, char * const *argv, int *argr)
{
    int fail_threshold =
        read_int("--reordering-fail-threshold", argc, argv, argr,
            reordering_test_default_fail_threshold);
    int warn_threshold =
        read_int("--reordering-warn-threshold", argc, argv, argr,
            reordering_test_default_warn_threshold);

    if (fail_threshold < 0) {
        fprintf(stderr, "REORDERING CHECK: Invalid failure threshold.\n");
        return 1;
    }
    if (warn_threshold < 0) {
        fprintf(stderr, "REORDERING CHECK: Invalid warning threshold.\n");
        return 1;
    }

    if (fail_threshold < warn_threshold)
        fprintf(stderr,
            "REORDERING CHECK: The warning threshold is tighter than the "
            "failure threshold.\n");

    return 0;
}

static int reordering_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    struct reordering_test_state *t =
        malloc(sizeof(struct reordering_test_state));

    int fail_threshold =
        read_double("--reordering-fail-threshold", argc, argv, NULL,
            reordering_test_default_fail_threshold);
    int warn_threshold =
        read_double("--reordering-warn-threshold", argc, argv, NULL,
            reordering_test_default_warn_threshold);

    t->repeat = repeat;
    t->fail_threshold = fail_threshold;
    t->warn_threshold = warn_threshold;

    t->warning = malloc(repeat*sizeof(int));
    t->fail = malloc(repeat*sizeof(int));

    t->mean = malloc(repeat*sizeof(double));
    t->min = malloc(repeat*sizeof(double));
    t->max = malloc(repeat*sizeof(double));

    t->real = malloc(repeat*sizeof(double *));
    memset(t->real, 0, repeat*sizeof(double *));
    t->imag = malloc(repeat*sizeof(double *));
    memset(t->imag, 0, repeat*sizeof(double *));
    t->beta = malloc(repeat*sizeof(double *));
    memset(t->beta, 0, repeat*sizeof(double *));

    *state = t;

    return 0;
}

static int reordering_test_clean(hook_state_t state)
{
    struct reordering_test_state *t = state;

    if (t == NULL)
        return 0;

    free(t->warning);
    free(t->fail);
    free(t->mean);
    free(t->min);
    free(t->max);

    for (int i = 0; i < t->repeat; i++) {
        free(t->real[i]);
        free(t->imag[i]);
        free(t->beta[i]);
    }
    free(t->real);
    free(t->imag);
    free(t->beta);

    free(t);

    return 0;
}

static hook_return_t reordering_test_after_data_init(
    int iter, hook_state_t state, struct hook_data_env *env)
{
     if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct reordering_test_state *t = state;

    t->real[iter] = malloc(GENERIC_MATRIX_N(pencil->mat_a)*sizeof(double));
    t->imag[iter] = malloc(GENERIC_MATRIX_N(pencil->mat_a)*sizeof(double));
    t->beta[iter] = malloc(GENERIC_MATRIX_N(pencil->mat_a)*sizeof(double));

    extract_eigenvalues(pencil->mat_a, pencil->mat_b,
        t->real[iter], t->imag[iter], t->beta[iter]);

    return HOOK_SUCCESS;
}

static hook_return_t reordering_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct reordering_test_state *t = state;

    check_eigenvalues(pencil, t->real[iter], t->imag[iter],  t->beta[iter],
        t->fail_threshold, t->warn_threshold,
        &t->fail[iter], &t->warning[iter],
        &t->mean[iter], &t->min[iter], &t->max[iter]);

    if (0 < t->fail[iter])
        printf("REORDERING CHECK: %d eigenvalue failures.\n", t->fail[iter]);

    if (0 < t->warning[iter])
        printf("REORDERING CHECK: %d eigenvalue warnings.\n", t->warning[iter]);

    if (0 < t->fail[iter])
        return HOOK_SOFT_FAIL;

    if (0 < t->warning[iter])
        return HOOK_WARNING;

    return HOOK_SUCCESS;
}

static hook_return_t reordering_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct reordering_test_state *t = state;

    qsort(t->warning, t->repeat, sizeof(int), &int_compare);
    qsort(t->fail, t->repeat, sizeof(int), &int_compare);
    qsort(t->mean, t->repeat, sizeof(double), &double_compare);
    qsort(t->min, t->repeat, sizeof(double), &double_compare);
    qsort(t->max, t->repeat, sizeof(double), &double_compare);

    int warning_runs = 0;
    int fail_runs = 0;
    for (int i = 0; i < t->repeat; i++) {
        if (0 < t->warning[i]) warning_runs++;
        if (0 < t->fail[i]) fail_runs++;
    }

    printf("REORDERING CHECK (WARNINGS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        warning_runs, int_mean(t->repeat, t->warning),
        int_cv(t->repeat, t->warning), t->warning[0], t->warning[t->repeat-1]);
    printf("REORDERING CHECK (FAILS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        fail_runs, int_mean(t->repeat, t->fail), int_cv(t->repeat, t->fail),
        t->fail[0], t->fail[t->repeat-1]);
    printf(
        "REORDERING CHECK (MEANS): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->repeat, t->mean), double_cv(t->repeat, t->mean),
        t->mean[0], t->mean[t->repeat-1]);
    printf("REORDERING CHECK (MIN): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->repeat, t->min), double_cv(t->repeat, t->min),
        t->min[0], t->min[t->repeat-1]);
    printf("REORDERING CHECK (MAX): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->repeat, t->max), double_cv(t->repeat, t->max),
        t->max[0], t->max[t->repeat-1]);

    return HOOK_SUCCESS;
}

const struct hook_t reordering_test = {
    .name = "reordering",
    .desc = "Eigenvalue reordering check",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &reordering_test_print_usage,
    .print_args = &reordering_test_print_args,
    .check_args = &reordering_test_check_args,
    .init = &reordering_test_init,
    .clean = &reordering_test_clean,
    .after_data_init = &reordering_test_after_data_init,
    .after_solver_run = &reordering_test_after_solver_run,
    .summary = &reordering_test_summary
};

const struct hook_descr_t default_reordering_test_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &reordering_test
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const struct hook_experiment_descr reorder_experiment = {
    .print_usage = &schur_initializer_print_usage,
    .initializers = (struct hook_initializer_t const *[])
    {
        &default_schur_initializer,
        &starpu_schur_initializer,
        &raw_initializer,
        0
    },
    .supplementers = (struct hook_supplementer_t const *[])
    {
        &selection_supplementer,
        0
    },
    .solvers = (struct hook_solver const *[])
    {
        &reorder_starpu_solver,
        &reorder_starpu_simple_solver,
        &reorder_lapack_solver,
#ifdef PDTRSEN_FOUND
        &reorder_scalapack_solver,
#endif
        0
    },
    .hook_descrs = (struct hook_descr_t const *[])
    {
        &default_schur_test_descr,
        &default_eigenvalues_descr,
        &default_analysis_descr,
        &default_reordering_test_descr,
        &default_residual_test_descr,
        &default_print_pencil_descr,
        &default_print_input_pencil_descr,
        &default_store_raw_pencil_descr,
        &default_store_raw_input_pencil_descr,
        0
    }
};
