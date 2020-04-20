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

static const int eigenvectors_default_fail_threshold = 10000;
static const int eigenvectors_default_warn_threshold = 1000;

struct eigenvectors_test_state {
    int repeat;             ///< experiment repetition count
    int fail_threshold;     ///< failure threshold
    int warn_threshold;     ///< warning threshold
    int *warning;
    int *fail;
    double *min;
    double *max;
    double *mean;
};

static void eigenvectors_test_print_usage(int argc, char * const *argv)
{
    printf(
        "  --eigenvectors-fail-threshold (num) -- Failure threshold\n"
        "  --eigenvectors-warn-threshold (num) -- Warning threshold\n"
    );
}

static void eigenvectors_test_print_args(int argc, char * const *argv)
{
    printf(" --eigenvectors-fail-threshold %d",
        read_int("--eigenvectors-fail-threshold", argc, argv, NULL,
            eigenvectors_default_fail_threshold));
    printf(" --eigenvectors-warn-threshold %d",
        read_int("--eigenvectors-warn-threshold", argc, argv, NULL,
            eigenvectors_default_warn_threshold));
}

static int eigenvectors_test_check_args(int argc, char * const *argv, int *argr)
{
    double fail_threshold =
        read_int("--eigenvectors-fail-threshold", argc, argv, argr,
            eigenvectors_default_fail_threshold);
    double warn_threshold =
        read_int("--eigenvectors-warn-threshold", argc, argv, argr,
            eigenvectors_default_warn_threshold);

    if (fail_threshold < 0) {
        fprintf(stderr, "Invalid failure threshold\n");
        return 1;
    }
    if (warn_threshold < 0) {
        fprintf(stderr, "Invalid warning threshold\n");
        return 1;
    }

    return 0;
}

static int eigenvectors_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    struct eigenvectors_test_state *t =
        malloc(sizeof(struct eigenvectors_test_state));

    int fail_threshold =
        read_double("--eigenvectors-fail-threshold", argc, argv, NULL,
            eigenvectors_default_fail_threshold);

    int warn_threshold =
        read_double("--eigenvectors-warn-threshold", argc, argv, NULL,
            eigenvectors_default_warn_threshold);

    t->repeat = repeat;
    t->fail_threshold = fail_threshold;
    t->warn_threshold = warn_threshold;

    t->warning = malloc(repeat*sizeof(t->warning[0]));
    t->fail = malloc(repeat*sizeof(t->fail[0]));

    t->min = malloc(repeat*sizeof(t->min[0]));
    t->max = malloc(repeat*sizeof(t->max[0]));
    t->mean = malloc(repeat*sizeof(t->mean[0]));

    *state = t;
    return 0;
}

static int eigenvectors_test_clean(hook_state_t state)
{
    struct eigenvectors_test_state *t = state;

    if (t == NULL)
        return 0;

    free(t->warning);
    free(t->fail);
    free(t->min);
    free(t->max);
    free(t->mean);
    free(t);

    return 0;
}

static hook_return_t eigenvectors_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    struct eigenvectors_test_state *t = state;

    pencil_t pencil = (pencil_t) env->data;

    fill_pencil(pencil);

    double *A = LOCAL_MATRIX_PTR(pencil->mat_a);
    size_t ldA = LOCAL_MATRIX_LD(pencil->mat_a);
    int n = LOCAL_MATRIX_N(pencil->mat_a);

    double *B = NULL; size_t ldB = 0;
    if (pencil->mat_b != NULL) {
        B = LOCAL_MATRIX_PTR(pencil->mat_b);
        ldB = LOCAL_MATRIX_LD(pencil->mat_b);
    }

    double *X = LOCAL_MATRIX_PTR(pencil->mat_x);
    size_t ldX = LOCAL_MATRIX_LD(pencil->mat_x);

    // CA * X
    matrix_t left = NULL;
    mul_C_AB("N", "N", 1.0, pencil->mat_ca, pencil->mat_x, 0.0, &left);

    double *L = LOCAL_MATRIX_PTR(left);
    size_t ldL = LOCAL_MATRIX_LD(left);

    // CB * X
    matrix_t right = NULL;
    if (pencil->mat_cb != NULL)
        mul_C_AB("N", "N", 1.0, pencil->mat_cb, pencil->mat_x, 0.0, &right);
    else
        right = pencil->mat_x;

    double *R = LOCAL_MATRIX_PTR(right);
    size_t ldR = LOCAL_MATRIX_LD(right);

    // norm of the matrix CA
    double norm_ca = norm_C(pencil->mat_ca);

    // norm of the matrix CB
    double norm_cb = 1.0;
    if (B != NULL)
        norm_cb = norm_C(pencil->mat_cb);

    int const *selected = get_supplementaty_selected(pencil->supp);

    t->warning[iter] = 0;
    t->fail[iter] = 0;

    t->min[iter] = 1.0/0.0;
    t->max[iter] = 0.0;
    double sum = 0.0;

    int p = 0;
    for (int i = 0; i < n; i++) {
        if (selected[i]) {

            //
            // 2-by-2 block
            //

            if (i+1 < n && A[i*ldA+i+1] != 0.0) {

                double real1, imag1, real2, imag2, beta1, beta2;
                compute_complex_eigenvalue(
                    ldA, ldB, &A[i*ldA+i], B != NULL ? &B[i*ldB+i] : NULL,
                    &real1, &imag1, &real2, &imag2, &beta1, &beta2);

                // just to be sure that nothing weird has not happened
                if (real1 != real2 || imag1 == 0.0 || imag1 != -imag2) {
                    fprintf(stderr, "EIGENVECTOR CHECK: Invalid matrix.\n");
                    return HOOK_HARD_FAIL;
                }

                double norm = 0.0;      // norm "acculator"
                double norm_x = 0.0;    // norm of the eigenvector
                for (int j = 0; j < n; j++) {
                    double lr = beta1 * L[p*ldL+j];
                    double li = beta1 * L[(p+1)*ldL+j];
                    double rr =
                        real1 * R[p*ldR+j] - imag1 * R[(p+1)*ldR+j];
                    double ri =
                        imag1 * R[p*ldR+j] + real1 * R[(p+1)*ldR+j];

                    norm += squ(lr-rr) + squ(li-ri);
                    norm_x += squ(X[p*ldX+j]) + squ(X[(p+1)*ldX+j]);
                }
                norm = ((long long)1<<52) * sqrt(norm) /
                    (sqrt(norm_x) * (beta1 * norm_ca + sqrt(squ(real1) + squ(imag1)) * norm_cb));

                if (t->fail_threshold <= norm || isinf(norm) || isnan(norm)) {
                    fprintf(stderr,
                        "EIGENVECTOR CHECK (FAILURE): Eigenvector pair "
                        "%d,%d residual %.0f u is above failure threshold.\n",
                        i, i+1, norm);
                    t->fail[iter]++;
                }
                else if (t->warn_threshold <= norm) {
                    fprintf(stderr,
                        "EIGENVECTOR CHECK (WARNING): Eigenvector pair "
                        "%d,%d residual %.0f u is above warning threshold.\n",
                        i, i+1, norm);
                    t->warning[iter]++;
                }

                t->min[iter] = MIN(t->min[iter], norm);
                t->max[iter] = MAX(t->max[iter], norm);
                sum += 2 * norm; // two vectors

                p += 2;
                i++;
            }

            //
            // 1-by-1 block
            //

            else {

                double real = A[i*ldA+i];
                double beta = 1.0;
                if (B != NULL) {
                    if (B[i*ldB+i] < 0.0) {
                        real = -real;
                        beta = -B[i*ldB+i];
                    }
                    else {
                        beta = B[i*ldB+i];
                    }
                }

                // norm of the eigenvector
                double norm_x = 0.0;
                for (int j = 0; j < n; j++)
                    norm_x += squ(X[p*ldX+j]);
                norm_x = sqrt(norm_x);

                double norm = 0.0;  // norm "acculator"

                if (real == 0.0 && beta != 0.0) {
                    for (int j = 0; j < n; j++)
                        norm += squ(L[p*ldL+j]);
                    norm = ((long long)1<<52) * sqrt(norm) / (norm_ca * norm_x);

                    if (t->fail_threshold <= norm || isinf(norm) || isnan(norm))
                    {
                        fprintf(stderr,
                            "EIGENVECTOR CHECK (FAILURE): Eigenvector %d "
                            "should be in kernel of A but residual %.0f u "
                            "is above failure threshold.\n", i, norm);
                        t->fail[iter]++;
                    }
                    else if (t->warn_threshold <= norm) {
                        fprintf(stderr,
                            "EIGENVECTOR CHECK (WARNING): Eigenvector %d "
                            "should be in kernel of A but residual %.0f u "
                            "is above warning threshold.\n", i, norm);
                        t->warning[iter]++;
                    }
                }
                else if (real != 0.0 && beta == 0.0) {
                    for (int j = 0; j < n; j++)
                        norm += squ(R[p*ldR+j]);
                    norm = ((long long)1<<52) * sqrt(norm) / (norm_cb * norm_x);

                    if (t->fail_threshold <= norm || isinf(norm) || isnan(norm))
                    {
                        fprintf(stderr,
                            "EIGENVECTOR CHECK (FAILURE): Eigenvector %d "
                            "should be in kernel of B but residual %.0f u "
                            "is above failure threshold.\n", i, norm);
                        t->fail[iter]++;
                    }
                    else if (t->warn_threshold <= norm) {
                        fprintf(stderr,
                            "EIGENVECTOR CHECK (WARNING): Eigenvector %d "
                            "should be in kernel of B but residual %.0f u "
                            "is above warning threshold.\n", i, norm);
                        t->warning[iter]++;
                    }
                }
                else if (real != 0.0 && beta != 0.0) {
                    for (int j = 0; j < n; j++)
                        norm += squ(beta * L[p*ldR+j] - real * R[p*ldL+j]);
                    norm = ((long long)1<<52) * sqrt(norm) /
                        (norm_x * (beta * norm_ca + fabs(real) * norm_cb));

                    if (t->fail_threshold <= norm || isinf(norm) || isnan(norm))
                    {
                        fprintf(stderr,
                            "EIGENVECTOR CHECK (FAILURE): Eigenvector %d "
                            "residual %.0f u is above failure threshold.\n",
                            i, norm);
                        t->fail[iter]++;
                    }
                    else if (t->warn_threshold <= norm) {
                        fprintf(stderr,
                            "EIGENVECTOR CHECK (WARNING): Eigenvector %d "
                            "residual %.0f u is above warning threshold.\n",
                            i, norm);
                        t->warning[iter]++;
                    }
                }

                t->min[iter] = MIN(t->min[iter], norm);
                t->max[iter] = MAX(t->max[iter], norm);
                sum += norm;

                p++;
            }
        }
    }

    t->mean[iter] = sum / p;

    printf("EIGENVECTOR CHECK: MEAN = %.2f u, MIN = %.2f u, MAX = %.2f u\n",
        t->mean[iter], t->min[iter], t->max[iter]);

    free_matrix_descr(left);
    if (right != pencil->mat_x)
        free_matrix_descr(right);

    if (0 < t->fail[iter])
        return HOOK_SOFT_FAIL;
    if (0 < t->warning[iter])
        return HOOK_WARNING;
    return HOOK_SUCCESS;
}

static hook_return_t eigenvectors_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct eigenvectors_test_state *t = state;

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

    printf(
        "EIGENVECTORS (WARNINGS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        warning_runs, int_mean(t->repeat, t->warning),
        int_cv(t->repeat, t->warning), t->warning[0], t->warning[t->repeat-1]);
    printf(
        "EIGENVECTORS (FAILS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        fail_runs, int_mean(t->repeat, t->fail), int_cv(t->repeat, t->fail),
        t->fail[0], t->fail[t->repeat-1]);
    printf(
        "EIGENVECTORS (MEANS): [avg %.2f u, cv %.2f, min %.2f u, max %.2f u]\n",
        double_mean(t->repeat, t->mean), double_cv(t->repeat, t->mean),
        t->mean[0], t->mean[t->repeat-1]);
    printf(
        "EIGENVECTORS (MIN): [avg %.2f u, cv %.2f, min %.2f u, max %.2f u]\n",
        double_mean(t->repeat, t->min), double_cv(t->repeat, t->min),
        t->min[0], t->min[t->repeat-1]);
    printf(
        "EIGENVECTORS (MAX): [avg %.2f u, cv %.2f, min %.2f u, max %.2f u]\n",
        double_mean(t->repeat, t->max), double_cv(t->repeat, t->max),
        t->max[0], t->max[t->repeat-1]);

    return HOOK_SUCCESS;
}

static const struct hook_t eigenvectors_test = {
    .name = "eigenvectors",
    .desc = "Eigenvectors check",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .print_usage = &eigenvectors_test_print_usage,
    .print_args = &eigenvectors_test_print_args,
    .check_args = &eigenvectors_test_check_args,
    .init = &eigenvectors_test_init,
    .clean = &eigenvectors_test_clean,
    .after_solver_run = &eigenvectors_test_after_solver_run,
    .summary = &eigenvectors_test_summary
};

static const struct hook_descr_t eigenvectors_test_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &eigenvectors_test
};

const struct hook_experiment_descr eigenvectors_experiment = {
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
        &eigenvectors_starpu_solver,
        &eigenvectors_starpu_simple_solver,
        &eigenvectors_lapack_solver,
        0
    },
    .hook_descrs = (struct hook_descr_t const *[])
    {
        &eigenvectors_test_descr,
        &default_analysis_descr,
        &default_print_pencil_descr,
        &default_print_input_pencil_descr,
        &default_store_raw_pencil_descr,
        &default_store_raw_input_pencil_descr,
        0
    }
};
