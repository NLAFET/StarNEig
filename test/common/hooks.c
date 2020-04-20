///
/// @file This file contains general purpose hooks.
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
#include "hooks.h"
#include "parse.h"
#include "pencil.h"
#include "checks.h"
#include "crawler.h"
#include "init.h"
#include "local_pencil.h"
#include <stdlib.h>
#include <stdio.h>

///
/// @brief Default residual failure threshold.
///
static const int residual_default_fail_threshold = 10000;

///
/// @brief Default residual warning threshold.
///
static const int residual_default_warn_threshold = 500;

///
/// @brief Residual check hook state.
///
struct residual_test_state_t {
    int repeat;             ///< experiment repetition count
    int fail_threshold;     ///< norm failure threshold
    int warn_threshold;     ///< norm warning threshold
    int fails;              ///< failure counter
    int warns;              ///< warning counter
    double *res_a;          ///< |Q^T A Z - ~A| / |A| norms
    double *res_b;          ///< |Q^T B Z - ~B| / |B| norms
    double *res_q;          ///< |Q^T Q - I| / |I| norms
    double *res_z;          ///< |Z^T Z - I| / |I| norms
};

///
/// @brief Computes ||Q A Q^T - CA||_F / u * ||CA||_F
///
/// @param[in] pencil
///         The matrix pencil.
///
/// @return ||Q A Q^T - CA||_F / u * ||CA||_F
///
static double check_qaqt(const pencil_t pencil)
{
    return compute_qazt_c_norm(
        pencil->mat_q, pencil->mat_a, pencil->mat_q, pencil->mat_ca);
}

///
/// @brief Computes ||Q A Z^T - CA||_F / u * ||CA||_F
///
/// @param[in] pencil
///         The matrix pencil.
///
/// @return ||Q A Z^T - CA||_F / u * ||CA||_F
///
static double check_qazt(const pencil_t pencil)
{
    return compute_qazt_c_norm(
        pencil->mat_q, pencil->mat_a, pencil->mat_z, pencil->mat_ca);
}

///
/// @brief Computes ||Q B Z^T - CB||_F / u * ||CB||_F
///
/// @param[in] pencil
///         The matrix pencil.
///
/// @return ||Q B Z^T - CB||_F / u * ||CB||_F
///
static double check_qbzt(const pencil_t pencil)
{
    return compute_qazt_c_norm(
        pencil->mat_q, pencil->mat_b, pencil->mat_z, pencil->mat_cb);
}

///
/// @brief Computes ||Q Q^T - I||_F / u * ||I||_F
///
/// @param[in] pencil
///         The matrix pencil.
///
/// @return ||Q Q^T - I||_F / u * ||I||_F
///
static double check_qqt(const pencil_t pencil)
{
    return compute_qqt_norm(pencil->mat_q);
}

///
/// @brief Computes ||Z Z^T - I||_F / u * ||I||_F
///
/// @param[in] pencil
///         The matrix pencil.
///
/// @return |Z Z^T - I||_F / u * ||I||_F
///
static double check_zzt(const pencil_t pencil)
{
    return compute_qqt_norm(pencil->mat_z);
}

static void residual_print_usage(int argc, char * const *argv)
{
    printf(
        "  --residual-fail-threshold (num) -- Failure threshold\n"
        "  --residual-warn-threshold (num) -- Warning threshold\n"
    );
}

static void residual_print_args(int argc, char * const *argv)
{
    printf(" --residual-fail-threshold %d",
        read_int("--residual-fail-threshold", argc, argv, NULL,
            residual_default_fail_threshold));
    printf(" --residual-warn-threshold %d",
        read_int("--residual-warn-threshold", argc, argv, NULL,
            residual_default_warn_threshold));
}

static int residual_check_args(int argc, char * const *argv, int *argr)
{
    double fail_threshold =
        read_int("--residual-fail-threshold", argc, argv, argr,
            residual_default_fail_threshold);
    double warn_threshold =
        read_int("--residual-warn-threshold", argc, argv, argr,
            residual_default_warn_threshold);

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

static int residual_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    struct residual_test_state_t *t =
        malloc(sizeof(struct residual_test_state_t));

    int fail_threshold =
        read_double("--residual-fail-threshold", argc, argv, NULL,
            residual_default_fail_threshold);
    int warn_threshold =
        read_double("--residual-warn-threshold", argc, argv, NULL,
            residual_default_warn_threshold);

    t->repeat = repeat;
    t->fail_threshold = fail_threshold;
    t->warn_threshold = warn_threshold;
    t->fails = 0;
    t->warns = 0;

    t->res_a = malloc(repeat*sizeof(double));
    t->res_q = malloc(repeat*sizeof(double));

    for (int i = 0; i < repeat; i++) {
        t->res_a[i] = 0.0;
        t->res_q[i] = 0.0;
    }

    t->res_b = NULL;
    t->res_z = NULL;

    *state = t;

    return 0;
}

static int residual_test_clean(hook_state_t state)
{
    struct residual_test_state_t *t = state;

    if (t == NULL)
        return 0;

    free(t->res_a);
    free(t->res_b);
    free(t->res_q);
    free(t->res_z);
    free(t);

    return 0;
}

static hook_return_t residual_test_after_data_init(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct residual_test_state_t *t = state;

    fill_pencil(pencil);

    if (pencil->mat_b != NULL && t->res_b == NULL) {
        t->res_b = malloc(t->repeat*sizeof(double));
        for (int i = 0; i < t->repeat; i++)
            t->res_b[i] = 0.0;
    }

    if (pencil->mat_z != NULL && t->res_z == NULL) {
        t->res_z = malloc(t->repeat*sizeof(double));
        for (int i = 0; i < t->repeat; i++)
            t->res_z[i] = 0.0;
    }

    return HOOK_SUCCESS;
}

static hook_return_t residual_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct residual_test_state_t *t = state;

    int fail = 0;
    int warn = 0;

    if (pencil->mat_b != NULL) {

        printf("|Q ~A Z^T - A| / |A|");
        fflush(stdout);

        t->res_a[iter] = check_qazt(pencil);

        if (t->warn_threshold < t->res_a[iter])
            warn++;
        if (t->fail_threshold < t->res_a[iter] || isnan(t->res_a[iter]))
            fail++;

        printf(" = %.0f u\n", t->res_a[iter]);

        printf("|Q ~B Z^T - B| / |B|");
        fflush(stdout);

        t->res_b[iter] = check_qbzt(pencil);

        if (t->warn_threshold < t->res_b[iter])
            warn++;
        if (t->fail_threshold < t->res_b[iter] || isnan(t->res_b[iter]))
            fail++;

        printf(" = %.0f u\n", t->res_b[iter]);

        printf("|Q Q^T - I| / |I|");
        fflush(stdout);

        t->res_q[iter] = check_qqt(pencil);

        if (t->warn_threshold < t->res_q[iter])
            warn++;
        if (t->fail_threshold < t->res_q[iter] || isnan(t->res_q[iter]))
            fail++;

        printf(" = %.0f u\n", t->res_q[iter]);

        printf("|Z Z^T - I| / |I|");
        fflush(stdout);

        t->res_z[iter] = check_zzt(pencil);

        if (t->warn_threshold < t->res_z[iter])
            warn++;
        if (t->fail_threshold < t->res_z[iter] || isnan(t->res_z[iter]))
            fail++;

        printf(" = %.0f u\n", t->res_z[iter]);
    }
    else {
        printf("|Q ~A Q^T - A| / |A|");
        fflush(stdout);

        t->res_a[iter] = check_qaqt(pencil);

        if (t->warn_threshold < t->res_a[iter])
            warn++;
        if (t->fail_threshold < t->res_a[iter] || isnan(t->res_a[iter]))
            fail++;

        printf(" = %.0f u\n", t->res_a[iter]);

        printf("|Q Q^T - I| / |I|");
        fflush(stdout);

        t->res_q[iter] = check_qqt(pencil);

        if (t->warn_threshold < t->res_q[iter])
            warn++;
        if (t->fail_threshold < t->res_q[iter] || isnan(t->res_q[iter]))
            fail++;

        printf(" = %.0f u\n", t->res_q[iter]);
    }

    if (fail)
        return HOOK_SOFT_FAIL;

    if (warn)
        return HOOK_WARNING;

    return HOOK_SUCCESS;
}

static hook_return_t residual_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct residual_test_state_t *t = state;

    qsort(t->res_a, t->repeat, sizeof(double), &double_compare);
    qsort(t->res_q, t->repeat, sizeof(double), &double_compare);

    if (t->res_b != NULL) {
        qsort(t->res_b, t->repeat, sizeof(double), &double_compare);
        qsort(t->res_z, t->repeat, sizeof(double), &double_compare);

        printf("|Q ~A Z^T - A| / |A| = "
            "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
            double_mean(t->repeat, t->res_a),
            double_cv(t->repeat, t->res_a),
            t->res_a[0], t->res_a[t->repeat-1]);
        printf("|Q ~B Z^T - B| / |B| = "
            "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
            double_mean(t->repeat, t->res_b),
            double_cv(t->repeat, t->res_b),
            t->res_b[0], t->res_b[t->repeat-1]);
        printf("|Q Q^T - I| / |I| = "
            "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
            double_mean(t->repeat, t->res_q),
            double_cv(t->repeat, t->res_q),
            t->res_q[0], t->res_q[t->repeat-1]);
        printf("|Z Z^T - I| / |I| = "
            "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
            double_mean(t->repeat, t->res_z),
            double_cv(t->repeat, t->res_z),
            t->res_z[0], t->res_z[t->repeat-1]);
    }
    else {
        printf("|Q ~A Q^T - A| / |A| = "
            "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
            double_mean(t->repeat, t->res_a),
            double_cv(t->repeat, t->res_a),
            t->res_a[0], t->res_a[t->repeat-1]);

        printf("|Q Q^T - I| / |I| = "
            "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
            double_mean(t->repeat, t->res_q),
            double_cv(t->repeat, t->res_q),
            t->res_q[0], t->res_q[t->repeat-1]);
    }

    return HOOK_SUCCESS;
}

const struct hook_t residual_test = {
    .name = "residual",
    .desc = "Residual check",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &residual_print_usage,
    .print_args = &residual_print_args,
    .check_args = &residual_check_args,
    .init = &residual_test_init,
    .clean = &residual_test_clean,
    .after_data_init = &residual_test_after_data_init,
    .after_solver_run = &residual_test_after_solver_run,
    .summary = &residual_test_summary
};

const struct hook_descr_t default_residual_test_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &residual_test
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static int crawl_hessenberg(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    int *status = arg;
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++)
        for (int j = offset+i+2; j < m; j++)
            if (A[i*ldA+j] != 0.0) (*status)++;

    if (1 < count) {
        double *B = ptrs[1];
        size_t ldB = lds[1];

        for (int i = 0; i < width; i++)
            for (int j = offset+i+1; j < m; j++)
                if (B[i*ldB+j] != 0.0) (*status)++;
    }

    return width;
}

static int hessenberg_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    *state = malloc(sizeof(int));
    *((int *) *state) = 0;
    return 0;
}

static int hessenberg_test_clean(hook_state_t state)
{
    free(state);
    return 0;
}

static hook_return_t hessenberg_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    pencil_t pencil = env->data;

    int error = 0;
    crawl_matrices(CRAWLER_R, CRAWLER_PANEL,
        &crawl_hessenberg, &error, sizeof(error), pencil->mat_a,
        pencil->mat_b, NULL);

    if (0 < error) {
        (*((int *) state))++;
        return HOOK_SOFT_FAIL;
    }
    return HOOK_SUCCESS;
}

static hook_return_t hessenberg_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (0 < *((int *) state)) {
        fprintf(stderr, "%d HESSENBERG FORM TESTS FAILED\n", *((int *) state));
        return HOOK_SOFT_FAIL;
    }
    printf("NO FAILED HESSENBERG FORM TESTS\n");
    return HOOK_SUCCESS;
}

const struct hook_t hessenberg_test = {
    .name = "hessenberg",
    .desc = "Hessenberg form check",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .init = &hessenberg_test_init,
    .clean = &hessenberg_test_clean,
    .after_solver_run = &hessenberg_test_after_solver_run,
    .summary = &hessenberg_test_summary
};

const struct hook_descr_t default_hessenberg_test_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &hessenberg_test
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct schur_crawler_arg {
    int errors;
    enum {
        NONE, NON_ZERO, NORMALIZATION, SINGULAR, FAKE, COLUMN, B_COLUMN
    } error_type;
};

static int crawl_schur(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    struct schur_crawler_arg *state = arg;

    double *A = ptrs[0];
    int ldA = lds[0];

    double *B = NULL;
    int ldB = 0;
    if (1 < count) {
        B = ptrs[1];
        ldB = lds[1];
    }

    #define _A(i,j) A[(j-offset)*ldA+(i)]
    #define _B(i,j) B[(j-offset)*ldB+(i)]

    extern double dlamch_(char const *);
    const double safmin = dlamch_("S");

    int _n = offset+width < n ? offset+width-1 : n;
    int marker = offset;

    int two_by_two = 0;
    for (int i = offset; i < _n; i++) {

        if (i+1 < n && _A(i+1,i) != 0.0) {
            if (two_by_two) {
                if (state->error_type != NON_ZERO) {
                    if (state->error_type != NONE)
                        printf("\n");
                    printf("SCHUR TEST: NON-ZERO SUB-DIAGONAL ENTRY AT ");
                }
                printf("(%d,%d) ", i+1, i);
                state->error_type = NON_ZERO;
                state->errors++;
                two_by_two = 0;
                continue;
            }

            if (B != NULL) {
                if (_B(i,i+1) != 0.0 || _B(i+1,i) != 0.0) {
                    if (state->error_type != NORMALIZATION) {
                        if (state->error_type != NONE)
                            printf("\n");
                        printf("SCHUR TEST: NON-NORMALIZED 2-BY-2 BLOCK AT ");
                    }
                    printf("(%d,%d) ", i, i);
                    state->error_type = NORMALIZATION;
                    state->errors++;
                    goto end;
                }

                if (_B(i,i) == 0.0 || _B(i+1,i+1) == 0.0) {
                    if (state->error_type != SINGULAR) {
                        if (state->error_type != NONE)
                            printf("\n");
                        printf("SCHUR TEST: SINGULAR 2-BY-2 BLOCK AT ");
                    }
                    printf("(%d,%d) ", i, i);
                    state->error_type = SINGULAR;
                    state->errors++;
                    goto end;
                }

                double s1, s2, wr1, wr2, wi;

                extern void dlag2_(double const *, int const *, double const *,
                    int const *, double const *, double const *, double *,
                    double *, double *, double *);

                dlag2_(&_A(i,i), &ldA, &_B(i,i), &ldB, &safmin,
                    &s1, &s2, &wr1, &wr2, &wi);

                if (wi == 0.0) {
                    if (state->error_type != FAKE) {
                        if (state->error_type != NONE)
                            printf("\n");
                        printf("SCHUR TEST: FAKE 2-BY-2 BLOCK AT ");
                    }
                    printf("(%d,%d) ", i, i);
                    state->error_type = FAKE;
                    state->errors++;
                    goto end;
                }
            }
            else {
                if (_A(i,i) != _A(i+1,i+1)) {
                    if (state->error_type != NORMALIZATION) {
                        if (state->error_type != NONE)
                            printf("\n");
                        printf("SCHUR TEST: NON-NORMALIZED 2-BY-2 BLOCK AT ");
                    }
                    printf("(%d,%d) ", i, i);
                    state->error_type = NORMALIZATION;
                    state->errors++;
                    goto end;
                }

                double a[] = {
                    _A(i,i), _A(i+1,i), _A(i,i+1), _A(i+1,i+1)
                };

                double rt1r, rt1i, rt2r, rt2i, cs, ss;

                extern void dlanv2_(
                    double *, double *, double *, double *, double *,
                    double *, double *, double *, double *, double *);

                dlanv2_(&a[0], &a[2], &a[1], &a[3],
                    &rt1r, &rt1i, &rt2r, &rt2i, &cs, &ss);

                if (rt1i == 0.0 || rt2i == 0.0) {
                    if (state->error_type != FAKE) {
                        if (state->error_type != NONE)
                            printf("\n");
                        printf("SCHUR TEST: FAKE 2-BY-2 BLOCK AT ");
                    }
                    printf("(%d,%d) ", i, i);
                    state->error_type = FAKE;
                    state->errors++;
                    goto end;
                }
            }
end:
            two_by_two = 1;
        }
        else {
            two_by_two = 0;
        }

        marker++;
    }

    for (int i = offset; i < _n; i++) {
        for (int j = i+2; j < m; j++) {
            if (_A(j,i) != 0.0) {
                if (state->error_type != COLUMN) {
                    if (state->error_type != NONE)
                        printf("\n");
                    printf("SCHUR TEST: NON-ZERO COLUMN AT ");
                }
                printf("(%d,%d)-> ", j, i);
                state->error_type = COLUMN;
                state->errors++;
                break;
            }
        }
    }

    if (B != NULL) {
        for (int i = offset; i < _n; i++) {
            for (int j = i+1; j < m; j++) {
                if (_B(j,i) != 0.0) {
                    if (state->error_type != B_COLUMN) {
                        if (state->error_type != NONE)
                            printf("\n");
                        printf("SCHUR TEST: NON-ZERO B MATRIX COLUMN AT ");
                    }
                    printf("(%d,%d)-> ", j, i);
                    state->error_type = B_COLUMN;
                    state->errors++;
                    break;
                }
            }
        }
    }

    if (state->error_type != NONE)
        printf("\n");
    state->error_type = NONE;

    #undef _A
    #undef _B

    return marker-offset;
}

static int schur_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    *state = malloc(sizeof(int));
    *((int *) *state) = 0;
    return 0;
}

static int schur_test_clean(hook_state_t state)
{
    free(state);
    return 0;
}

static hook_return_t schur_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct schur_crawler_arg arg = {
        .errors = 0,
        .error_type = NONE
    };

    pencil_t pencil = env->data;
    crawl_matrices(CRAWLER_R, CRAWLER_PANEL,
        &crawl_schur, &arg, sizeof(arg), pencil->mat_a, pencil->mat_b, NULL);

    if (0 < arg.errors) {
        (*((int *) state))++;
        return HOOK_SOFT_FAIL;
    }
    return HOOK_SUCCESS;
}

static hook_return_t schur_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (0 < *((int *) state)) {
        fprintf(stderr, "%d SCHUR FORM TESTS FAILED\n", *((int *) state));
        return HOOK_SOFT_FAIL;
    }
    printf("NO FAILED SCHUR FORM TESTS\n");
    return HOOK_SUCCESS;
}

const struct hook_t schur_test = {
    .name = "schur",
    .desc = "Schur form check",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .init = &schur_test_init,
    .clean = &schur_test_clean,
    .after_solver_run = &schur_test_after_solver_run,
    .summary = &schur_test_summary
};

const struct hook_descr_t default_schur_test_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &schur_test
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static const int eigenvalues_test_default_fail_threshold = 10000;
static const int eigenvalues_test_default_warn_threshold = 1000;

struct eigenvalues_test_state {
    int repeat;             ///< experiment repetition count
    int fail_threshold;     ///< norm failure threshold
    int warn_threshold;     ///< norm warning threshold
    int *warning;
    int *fail;
    double *mean;
    double *min;
    double *max;
};

static void eigenvalues_test_print_usage(int argc, char * const *argv)
{
    printf(
        "  --eigenvalues-fail-threshold (num) -- Failure threshold\n"
        "  --eigenvalues-warn-threshold (num) -- Warning threshold\n"
    );
}

static void eigenvalues_test_print_args(int argc, char * const *argv)
{
    printf(" --eigenvalues-fail-threshold %d",
        read_int("--eigenvalues-fail-threshold", argc, argv, NULL,
            eigenvalues_test_default_fail_threshold));
    printf(" --eigenvalues-warn-threshold %d",
        read_int("--eigenvalues-warn-threshold", argc, argv, NULL,
            eigenvalues_test_default_warn_threshold));
}

static int eigenvalues_test_check_args(int argc, char * const *argv, int *argr)
{
    int fail_threshold =
        read_int("--eigenvalues-fail-threshold", argc, argv, argr,
            eigenvalues_test_default_fail_threshold);
    int warn_threshold =
        read_int("--eigenvalues-warn-threshold", argc, argv, argr,
            eigenvalues_test_default_warn_threshold);

    if (fail_threshold < 0) {
        fprintf(stderr, "EIGENVALUES CHECK: Invalid failure threshold.\n");
        return 1;
    }
    if (warn_threshold < 0) {
        fprintf(stderr, "EIGENVALUES CHECK: Invalid warning threshold.\n");
        return 1;
    }

    if (fail_threshold < warn_threshold)
        fprintf(stderr,
            "EIGENVALUES CHECK: The warning threshold is tighter than the "
            "failure threshold.\n");

    return 0;
}

static int eigenvalues_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    struct eigenvalues_test_state *t =
        malloc(sizeof(struct eigenvalues_test_state));

    int fail_threshold =
        read_double("--eigenvalues-fail-threshold", argc, argv, NULL,
            eigenvalues_test_default_fail_threshold);
    int warn_threshold =
        read_double("--eigenvalues-warn-threshold", argc, argv, NULL,
            eigenvalues_test_default_warn_threshold);

    t->repeat = repeat;
    t->fail_threshold = fail_threshold;
    t->warn_threshold = warn_threshold;

    t->warning = malloc(repeat*sizeof(int));
    t->fail = malloc(repeat*sizeof(int));

    t->mean = malloc(repeat*sizeof(double));
    t->min = malloc(repeat*sizeof(double));
    t->max = malloc(repeat*sizeof(double));

    *state = t;

    return 0;
}

static int eigenvalues_test_clean(hook_state_t state)
{
    struct eigenvalues_test_state *t = state;

    if (t == NULL)
        return 0;

    free(t->warning);
    free(t->fail);
    free(t->mean);
    free(t->min);
    free(t->max);
    free(t);

    return 0;
}

static hook_return_t eigenvalues_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct eigenvalues_test_state *t = state;

    int n = GENERIC_MATRIX_N(pencil->mat_a);

    double *real1 = malloc(n*sizeof(double));
    double *imag1 = malloc(n*sizeof(double));
    double *beta1 = malloc(n*sizeof(double));
    extract_eigenvalues(pencil->mat_a, pencil->mat_b, real1, imag1, beta1);

    double *real2, *imag2, *beta2;
    get_supplementaty_eigenvalues(pencil->supp, &real2, &imag2, &beta2);

    t->warning[iter] = 0;
    t->fail[iter] = 0;
    t->mean[iter] = 0.0;
    t->min[iter] = 1.0/0.0;
    t->max[iter] = 0.0;
    for (int i = 0; i < n; i++) {
        if (real1[i] == 0.0 && imag1[i] == 0.0 &&
        real2[i] == 0.0 && imag2[i] == 0.0) {
            // no problem ...
        }
        else if (beta1[i] == 0.0 && beta2[i] == 0.0) {
            // no problem ...
        }
        else if (beta1[i] == 0.0 && beta2[i] != 0.0) {
            printf(
                "EIGENVALUES TEST (FAILURE): A finite eigenvalue was returned "
                "in place of a infinite eigenvalue at %d.\n", i);
            t->fail[iter]++;
            t->mean[iter] += 1.0/0.0;
            t->min[iter] = MIN(t->min[iter], 1.0/0.0);
            t->max[iter] = MAX(t->max[iter], 1.0/0.0);
        }
        else {
            double diff = ((long long)1<<52) *
                sqrt(
                    squ(real1[i]/beta1[i]-real2[i]/beta2[i]) +
                    squ(imag1[i]/beta1[i]-imag2[i]/beta2[i])
                ) / sqrt(squ(real1[i]/beta1[i]) + squ(imag1[i]/beta1[i]));

            if (t->warn_threshold < diff || t->fail_threshold < diff ||
            isnan(diff))
            {
                if (t->fail_threshold < diff || isnan(diff)) {
                    printf(
                        "EIGENVALUES TEST (FAILURE): An incorrect eigenvalue "
                        "was returned at %d. Returned (%e,%e), correct " \
                        "(%e, %e) (diff = %.0f u).\n", i,
                        real2[i]/beta2[i], imag2[i]/beta2[i],
                        real1[i]/beta1[i], imag1[i]/beta1[i], diff);
                    t->fail[iter]++;
                }
                else {
                    printf(
                        "EIGENVALUES TEST (WARNING): An incorrect eigenvalue "
                        "was returned at %d. Returned (%e,%e), correct " \
                        "(%e, %e) (diff = %.0f u).\n", i,
                        real2[i]/beta2[i], imag2[i]/beta2[i],
                        real1[i]/beta1[i], imag1[i]/beta1[i], diff);
                    t->warning[iter]++;
                }
            }
            t->mean[iter] += diff;
            t->min[iter] = MIN(t->min[iter], diff);
            t->max[iter] = MAX(t->max[iter], diff);
        }
    }

    t->mean[iter] /= n;

    printf("EIGENVALUES CHECK: mean = %.0f u, min = %.0f u, max = %.0f u\n",
        t->mean[iter], t->min[iter], t->max[iter]);

    if (0 < t->fail[iter])
        printf(
            "EIGENVALUES CHECK: %d eigenvalue failures.\n", t->fail[iter]);

    if (0 < t->warning[iter])
        printf(
            "EIGENVALUES CHECK: %d eigenvalue warnings.\n", t->warning[iter]);

    free(real1);
    free(imag1);
    free(beta1);

    if (0 < t->fail[iter])
        return HOOK_SOFT_FAIL;

    if (0 < t->warning[iter])
        return HOOK_WARNING;

    return HOOK_SUCCESS;
}

static hook_return_t eigenvalues_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct eigenvalues_test_state *t = state;

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

    printf("EIGENVALUES CHECK (WARNINGS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        warning_runs, int_mean(t->repeat, t->warning),
        int_cv(t->repeat, t->warning), t->warning[0], t->warning[t->repeat-1]);
    printf("EIGENVALUES CHECK (FAILS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        fail_runs, int_mean(t->repeat, t->fail), int_cv(t->repeat, t->fail),
        t->fail[0], t->fail[t->repeat-1]);
    printf(
        "EIGENVALUES CHECK (MEANS): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->repeat, t->mean), double_cv(t->repeat, t->mean),
        t->mean[0], t->mean[t->repeat-1]);
    printf("EIGENVALUES CHECK (MIN): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->repeat, t->min), double_cv(t->repeat, t->min),
        t->min[0], t->min[t->repeat-1]);
    printf("EIGENVALUES CHECK (MAX): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->repeat, t->max), double_cv(t->repeat, t->max),
        t->max[0], t->max[t->repeat-1]);

    return HOOK_SUCCESS;
}

const struct hook_t eigenvalues_test = {
    .name = "eigenvalues",
    .desc = "Checks returned eigenvalues",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &eigenvalues_test_print_usage,
    .print_args = &eigenvalues_test_print_args,
    .check_args = &eigenvalues_test_check_args,
    .init = &eigenvalues_test_init,
    .clean = &eigenvalues_test_clean,
    .after_solver_run = &eigenvalues_test_after_solver_run,
    .summary = &eigenvalues_test_summary
};

const struct hook_descr_t default_eigenvalues_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &eigenvalues_test
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static const int known_eigenvalues_test_default_fail_threshold = 1000000;
static const int known_eigenvalues_test_default_warn_threshold = 10000;

struct known_eigenvalues_test_state {
    int used;
    int fail_threshold;     ///< norm failure threshold
    int warn_threshold;     ///< norm warning threshold
    int *warning;
    int *fail;
    double *mean;
    double *min;
    double *max;
};

static void known_eigenvalues_test_print_usage(int argc, char * const *argv)
{
    printf(
        "  --known-eigenvalues-fail-threshold (num) -- Failure threshold\n"
        "  --known-eigenvalues-warn-threshold (num) -- Warning threshold\n"
    );
}

static void known_eigenvalues_test_print_args(int argc, char * const *argv)
{
    printf(" --known-eigenvalues-fail-threshold %d",
        read_int("--known-eigenvalues-fail-threshold", argc, argv, NULL,
            known_eigenvalues_test_default_fail_threshold));
    printf(" --known-eigenvalues-warn-threshold %d",
        read_int("--known-eigenvalues-warn-threshold", argc, argv, NULL,
            known_eigenvalues_test_default_warn_threshold));
}

static int known_eigenvalues_test_check_args(
    int argc, char * const *argv, int *argr)
{
    int fail_threshold =
        read_int("--known-eigenvalues-fail-threshold", argc, argv, argr,
            known_eigenvalues_test_default_fail_threshold);
    int warn_threshold =
        read_int("--known-eigenvalues-warn-threshold", argc, argv, argr,
            known_eigenvalues_test_default_warn_threshold);

    if (fail_threshold < 0) {
        fprintf(stderr,
            "KNOWN EIGENVALUES CHECK: Invalid failure threshold.\n");
        return 1;
    }
    if (warn_threshold < 0) {
        fprintf(stderr,
            "KNOWN EIGENVALUES CHECK: Invalid warning threshold.\n");
        return 1;
    }

    if (fail_threshold < warn_threshold)
        fprintf(stderr,
            "KNOWN EIGENVALUES CHECK: The warning threshold is tighter than " \
            "the failure threshold.\n");

    return 0;
}

static int known_eigenvalues_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    struct known_eigenvalues_test_state *t =
        malloc(sizeof(struct known_eigenvalues_test_state));

    int fail_threshold =
        read_double("--known-eigenvalues-fail-threshold", argc, argv, NULL,
            known_eigenvalues_test_default_fail_threshold);
    int warn_threshold =
        read_double("--known-eigenvalues-warn-threshold", argc, argv, NULL,
            known_eigenvalues_test_default_warn_threshold);

    t->used = 0;
    t->fail_threshold = fail_threshold;
    t->warn_threshold = warn_threshold;

    t->warning = malloc(repeat*sizeof(int));
    t->fail = malloc(repeat*sizeof(int));

    t->mean = malloc(repeat*sizeof(double));
    t->min = malloc(repeat*sizeof(double));
    t->max = malloc(repeat*sizeof(double));

    *state = t;

    return 0;
}

static int known_eigenvalues_test_clean(hook_state_t state)
{
    struct known_eigenvalues_test_state *t = state;

    if (t == NULL)
        return 0;

    free(t->warning);
    free(t->fail);
    free(t->mean);
    free(t->min);
    free(t->max);
    free(t);

    return 0;
}

static hook_return_t known_eigenvalues_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct known_eigenvalues_test_state *t = state;

    int n = GENERIC_MATRIX_N(pencil->mat_a);

    double *real2, *imag2, *beta2;
    get_supplementaty_known_eigenvalues(pencil->supp, &real2, &imag2, &beta2);
    if (real2 == NULL || imag2 == NULL || beta2 == NULL) {
        fprintf(stderr,
            "KNOWN EIGENVALUES CHECK: The stored pencil does not contain " \
            "the known eigenvalues. Skipping.\n");
        return HOOK_SUCCESS;
    }

    double *real1 = malloc(n*sizeof(double));
    double *imag1 = malloc(n*sizeof(double));
    double *beta1 = malloc(n*sizeof(double));
    extract_eigenvalues(pencil->mat_a, pencil->mat_b, real1, imag1, beta1);

    t->warning[t->used] = 0;
    t->fail[t->used] = 0;
    t->mean[t->used] = 0.0;
    t->min[t->used] = 1.0/0.0;
    t->max[t->used] = 0.0;

    int *used = malloc(n*sizeof(int));
    memset(used, 0, n*sizeof(int));

    for (int i = 0; i < n; i++) {

        // find closest match
        int closest = n;
        double closest_diff = 1.0/0.0;
        for (int j = 0; j < n; j++) {
            if (!used[j]) {
                if (beta1[i] == 0.0 && beta2[j] == 0.0) {
                    closest = j;
                    closest_diff = 0.0;
                    break;
                } else {
                    double diff;
                    if (real2[j] == 0.0 && imag2[j] == 0.0)
                        diff = ((long long)1<<52) * sqrt(
                            squ(real1[i]/beta1[i]) +
                            squ(imag1[i]/beta1[i]));
                    else
                        diff = ((long long)1<<52) * sqrt(
                            squ(real1[i]/beta1[i]-real2[j]/beta2[j]) +
                            squ(imag1[i]/beta1[i]-imag2[j]/beta2[j])
                        ) / sqrt(
                            squ(real2[j]/beta2[j]) + squ(imag2[j]/beta2[j]));

                    if (diff < closest_diff) {
                        closest = j;
                        closest_diff = diff;
                    }
                }
            }
        }
        if (closest < n)
            used[closest] = 1;

        if (t->warn_threshold < closest_diff ||
        t->fail_threshold < closest_diff || isnan(closest_diff)) {
            if (t->fail_threshold < closest_diff || isnan(closest_diff)) {
                printf(
                    "KNOWN EIGENVALUES TEST (FAILURE): An incorrect eigenvalue "
                    "was returned at %d (diff = %.0f u).\n", i, closest_diff);
                t->fail[t->used]++;
            }
            else {
                printf(
                    "KNOWN EIGENVALUES TEST (WARNING): An incorrect eigenvalue "
                    "was returned at %d (diff = %.0f u).\n", i, closest_diff);
                t->warning[t->used]++;
            }
        }
        t->mean[t->used] += closest_diff;
        t->min[t->used] = MIN(t->min[t->used], closest_diff);
        t->max[t->used] = MAX(t->max[t->used], closest_diff);
    }

    t->mean[t->used] /= n;

    printf(
        "KNOWN EIGENVALUES CHECK: mean = %.0f u, min = %.0f u, max = %.0f u\n",
        t->mean[t->used], t->min[t->used], t->max[t->used]);

    if (0 < t->fail[t->used])
        printf(
            "KNOWN EIGENVALUES CHECK: %d eigenvalue failures.\n",
            t->fail[t->used]);

    if (0 < t->warning[t->used])
        printf(
            "KNOWN EIGENVALUES CHECK: %d eigenvalue warnings.\n",
            t->warning[t->used]);

    t->used++;

    free(real1);
    free(imag1);
    free(beta1);
    free(used);

    if (0 < t->fail[t->used-1])
        return HOOK_SOFT_FAIL;

    if (0 < t->warning[t->used-1])
        return HOOK_WARNING;

    return HOOK_SUCCESS;
}

static hook_return_t known_eigenvalues_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct known_eigenvalues_test_state *t = state;

    if (t->used == 0)
        return HOOK_SUCCESS;

    qsort(t->warning, t->used, sizeof(int), &int_compare);
    qsort(t->fail, t->used, sizeof(int), &int_compare);
    qsort(t->mean, t->used, sizeof(double), &double_compare);
    qsort(t->min, t->used, sizeof(double), &double_compare);
    qsort(t->max, t->used, sizeof(double), &double_compare);

    int warning_runs = 0;
    int fail_runs = 0;
    for (int i = 0; i < t->used; i++) {
        if (0 < t->warning[i]) warning_runs++;
        if (0 < t->fail[i]) fail_runs++;
    }

    printf("KNOWN EIGENVALUES CHECK (WARNINGS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        warning_runs, int_mean(t->used, t->warning),
        int_cv(t->used, t->warning), t->warning[0], t->warning[t->used-1]);
    printf("KNOWN EIGENVALUES CHECK (FAILS): %d runs effected "
        "[avg %.1f, cv %.2f, min %d, max %d]\n",
        fail_runs, int_mean(t->used, t->fail), int_cv(t->used, t->fail),
        t->fail[0], t->fail[t->used-1]);
    printf(
        "KNOWN EIGENVALUES CHECK (MEANS): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->used, t->mean), double_cv(t->used, t->mean),
        t->mean[0], t->mean[t->used-1]);
    printf("KNOWN EIGENVALUES CHECK (MIN): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->used, t->min), double_cv(t->used, t->min),
        t->min[0], t->min[t->used-1]);
    printf("KNOWN EIGENVALUES CHECK (MAX): "
        "[avg %.0f u, cv %.2f, min %.0f u, max %.0f u]\n",
        double_mean(t->used, t->max), double_cv(t->used, t->max),
        t->max[0], t->max[t->used-1]);

    return HOOK_SUCCESS;
}

const struct hook_t known_eigenvalues_test = {
    .name = "known-eigenvalues",
    .desc = "Checks computed eigenvalues agains known values",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &known_eigenvalues_test_print_usage,
    .print_args = &known_eigenvalues_test_print_args,
    .check_args = &known_eigenvalues_test_check_args,
    .init = &known_eigenvalues_test_init,
    .clean = &known_eigenvalues_test_clean,
    .after_solver_run = &known_eigenvalues_test_after_solver_run,
    .summary = &known_eigenvalues_test_summary
};

const struct hook_descr_t default_known_eigenvalues_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &known_eigenvalues_test
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct analysis_test_state {
    int repeat;
    int *zeros;
    int *close_zeros;
    int *infs;
    int *close_infs;
    int *indefinites;
    int *close_indefinites;
};

static int analysis_test_init(
    int argc, char * const *argv, int repeat, int warmup, hook_state_t *state)
{
    struct analysis_test_state *t =
        malloc(sizeof(struct analysis_test_state));

    t->repeat = repeat;
    t->zeros = malloc(repeat*sizeof(int));
    t->close_zeros = malloc(repeat*sizeof(int));
    t->infs = malloc(repeat*sizeof(int));
    t->close_infs = malloc(repeat*sizeof(int));
    t->indefinites = malloc(repeat*sizeof(int));
    t->close_indefinites = malloc(repeat*sizeof(int));

    *state = t;

    return 0;
}

static int analysis_test_clean(hook_state_t state)
{
    struct analysis_test_state *t = state;

    if (t == NULL)
        return 0;

    free(t->zeros);
    free(t->close_zeros);
    free(t->infs);
    free(t->close_infs);
    free(t->indefinites);
    free(t->close_indefinites);
    free(t);

    return 0;
}

static hook_return_t analysis_test_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    if (iter < 0)
        return HOOK_SUCCESS;

    pencil_t pencil = (pencil_t) env->data;
    struct analysis_test_state *t = state;

    int n = GENERIC_MATRIX_N(pencil->mat_a);

    double *real1 = malloc(n*sizeof(double));
    double *imag1 = malloc(n*sizeof(double));
    double *beta1 = malloc(n*sizeof(double));
    extract_eigenvalues(pencil->mat_a, pencil->mat_b, real1, imag1, beta1);

    double thres_a = norm_C(pencil->mat_a) / ((long long)1<<52);

    double thres_b = 1.0/((long long)1<<52);
    if (pencil->mat_b != NULL)
        thres_b = norm_C(pencil->mat_b) / ((long long)1<<52);

    t->zeros[iter] = 0;
    t->close_zeros[iter] = 0;
    t->infs[iter] = 0;
    t->close_infs[iter] = 0;
    t->indefinites[iter] = 0;
    t->close_indefinites[iter] = 0;
    for (int i = 0; i < n; i++) {
        if (imag1[i] == 0.0) {
            if (real1[i] == 0.0 && beta1[i] == 0.0)
                t->indefinites[iter]++;
            else if (fabs(real1[i]) < thres_a && fabs(beta1[i]) < thres_b)
                t->close_indefinites[iter]++;
            else if (real1[i] == 0.0)
                t->zeros[iter]++;
            else if (fabs(real1[i]) < thres_a)
                t->close_zeros[iter]++;
            else if (beta1[i] == 0.0)
                t->infs[iter]++;
            else if (fabs(beta1[i]) < thres_b)
                t->close_infs[iter]++;
        }
    }

    printf("EIGENVALUES ANALYSIS: zeros = %d, infinities = %d, indefinites = %d\n",
        t->zeros[iter], t->infs[iter], t->indefinites[iter]);
    printf(
        "EIGENVALUES ANALYSIS: close zeros = %d, close infinities = %d, "
        "close indefinites = %d\n",
        t->close_zeros[iter], t->close_infs[iter], t->close_indefinites[iter]);

    free(real1);
    free(imag1);
    free(beta1);

    return HOOK_SUCCESS;
}

static hook_return_t analysis_test_summary(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    struct analysis_test_state *t = state;

    qsort(t->zeros, t->repeat, sizeof(int), &int_compare);
    qsort(t->close_zeros, t->repeat, sizeof(int), &int_compare);
    qsort(t->infs, t->repeat, sizeof(int), &int_compare);
    qsort(t->close_infs, t->repeat, sizeof(int), &int_compare);
    qsort(t->indefinites, t->repeat, sizeof(int), &int_compare);
    qsort(t->close_indefinites, t->repeat, sizeof(int), &int_compare);

#define REPORT(CAP, name) \
    printf("EIGENVALUES ANALYSIS ("CAP"): " \
        "[avg %.1f, cv %.2f, min %d, max %d]\n", \
        int_mean(t->repeat, t->name), \
        int_cv(t->repeat, t->name), t->name[0], t->name[t->repeat-1]);

    REPORT("ZEROS", zeros);
    REPORT("CLOSE ZEROS", close_zeros);
    REPORT("INFINITIES", infs);
    REPORT("CLOSE INFINITIES", close_infs);
    REPORT("INDEFINITES", indefinites);
    REPORT("CLOSE INDEFINITES", close_indefinites);

#undef REPORT

    return HOOK_SUCCESS;
}

const struct hook_t analysis_test = {
    .name = "analysis",
    .desc = "Analyses a (generalized) schur form",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .init = &analysis_test_init,
    .clean = &analysis_test_clean,
    .after_solver_run = &analysis_test_after_solver_run,
    .summary = &analysis_test_summary
};

const struct hook_descr_t default_analysis_descr = {
    .is_enabled = 1,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &analysis_test
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static hook_return_t print_pencil_print(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    pencil_t pencil = (pencil_t) env->data;

#define PRINT_MATRIX(name, mat_x) \
    if (pencil->mat_x) { \
        printf("Matric %s:\n", name); \
        print_matrix_descr(pencil->mat_x, stdout); \
    }

    PRINT_MATRIX("A", mat_a);
    PRINT_MATRIX("B", mat_b);
    PRINT_MATRIX("Q", mat_q);
    PRINT_MATRIX("Z", mat_z);
    PRINT_MATRIX("X", mat_x);
    PRINT_MATRIX("CA", mat_ca);
    PRINT_MATRIX("CB", mat_cb);

#undef PRINT_MATRIX

    print_supplementary(pencil->supp);

    return HOOK_SUCCESS;
}

const struct hook_t print_input_pencil = {
    .name = "print-input",
    .desc = "Prints the input pencil",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .before_solver_run = &print_pencil_print,
};

const struct hook_descr_t default_print_input_pencil_descr = {
    .is_enabled = 0,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &print_input_pencil
};

const struct hook_t print_pencil = {
    .name = "print",
    .desc = "Prints the output pencil",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .after_solver_run = &print_pencil_print,
};

const struct hook_descr_t default_print_pencil_descr = {
    .is_enabled = 0,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &print_pencil
};
