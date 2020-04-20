///
/// @file This file contains the 2-by-2 block generator modules.
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
#include "complex_distr.h"
#include "common.h"
#include "parse.h"
#include "block_placer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static const double default_ratio = 0.5;
static const double default_zero_ratio = 0.01;
static const double default_inf_ratio = 0.01;

///
/// @brief
///
static void generate_special_cases(
    int n, double complex_ratio, double zero_ratio, double inf_ratio,
    double *real, double *imag, double *beta)
{
    // place zero blocks (zero eigenvaleus)
    for (int i = 0; i < n; i++) {
        if (i+1 < n && imag[i] != 0.0)
            i++;
        else if (1.0 * prand() / PRAND_MAX < zero_ratio/(1.0-complex_ratio))
            real[i] = 0.0;
    }

    // place zero blocks (infinite eigenvalues)
    for (int i = 0; beta != NULL && i < n; i++) {
        if (i+1 < n && imag[i] != 0.0)
            i++;
        else if (real[i] != 0.0 &&
        1.0 * prand() / PRAND_MAX < inf_ratio/(1.0-complex_ratio-zero_ratio))
            beta[i] = 0.0;
    }
}

////////////////////////////////////////////////////////////////////////////////

static void uniform_complex_distr_print_usage()
{
    printf(
        "  --fortify -- Fortify against failed swaps\n"
        "  --complex-ratio (0.0-1.0) -- Complex eigenvalue ratio\n"
        "  --zero-ratio (0.0-1.0) -- Zero eigenvalue ratio\n"
        "  --inf-ratio (0.0-1.0) -- Infinite eigenvalue ratio\n"
    );
}

static int uniform_complex_distr_check_args(
    int argc, char * const *argv, int *argr)
{
    int fortify = read_opt("--fortify", argc, argv, argr);

    if (!fortify) {
        double complex_ratio =
            read_double("--complex-ratio", argc, argv, argr, default_ratio);

        if (complex_ratio < 0.0 || 1.0 < complex_ratio) {
            fprintf(stderr, "Invalid complex ratio.\n");
            return -1;
        }

        double zero_ratio =
            read_double("--zero-ratio", argc, argv, argr, default_zero_ratio);

        if (zero_ratio < 0.0 && 1.0 < zero_ratio) {
            fprintf(stderr, "Invalid zero eigenvalue ratio.\n");
            return -1;
        }

        double inf_ratio =
            read_double("--inf-ratio", argc, argv, argr, default_inf_ratio);

        if (inf_ratio < 0.0 && 1.0 < inf_ratio) {
            fprintf(stderr, "Invalid infinite eigenvalue ratio.\n");
            return -1;
        }
    }

    return 0;
}

static void uniform_complex_distr_print_args(
    int argc, char * const *argv)
{
    int fortify = read_opt("--fortify", argc, argv, NULL);

    if (fortify) {
        printf(" --fortify");
    }
    else {
        double complex_ratio =
            read_double("--complex-ratio", argc, argv, NULL, default_ratio);
        printf(" --complex-ratio %f", complex_ratio);

        printf(" --zero-ratio %f",
            read_double("--zero-ratio", argc, argv, NULL, default_zero_ratio));

        printf(" --inf-ratio %f",
            read_double("--inf-ratio", argc, argv, NULL, default_inf_ratio));
    }
}

static int uniform_complex_distr_init(
    int argc, char * const *argv, matrix_t A, matrix_t B)
{
    int fortify = read_opt("--fortify", argc, argv, NULL);

    double complex_ratio =
        read_double("--complex-ratio", argc, argv, NULL, default_ratio);

    double zero_ratio =
        read_double("--zero-ratio", argc, argv, NULL, default_zero_ratio);
    double inf_ratio =
        read_double("--inf-ratio", argc, argv, NULL, default_inf_ratio);

    int n = GENERIC_MATRIX_M(A);

    int complex_count = complex_ratio * n / 2;
    int real_count = n - 2 * complex_count;

    // place the 1x1 blocks between the 2-by-2 blocks
    int spaces[complex_count+1];
    for (int i = 0; i < complex_count+1; i++)
        spaces[i] = 0;
    for (int i = 0; i < real_count; i++)
        spaces[prand() % (complex_count+1)]++;

    double *real = malloc(n*sizeof(double));
    double *imag = malloc(n*sizeof(double));
    double *beta = malloc(n*sizeof(double));

    // place the 1-by-1 blocks
    if (fortify) {
        for (int i = 0; i < n; i++) {
            real[i] = 2.0*(i - n/2 + 0.5);
            imag[i] = 0.0;
            beta[i] = 1.0;
        }
    }
    else {
        for (int i = 0; i < n; i++) {
            real[i] = (2.0*prand()/PRAND_MAX-1.0) * n;
            imag[i] = 0.0;
            if (B != NULL)
                beta[i] = 1.0*prand()/PRAND_MAX;
            else
                beta[i] = 1.0;
        }
    }

    // place the 2-by-2 blocks into the diagonal
    if (fortify) {
        int i = 0;
        for (int j = 0; j < complex_count; j++) {
            i += spaces[j];
            imag[i]   =  fabs(real[i]);
            real[i+1] =       real[i];
            imag[i+1] =      -imag[i];
            i += 2;
        }
    }
    else {
        int i = 0;
        for (int j = 0; j < complex_count; j++) {
            i += spaces[j];
            real[i] = (2.0*prand()/PRAND_MAX-1.0) * n;
            imag[i] = (2.0*prand()/PRAND_MAX-1.0) * n;
            real[i+1] =  real[i];
            imag[i+1] = -imag[i];
            beta[i] = beta[i+1] = 1.0*prand()/PRAND_MAX;
            i += 2;
        }
    }

    if (!fortify)
        generate_special_cases(n, complex_ratio, zero_ratio, inf_ratio,
            real, imag, beta);

    block_placer(real, imag, beta, A, B);

    free(real);
    free(imag);
    free(beta);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////

static void bulk_complex_distr_print_usage()
{
    printf(
        "  --fortify -- Fortify against failed swaps\n"
        "  --bulk-complex-begin [top,middle,(num)] -- All diagonal entries"
        "between begin and end are 2-by-2 blocks\n"
        "  --bulk-complex-end [middle,bottom,(num)] -- All diagonal "
        "entries between begin and end are 2-by-2 blocks\n"
        "  --bulk-complex-top -- All diagonal entries at the top half of "
        "the matrix are 2-by-2 blocks\n"
        "  --bulk-complex-bottom -- All diagonal entries at the bottom "
        "half of the matrix are 2-by-2 blocks\n"
        "  --zero-ratio (0.0-1.0) -- Zero eigenvalue ratio\n"
        "  --inf-ratio (0.0-1.0) -- Infinite eigenvalue ratio\n"
    );
}

static int bulk_complex_distr_check_args(
    int argc, char * const *argv, int *argr)
{
    read_opt("--fortify", argc, argv, argr);

    if (read_opt("--bulk-complex-top", argc, argv, argr))
        return 0;
    if (read_opt("--bulk-complex-bottom", argc, argv, argr))
        return 0;

    struct multiarg_t begin = read_multiarg(
        "--bulk-complex-begin", argc, argv, argr, "top", "middle", NULL);

    if (begin.type == MULTIARG_INVALID ||
    (begin.type == MULTIARG_INT && begin.int_value < 0)) {
        fprintf(stderr, "Invalid --bulk-complex-begin value.\n");
        return -1;
    }

    struct multiarg_t end = read_multiarg(
        "--bulk-complex-end", argc, argv, argr, "middle", "bottom", NULL);

    if (end.type == MULTIARG_INVALID ||
    (end.type == MULTIARG_INT && end.int_value < 0)) {
        fprintf(stderr, "Invalid --bulk-complex-end value.\n");
        return -1;
    }

    if (begin.type == MULTIARG_INT && end.type == MULTIARG_INT) {
        if (end.int_value < begin.int_value) {
            fprintf(stderr,
                "Invalid --bulk-complex-begin or --bulk-complex-end.\n");
                return -1;
        }
    }

    double zero_ratio =
        read_double("--zero-ratio", argc, argv, argr, default_zero_ratio);

    if (zero_ratio < 0.0 && 1.0 < zero_ratio) {
        fprintf(stderr, "Invalid zero eigenvalue ratio.\n");
        return -1;
    }

    double inf_ratio =
        read_double("--inf-ratio", argc, argv, argr, default_inf_ratio);

    if (inf_ratio < 0.0 && 1.0 < inf_ratio) {
        fprintf(stderr, "Invalid infinite eigenvalue ratio.\n");
        return -1;
    }

    return 0;
}

static void bulk_complex_distr_print_args(int argc, char * const *argv)
{
    if (read_opt("--fortify", argc, argv, NULL))
        printf(" --fortify");

    if (read_opt("--bulk-complex-top", argc, argv, NULL)) {
        printf(" --bulk-complex-top");
        return;
    }

    if (read_opt("--bulk-complex-bottom", argc, argv, NULL)) {
        printf(" --bulk-complex-bottom");
        return;
    }

    print_multiarg("--bulk-complex-begin", argc, argv, "top", "middle", NULL);
    print_multiarg("--bulk-complex-end", argc, argv, "middle", "bottom", NULL);

    printf(" --zero-ratio %f",
        read_double("--zero-ratio", argc, argv, NULL, default_zero_ratio));

    printf(" --inf-ratio %f",
        read_double("--inf-ratio", argc, argv, NULL, default_inf_ratio));
}

static int bulk_complex_distr_init(
    int argc, char * const *argv, matrix_t A, matrix_t B)
{
    int fortify = read_opt("--fortify", argc, argv, NULL);

    int n = GENERIC_MATRIX_N(A);

    int begin = 0, end = n;
    if (read_opt("--bulk-complex-top", argc, argv, NULL)) {
        begin = 0;
        end = n/2;
    }
    else if (read_opt("--bulk-complex-bottom", argc, argv, NULL)) {
        begin = n/2;
        end = n;
    }
    else {
        struct multiarg_t begin_arg = read_multiarg(
            "--bulk-complex-begin", argc, argv, NULL, "top", "middle", NULL);
        struct multiarg_t end_arg = read_multiarg(
            "--bulk-complex-end", argc, argv, NULL, "middle", "bottom", NULL);

        if (begin_arg.type == MULTIARG_STR &&
        strcmp("top", begin_arg.str_value) == 0)
            begin = 0;
        if (begin_arg.type == MULTIARG_STR &&
        strcmp("middle", begin_arg.str_value) == 0)
            begin = n/2;
        if (begin_arg.type == MULTIARG_INT)
            begin = MIN(n, begin_arg.int_value);

        if (end_arg.type == MULTIARG_STR &&
        strcmp("middle", end_arg.str_value) == 0)
            end = n/2;
        if (end_arg.type == MULTIARG_STR &&
        strcmp("bottom", end_arg.str_value) == 0)
            end = n;
        if (end_arg.type == MULTIARG_INT)
            end = MIN(n, MAX(begin, end_arg.int_value));
    }

    double complex_ratio = 1.0*n/(end-begin);
    double zero_ratio =
        read_double("--zero-ratio", argc, argv, NULL, default_zero_ratio);
    double inf_ratio =
        read_double("--inf-ratio", argc, argv, NULL, default_inf_ratio);

    double *real = malloc(n*sizeof(double));
    double *imag = malloc(n*sizeof(double));
    double *beta = malloc(n*sizeof(double));

    // place the 1-by-1 blocks
    if (fortify) {
        for (int i = 0; i < n; i++) {
            real[i] = 2.0*(i - n/2 + 0.5);
            imag[i] = 0.0;
            beta[i] = 1.0;
        }
    }
    else {
        for (int i = 0; i < n; i++) {
            real[i] = (2.0*prand()/PRAND_MAX-1.0) * n;
            imag[i] = 0.0;
            if (B != NULL)
                beta[i] = 1.0*prand()/PRAND_MAX;
            else
                beta[i] = 1.0;
        }
    }

    // place the 2-by-2 blocks into the diagonal
    if (fortify) {
        for (int i = begin; i+1 < end; i += 2) {
            imag[i]   =  fabs(real[i]);
            real[i+1] =       real[i];
            imag[i+1] =      -imag[i];
        }
    }
    else {
        for (int i = begin; i+1 < end; i += 2) {
            real[i] = (2.0*prand()/PRAND_MAX-1.0) * n;
            imag[i] = (2.0*prand()/PRAND_MAX-1.0) * n;
            real[i+1] =  real[i];
            imag[i+1] = -imag[i];
            beta[i] = beta[i+1] = 1.0*prand()/PRAND_MAX;
        }
    }

    if (!fortify)
        generate_special_cases(n, complex_ratio, zero_ratio, inf_ratio,
            real, imag, beta);

    block_placer(real, imag, beta, A, B);

    free(real);
    free(imag);
    free(beta);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////

static const struct complex_distr complex_distrs[] = {
    { .name = "uniform",
        .desc = "Uniform distribution module",
        .print_usage = &uniform_complex_distr_print_usage,
        .check_args = &uniform_complex_distr_check_args,
        .print_args = &uniform_complex_distr_print_args,
        .init = &uniform_complex_distr_init
    },
    { .name = "bulk",
        .desc = "Bulk distribution module",
        .print_usage = &bulk_complex_distr_print_usage,
        .check_args = &bulk_complex_distr_check_args,
        .print_args = &bulk_complex_distr_print_args,
        .init = &bulk_complex_distr_init
    }
};

PRINT_AVAIL(print_avail_complex_distr,
    "Available 2-by-2 block distribution modules:",
    name, desc, complex_distrs, 0)

PRINT_OPT(print_opt_complex_distr,
    "2-by-2 block distribution module specific options:",
    name, print_usage, complex_distrs)

READ_FROM_ARGV(
    read_complex_distr, struct complex_distr const, name, complex_distrs, 0)
