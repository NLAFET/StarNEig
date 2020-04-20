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
#include "init.h"
#include "common.h"
#include "parse.h"
#include "crawler.h"
#include "local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "starneig_pencil.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

struct init_helper {
    matrix_type_t type;
    data_type_t dtype;
#ifdef STARNEIG_ENABLE_MPI
    starneig_distr_t distr;
    int section_height;
    int section_width;
#endif
};

static int crawl_zero_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];
    for (int i = 0; i < width; i++)
        for (int j = 0; j < m; j++)
            A[i*ldA+j] = 0.0;

    return width;
}

static int crawl_identity_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < m; j++) {
            if (offset+i == j)
                A[i*ldA+j] = 1.0;
            else
                A[i*ldA+j] = 0.0;
        }
    }

    return width;
}

static int crawl_random_full_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++)
        for (int j = 0; j < m; j++)
            A[i*ldA+j] = 2.0*(1.0*prand()/PRAND_MAX)-1.0;

    return width;
}

static int crawl_random_fullpos_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++)
        for (int j = 0; j < m; j++)
            A[i*ldA+j] = 1.0*prand()/PRAND_MAX;

    return width;
}

static int crawl_random_uptriag_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++) {
        int end = MIN(m, offset+i+1);
        for (int j = 0; j < end; j++)
            A[i*ldA+j] = 2.0*(1.0*prand()/PRAND_MAX)-1.0;
        for (int j = end; j < m; j++)
            A[i*ldA+j] = 0.0;
    }

    return width;
}

static int crawl_random_uptriagpos_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++) {
        int end = MIN(m, offset+i);
        for (int j = 0; j < end; j++)
            A[i*ldA+j] = 2.0*(1.0*prand()/PRAND_MAX)-1.0;
        A[i*ldA+end] = prand()/PRAND_MAX;
        for (int j = end+1; j < m; j++)
            A[i*ldA+j] = 0.0;
    }

    return width;
}

static int crawl_random_hessenberg_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++) {
        int end = MIN(m, offset+i+2);
        for (int j = 0; j < end; j++)
            A[i*ldA+j] = 2.0*(1.0*prand()/PRAND_MAX)-1.0;
        for (int j = end; j < m; j++)
            A[i*ldA+j] = 0.0;
    }

    return width;
}

static int crawl_householder_dr(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *vec = arg;
    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < m; j++) {
            if (offset+i == j)
                A[i*ldA+j] = 1.0 - 2.0*vec[offset+i]*vec[j];
            else
                A[i*ldA+j] = -2.0*vec[offset+i]*vec[j];
        }
    }

    return width;
}

///
/// @brief Scales a vector to unit length.
///
/// @param[in] n
///         Row count.
///
/// @param[inout] vec
///         A pointer to the vector.
///
static void scale_to_unit_dr(int n, double *vec)
{
    double scal = 0.0;
    for (int i = 0; i < n; i++)
        scal += vec[i]*vec[i];
    scal = 1.0/sqrt(scal);
    for (int i = 0; i < n; i++)
        vec[i] *= scal;
}

#ifdef STARNEIG_ENABLE_MPI
static void get_strs(char const *prefix, char **str1, char **str2, char **str3)
{
    *str1 = malloc(strlen(prefix) + strlen("--data-distr") + 1);
    sprintf(*str1, "--%sdata-distr", prefix);
    *str2 = malloc(strlen(prefix) + strlen("--section-height") + 1);
    sprintf(*str2, "--%ssection-height", prefix);
    *str3 = malloc(strlen(prefix) + strlen("--section-width") + 1);
    sprintf(*str3, "--%ssection-width", prefix);
}
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void init_helper_print_usage(
    char const *prefix, init_helper_mode_t mode, int argc, char * const *argv)
{
#ifdef STARNEIG_ENABLE_MPI
    if (mode == INIT_HELPER_ALL || mode == INIT_HELPER_STARNEIG_PENCIL)
        printf(
            "  --%sdata-distr (data distribution) -- Data distribution\n",
            prefix);

    printf(
        "  --%ssection-height [default,(num)] -- Section height\n"
        "  --%ssection-width [default,(num)] -- Section width\n",
        prefix, prefix);

    if (mode == INIT_HELPER_ALL || mode == INIT_HELPER_STARNEIG_PENCIL)
        print_avail_data_distr();
#endif
}


void init_helper_print_args(
    char const *prefix, init_helper_mode_t mode, int argc, char * const *argv)
{
#ifdef STARNEIG_ENABLE_MPI
    char *str1, *str2, *str3;
    get_strs(prefix, &str1, &str2, &str3);

    if (mode == INIT_HELPER_ALL || mode == INIT_HELPER_STARNEIG_PENCIL) {
        struct data_distr_t const *data_distr = read_data_distr(
            prefix, argc, argv, NULL);
        printf(" ");
        printf(str1, prefix);
        printf(" %s", data_distr->name);
    }

    print_multiarg(str2, argc, argv, "default", NULL);
    print_multiarg(str3, argc, argv, "default", NULL);

    free(str1); free(str2); free(str3);
#endif
}


int init_helper_check_args(
    char const *prefix, init_helper_mode_t mode, int argc, char * const *argv,
    int *argr)
{
#ifdef STARNEIG_ENABLE_MPI
    char *str1, *str2, *str3;
    get_strs(prefix, &str1, &str2, &str3);

    if (mode == INIT_HELPER_ALL || mode == INIT_HELPER_STARNEIG_PENCIL) {
        if (read_data_distr(str1, argc, argv, argr) == NULL) {
            fprintf(stderr, "Invalid data distribution.\n");
            return 1;
        }
    }

    struct multiarg_t section_height = read_multiarg(
        str2, argc, argv, argr, "default", NULL);

    if (section_height.type == MULTIARG_INVALID ||
    (section_height.type == MULTIARG_INT && section_height.int_value < 8)) {
        fprintf(stderr, "Invalid section height.\n");
        return 1;
    }

    struct multiarg_t section_width = read_multiarg(
        str3, argc, argv, argr, "default", NULL);

    if (section_width.type == MULTIARG_INVALID ||
    (section_width.type == MULTIARG_INT && section_width.int_value < 8)) {
        fprintf(stderr, "Invalid section width.\n");
        return -1;
    }

    free(str1); free(str2); free(str3);
#endif

    return 0;
}

init_helper_t init_helper_init_hook(
    char const *prefix, hook_data_format_t format, int m, int n,
    data_type_t dtype, int argc, char * const *argv)
{
    switch (format) {
        case HOOK_DATA_FORMAT_PENCIL_LOCAL:
            return init_helper_init(
                prefix, LOCAL_MATRIX, m, n, dtype, argc, argv);
#ifdef STARNEIG_ENABLE_MPI
        case HOOK_DATA_FORMAT_PENCIL_STARNEIG:
            return init_helper_init(
                prefix, STARNEIG_MATRIX, m, n, dtype, argc, argv);
        case HOOK_DATA_FORMAT_PENCIL_BLACS:
            return init_helper_init(
                prefix, BLACS_MATRIX, m, n, dtype, argc, argv);
#endif
        default:
            fprintf(stderr,
                "init_helper_init_hook() encountered an invalid hook data "
                "envelope format.\n");
            abort();
    }
}

init_helper_t init_helper_init(
    char const *prefix, matrix_type_t type, int m, int n, data_type_t dtype,
    int argc, char * const *argv)
{
    init_helper_t helper = malloc(sizeof(struct init_helper));
    helper->type = type;
    helper->dtype = dtype;
#ifdef STARNEIG_ENABLE_MPI
    helper->distr = NULL;
    helper->section_height = 0;
    helper->section_width = 0;
#endif

#ifdef STARNEIG_ENABLE_MPI
    char *str1, *str2, *str3;
    get_strs(prefix, &str1, &str2, &str3);

    if (type == STARNEIG_MATRIX || type == BLACS_MATRIX) {

        if (type == STARNEIG_MATRIX) {
            struct data_distr_t const *data_distr =
                read_data_distr(str1, argc, argv, NULL);

            if (data_distr->func != NULL)
                helper->distr =
                    starneig_distr_init_func(data_distr->func, NULL, 0);
            else
                helper->distr = starneig_distr_init();
        }
        else {
            helper->distr = starneig_distr_init();
        }

        struct multiarg_t section_height = read_multiarg(
            str2, argc, argv, NULL, "default", NULL);
        struct multiarg_t section_width = read_multiarg(
            str3, argc, argv, NULL, "default", NULL);

        if (section_height.type == MULTIARG_STR)
            helper->section_height = -1;
        else
            helper->section_height = section_height.int_value;

        if (section_width.type == MULTIARG_STR)
            helper->section_width = -1;
        else
            helper->section_width = section_width.int_value;
    }

    free(str1); free(str2); free(str3);
#endif

    return helper;
}

void init_helper_free(init_helper_t helper)
{
    if (helper != NULL) {
#ifdef STARNEIG_ENABLE_MPI
        if (helper->distr != NULL)
            starneig_distr_destroy(helper->distr);
#endif
        free(helper);
    }
}

matrix_t init_matrix(int m, int n, init_helper_t helper)
{
    switch (helper->type) {
        case LOCAL_MATRIX:
            return init_local_matrix(m, n, helper->dtype);
#ifdef STARNEIG_ENABLE_MPI
        case STARNEIG_MATRIX:
        case BLACS_MATRIX:
            return init_starneig_matrix(
                m, n, helper->section_height, helper->section_width,
                helper->dtype, helper->distr);
#endif
        default:
            fprintf(stderr,
                "init_matrix() encountered an invalid matrix type.\n");
            abort();
    }
}

void init_zero(matrix_t matrix)
{
    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_zero_dr, NULL, 0, matrix, NULL);
}

void init_identity(matrix_t matrix)
{
    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_identity_dr, NULL, 0, matrix, NULL);
}

void init_random_full(matrix_t matrix)
{
    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_random_full_dr, NULL, 0, matrix, NULL);
}

void init_random_fullpos(matrix_t matrix)
{
    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_random_fullpos_dr, NULL, 0, matrix,
        NULL);
}

matrix_t generate_zero(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);
    init_zero(desc);
    return desc;
}

matrix_t generate_identity(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);
    init_identity(desc);
    return desc;
}

matrix_t generate_random_full(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);
    init_random_full(desc);
    return desc;
}

matrix_t generate_random_fullpos(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);
    init_random_fullpos(desc);
    return desc;
}

matrix_t generate_random_uptriag(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);

    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_random_uptriag_dr, NULL, 0, desc,
        NULL);

    return desc;
}

matrix_t generate_random_uptriagpos(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);

    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_random_uptriagpos_dr, NULL, 0, desc,
        NULL);

    return desc;
}

matrix_t generate_random_hessenberg(int m, int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(m, n, helper);

    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_random_hessenberg_dr, NULL, 0, desc,
        NULL);

    return desc;
}

matrix_t generate_random_householder(int n, init_helper_t helper)
{
    assert(check_data_type_against(helper->dtype, PREC_DOUBLE, NUM_REAL));

    matrix_t desc = init_matrix(n, n, helper);

    double *vec_v = malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
        vec_v[i] = 2.0*(1.0*prand()/PRAND_MAX)-1.0;
    scale_to_unit_dr(n, vec_v);

    crawl_matrices(
        CRAWLER_W, CRAWLER_PANEL, &crawl_householder_dr,
        vec_v, 0, desc, NULL);

    free(vec_v);

    return desc;
}

void mul_QAZT(
    matrix_t mat_q, matrix_t mat_a, matrix_t mat_z, matrix_t *mat_c)
{
    matrix_t tmp = NULL;
    mul_C_AB("N", "N", 1.0, mat_q, mat_a, 0.0, &tmp);
    mul_C_AB("N", "T", 1.0, tmp, mat_z, 0.0, mat_c);
    free_matrix_descr(tmp);
}
