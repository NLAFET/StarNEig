///
/// @file This file contains definitions of an opaque matrix object and an
/// opaque matrix pencil.
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
#include "pencil.h"
#include "init.h"
#include "math.h"
#include "crawler.h"
#include "local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "starneig_pencil.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static struct pencil_handler const *handlers[] = {
    &local_handler,
#ifdef STARNEIG_ENABLE_MPI
    &starneig_handler,
    &blacs_handler,
#endif
    0
};

static struct pencil_handler const * get_handler(matrix_type_t type)
{
    for (struct pencil_handler const **i = handlers; *i != NULL; i++)
        if ((*i)->type == type)
            return *i;

    return NULL;
}

static int identity_crawler(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    int *ret = arg;

    if (*ret == 0)
        return -1;

    double *A = ptrs[0];
    size_t ldA = lds[0];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < m; j++) {
            if (offset+i == j) {
                if (A[i*ldA+j] != 1.0) {
                    *ret = 0;
                    return -1;
                }
            }
            else if (A[i*ldA+j] != 0.0) {
                *ret = 0;
                return -1;
            }
        }
    }

    return width;
}

struct norm_crawler_arg {
    double scale;
    double sumsq;
};

static int norm_crawler(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    extern double dlassq_(
        int const *, double const *, int const *, double *, double *);

    struct norm_crawler_arg *ret = arg;

    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++)
        dlassq_(&m, &A[i*ldA], (int[]){1}, &ret->scale, &ret->sumsq);

    return width;
}

static int print_crawler(
    int offset, int height, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    FILE *stream = arg;

    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < n; j++) {
            if(A[j*ldA+i] != 0.0)
                fprintf(stream, "%10f ", A[j*ldA+i]);
            else
                fprintf(stream, "  -------- ");
        }
        printf("\n");
    }

    return height;
}

///
/// @brief Checks whether a matrix characterizes an identity.
///
/// @param[in] matrix
///         Matrix.
///
/// @return 0 if the matrix is not an identity; non-zero otherwise
///
static int is_identity(const matrix_t matrix)
{
    assert(check_data_type_against(matrix->dtype, PREC_DOUBLE, NUM_REAL));

    int ret = 1;
    crawl_matrices(
        CRAWLER_R, CRAWLER_PANEL, &identity_crawler,
        &ret, sizeof(ret), matrix, NULL);

    return ret;
}

void free_matrix_descr(matrix_t matrix)
{
    if (matrix == NULL)
        return;

    struct pencil_handler const *handler = get_handler(matrix->type);
    if (handler == NULL || handler->free == NULL) {
        fprintf(stderr,
            "free_matrix_descr encountered an invalid matrix.\n");
        abort();
    }

    handler->free(matrix->ptr);
    free(matrix);
}

matrix_t copy_matrix_descr(const matrix_t matrix)
{
    if (matrix == NULL)
        return NULL;

    struct pencil_handler const *handler = get_handler(matrix->type);
    if (handler == NULL || handler->copy == NULL) {
        fprintf(stderr,
            "copy_matrix_descr encountered an invalid matrix.\n");
        abort();
    }

    struct matrix *new = malloc(sizeof(struct matrix));
    new->type = matrix->type;
    new->dtype = matrix->dtype;
    new->ptr = handler->copy(matrix->ptr);
    return new;
}

size_t GENERIC_MATRIX_M(const matrix_t matrix)
{
    struct pencil_handler const *handler = get_handler(matrix->type);
    if (handler == NULL || handler->get_rows == NULL) {
        fprintf(stderr,
            "GENERIC_MATRIX_M encountered an invalid matrix.\n");
        abort();
    }
    return handler->get_rows(matrix);
}

size_t GENERIC_MATRIX_N(const matrix_t matrix)
{
    struct pencil_handler const *handler = get_handler(matrix->type);
    if (handler == NULL || handler->get_cols == NULL) {
        fprintf(stderr,
            "GENERIC_MATRIX_M encountered an invalid matrix.\n");
        abort();
    }
    return handler->get_cols(matrix);
}

void mul_C_AB(
    char const *trans_a, char const *trans_b, double alpha,
    const matrix_t mat_a, const matrix_t mat_b, double beta, matrix_t *mat_c)
{
    struct pencil_handler const *handler = get_handler(mat_a->type);
    if (handler == NULL || handler->gemm == NULL) {
        fprintf(stderr,
            "mul_C_AB encountered an invalid matrix.\n");
        abort();
    }
    return handler->gemm(trans_a, trans_b, alpha, mat_a, mat_b, beta, mat_c);
}

double norm_C(const matrix_t matrix)
{
    assert(check_data_type_against(matrix->dtype, PREC_DOUBLE, NUM_REAL));

    struct norm_crawler_arg arg = { .scale = 0.0, .sumsq = 1.0 };

    crawl_matrices(
        CRAWLER_R, CRAWLER_PANEL, &norm_crawler, &arg, sizeof(arg), matrix,
        NULL);

    return arg.scale * sqrt(arg.sumsq);
}

void print_matrix_descr(const matrix_t matrix, FILE * stream)
{
    crawl_matrices(
        CRAWLER_R, CRAWLER_HPANEL, &print_crawler, stream, 0, matrix, NULL);
}

pencil_t init_pencil()
{
    pencil_t pencil = malloc(sizeof(struct pencil));
    memset(pencil, 0, sizeof(struct pencil));
    return pencil;
}

void free_pencil(pencil_t pencil)
{
    free_matrix_descr(pencil->mat_a);
    free_matrix_descr(pencil->mat_b);
    free_matrix_descr(pencil->mat_q);
    free_matrix_descr(pencil->mat_z);
    free_matrix_descr(pencil->mat_x);
    free_matrix_descr(pencil->mat_ca);
    free_matrix_descr(pencil->mat_cb);
    free_supplementary(pencil->supp);
    free(pencil);
}

pencil_t copy_pencil(const pencil_t pencil)
{
    pencil_t new = init_pencil();
    new->mat_a = copy_matrix_descr(pencil->mat_a);
    new->mat_b = copy_matrix_descr(pencil->mat_b);
    new->mat_q = copy_matrix_descr(pencil->mat_q);
    new->mat_z = copy_matrix_descr(pencil->mat_z);
    new->mat_x = copy_matrix_descr(pencil->mat_x);
    new->mat_ca = copy_matrix_descr(pencil->mat_ca);
    new->mat_cb = copy_matrix_descr(pencil->mat_cb);
    new->supp = copy_supplementary(pencil->supp);

    return new;
}

void fill_pencil(pencil_t pencil)
{
    if (pencil == NULL)
        return;

    if (pencil->mat_b != NULL) {
        if (is_identity(pencil->mat_q) && is_identity(pencil->mat_z)) {
            if (pencil->mat_ca == NULL)
                pencil->mat_ca = copy_matrix_descr(pencil->mat_a);
            if (pencil->mat_cb == NULL)
                pencil->mat_cb = copy_matrix_descr(pencil->mat_b);
        }
        else {
            if (pencil->mat_ca == NULL)
                mul_QAZT(
                    pencil->mat_q, pencil->mat_a, pencil->mat_z,
                    &pencil->mat_ca);
            if (pencil->mat_cb == NULL)
                mul_QAZT(
                    pencil->mat_q, pencil->mat_b, pencil->mat_z,
                    &pencil->mat_cb);
        }
    }
    else {
        if (is_identity(pencil->mat_q)) {
            if (pencil->mat_ca == NULL)
                pencil->mat_ca = copy_matrix_descr(pencil->mat_a);
        }
        else {
            if (pencil->mat_ca == NULL)
                mul_QAZT(
                    pencil->mat_q, pencil->mat_a, pencil->mat_q,
                    &pencil->mat_ca);
        }
    }
}
