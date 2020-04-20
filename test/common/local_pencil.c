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
#include "local_pencil.h"
#include "common.h"
#include "init.h"
#include "threads.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

///
/// @brief Local matrix.
///
struct local_matrix {
    void *ptr;                 ///< A pointer to the first element.
    size_t m;                  ///< The row count.
    size_t n;                  ///< The column count.
    size_t ld;                 ///< The leading dimension.
    size_t elemsize;           ///< The element size.
};

typedef struct local_matrix * local_matrix_t;

static void free_local_matrix(local_matrix_t descr)
{
    if (descr == NULL)
        return;

    free_matrix(descr->ptr);
    free(descr);
}

static local_matrix_t copy_local_matrix(const local_matrix_t descr)
{
    if (descr == NULL)
        return NULL;

    local_matrix_t new = malloc(sizeof(struct local_matrix));
    new->m = descr->m;
    new->n = descr->n;
    new->elemsize = descr->elemsize;
    new->ptr =
        alloc_matrix(new->m, new->n, new->elemsize, &new->ld);

    copy_matrix(descr->m, descr->n, descr->ld, new->ld, descr->elemsize,
        descr->ptr, new->ptr);

    return new;
}

static void mul_local_C_AB(
    char const *trans_a, char const *trans_b, double alpha,
    const matrix_t mat_a, const matrix_t mat_b, double beta, matrix_t *mat_c)
{
    assert(check_data_type_against(mat_a->dtype, PREC_DOUBLE, NUM_REAL));
    assert(check_data_type_against(mat_b->dtype, PREC_DOUBLE, NUM_REAL));
    assert(*mat_c == NULL ||
        check_data_type_against((*mat_c)->dtype, PREC_DOUBLE, NUM_REAL));

    // no-trans no-trans test
    assert (*trans_a == 'T' || *trans_b == 'T' ||
        LOCAL_MATRIX_N(mat_a) == LOCAL_MATRIX_M(mat_b));

    // no-trans trans test
    assert (*trans_a == 'T' || *trans_b == 'N' ||
        LOCAL_MATRIX_N(mat_a) == LOCAL_MATRIX_N(mat_b));

    // trans no-trans test
    assert (*trans_a == 'N' || *trans_b == 'T' ||
        LOCAL_MATRIX_M(mat_a) == LOCAL_MATRIX_M(mat_b));

    // trans trans test
    assert (*trans_a == 'N' || *trans_b == 'N' ||
        LOCAL_MATRIX_M(mat_a) == LOCAL_MATRIX_N(mat_b));

    int m = *trans_a == 'T' ? LOCAL_MATRIX_N(mat_a) : LOCAL_MATRIX_M(mat_a);
    int n = *trans_b == 'T' ? LOCAL_MATRIX_M(mat_b) : LOCAL_MATRIX_N(mat_b);
    int k = *trans_a == 'T' ? LOCAL_MATRIX_M(mat_a) : LOCAL_MATRIX_N(mat_a);

    if (*mat_c == NULL)
        *mat_c = init_local_matrix(m, n, NUM_REAL | PREC_DOUBLE);

    assert(LOCAL_MATRIX_M(*mat_c) == m && LOCAL_MATRIX_N(*mat_c) == n);

    threads_set_mode(THREADS_MODE_BLAS);

    dgemm(trans_a, trans_b, m, n, k, alpha,
        LOCAL_MATRIX_PTR(mat_a), LOCAL_MATRIX_LD(mat_a),
        (double *)LOCAL_MATRIX_PTR(mat_b),
        LOCAL_MATRIX_LD(mat_b), beta,
        (double *)LOCAL_MATRIX_PTR(*mat_c),
        LOCAL_MATRIX_LD(*mat_c));

    threads_set_mode(THREADS_MODE_DEFAULT);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

matrix_t init_local_matrix(int m, int n, data_type_t dtype)
{
    assert(1 <= m && 1 <= n);

    matrix_t descr = malloc(sizeof(struct matrix));
    descr->type = LOCAL_MATRIX;
    descr->dtype = dtype;

    local_matrix_t local_descr = malloc(sizeof(struct local_matrix));
    local_descr->m = m;
    local_descr->n = n;
    local_descr->ptr =
        alloc_matrix(m, n, data_type_size(dtype), &local_descr->ld);
    local_descr->elemsize = data_type_size(dtype);

    descr->ptr = local_descr;

    return descr;
}

void* LOCAL_MATRIX_PTR(const matrix_t descr)
{
    if (descr != NULL) {
        assert(descr->type == LOCAL_MATRIX);
        return ((const local_matrix_t)descr->ptr)->ptr;
    }
    return NULL;
}

size_t LOCAL_MATRIX_M(const matrix_t descr)
{
    if (descr != NULL) {
        assert(descr->type == LOCAL_MATRIX);
        return ((const local_matrix_t)descr->ptr)->m;
    }
    return 0;
}

size_t LOCAL_MATRIX_N(const matrix_t descr)
{
    if (descr != NULL) {
        assert(descr->type == LOCAL_MATRIX);
        return ((const local_matrix_t)descr->ptr)->n;
    }
    return 0;
}

size_t LOCAL_MATRIX_LD(const matrix_t descr)
{
    if (descr != NULL) {
        assert(descr->type == LOCAL_MATRIX);
        return ((const local_matrix_t)descr->ptr)->ld;
    }
    return 0;
}

struct pencil_handler local_handler = {
    .type = LOCAL_MATRIX,
    .copy = (matrix_copy_t) copy_local_matrix,
    .free = (matrix_free_t) free_local_matrix,
    .get_rows = LOCAL_MATRIX_M,
    .get_cols = LOCAL_MATRIX_N,
    .gemm = mul_local_C_AB
};
