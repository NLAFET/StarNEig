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
#include "starneig_pencil.h"
#include "local_pencil.h"
#include "hook_converter.h"
#include "common.h"
#include "parse.h"
#include "init.h"
#include "threads.h"
#include "crawler.h"
#include <starneig/starneig.h>
#include <assert.h>
#include <mpi.h>

static inline starneig_datatype_t get_datatype(data_type_t type)
{
    if (check_data_type_against(type, PREC_DOUBLE, NUM_REAL))
        return STARNEIG_REAL_DOUBLE;
    assert(0);
}

static void free_starneig_matrix(starneig_distr_matrix_t matrix)
{
    starneig_distr_matrix_destroy(matrix);
}

static starneig_distr_matrix_t copy_starneig_matrix(
    starneig_distr_matrix_t matrix)
{
    int initialized = starneig_node_initialized();
    if (!initialized)
        starneig_node_init(threads_get_workers(), 0,
            STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    starneig_distr_matrix_t new = starneig_distr_matrix_create(
        starneig_distr_matrix_get_rows(matrix),
        starneig_distr_matrix_get_cols(matrix),
        starneig_distr_matrix_get_row_blksz(matrix),
        starneig_distr_matrix_get_col_blksz(matrix),
        starneig_distr_matrix_get_datatype(matrix),
        starneig_distr_matrix_get_distr(matrix));

    starneig_distr_matrix_copy(matrix, new);

    if (!initialized)
        starneig_node_finalize();

    return new;
}

#ifdef STARNEIG_ENABLE_BLACS

static void mul_blacs_C_AB(
    char const *trans_a, char const *trans_b, double alpha,
    const matrix_t mat_a, const matrix_t mat_b, double beta, matrix_t *mat_c)
{
    assert(mat_a->type == BLACS_MATRIX);
    assert(mat_b->type == BLACS_MATRIX);
    assert(*mat_c == NULL || (*mat_c)->type == BLACS_MATRIX);

    assert(check_data_type_against(mat_a->dtype, PREC_DOUBLE, NUM_REAL));
    assert(check_data_type_against(mat_b->dtype, PREC_DOUBLE, NUM_REAL));
    assert(*mat_c == NULL ||
        check_data_type_against((*mat_c)->dtype, PREC_DOUBLE, NUM_REAL));

    // no-trans no-trans test
    assert (*trans_a == 'T' || *trans_b == 'T' ||
        (
            STARNEIG_MATRIX_N(mat_a) == STARNEIG_MATRIX_M(mat_b) &&
            STARNEIG_MATRIX_BN(mat_a) == STARNEIG_MATRIX_BM(mat_b)
        ));

    // no-trans trans test
    assert (*trans_a == 'T' || *trans_b == 'N' ||
        (
            STARNEIG_MATRIX_N(mat_a) == STARNEIG_MATRIX_N(mat_b) &&
            STARNEIG_MATRIX_BN(mat_a) == STARNEIG_MATRIX_BN(mat_b)
        ));

    // trans no-trans test
    assert (*trans_a == 'N' || *trans_b == 'T' ||
        (
            STARNEIG_MATRIX_M(mat_a) == STARNEIG_MATRIX_M(mat_b) &&
            STARNEIG_MATRIX_BM(mat_a) == STARNEIG_MATRIX_BM(mat_b)
        ));

    // trans trans test
    assert (*trans_a == 'N' || *trans_b == 'N' ||
        (
            STARNEIG_MATRIX_M(mat_a) == STARNEIG_MATRIX_N(mat_b) &&
            STARNEIG_MATRIX_BM(mat_a) == STARNEIG_MATRIX_BN(mat_b)
        ));

    int m = *trans_a == 'T' ?
        STARNEIG_MATRIX_N(mat_a) : STARNEIG_MATRIX_M(mat_a);
    int n = *trans_b == 'T' ?
        STARNEIG_MATRIX_M(mat_b) : STARNEIG_MATRIX_N(mat_b);
    int k = *trans_a == 'T' ?
        STARNEIG_MATRIX_M(mat_a) : STARNEIG_MATRIX_N(mat_a);

    int bm = *trans_a == 'T' ?
        STARNEIG_MATRIX_BN(mat_a) : STARNEIG_MATRIX_BM(mat_a);
    int bn = *trans_b == 'T' ?
        STARNEIG_MATRIX_BM(mat_b) : STARNEIG_MATRIX_BN(mat_b);

    starneig_distr_t distr = STARNEIG_MATRIX_DISTR(mat_a);
    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    if (*mat_c == NULL)
        *mat_c = init_starneig_matrix(
            m, n, bm, bn, PREC_DOUBLE | NUM_REAL, distr);

    assert(STARNEIG_MATRIX_M(*mat_c) == m);
    assert(STARNEIG_MATRIX_N(*mat_c) == n);
    assert(STARNEIG_MATRIX_BM(*mat_c) == bm);
    assert(STARNEIG_MATRIX_BN(*mat_c) == bn);

    starneig_blacs_descr_t descr_a, descr_b, descr_c;
    double *local_a, *local_b, *local_c;

    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        mat_a, context, &descr_a, (void **)&local_a);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        mat_b, context, &descr_b, (void **)&local_b);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        *mat_c, context, &descr_c, (void **)&local_c);

    extern void pdgemm_(
        char const *, char const *, int const *, int const *, int const *,
        double const *,
        double const *, int const *, int const *,
            starneig_blacs_descr_t const *,
        double const *, int const *, int const *,
            starneig_blacs_descr_t const *,
        double const *,
        double *, int const *, int const *,
            starneig_blacs_descr_t const *);

    threads_set_mode(THREADS_MODE_BLAS);

    pdgemm_(trans_a, trans_b, &m, &n, &k,
        &alpha,
        local_a, (const int[]){1}, (const int[]){1}, &descr_a,
        local_b, (const int[]){1}, (const int[]){1}, &descr_b,
        &beta,
        local_c, (const int[]){1}, (const int[]){1}, &descr_c);

    threads_set_mode(THREADS_MODE_DEFAULT);

    starneig_blacs_gridexit(context);
}

#endif

matrix_t init_starneig_matrix(
    int m, int n, int bm, int bn, data_type_t dtype, starneig_distr_t distr)
{
    matrix_t matrix = malloc(sizeof(struct matrix));

#ifdef STARNEIG_ENABLE_BLACS
    if (starneig_distr_is_blacs_compatible(distr))
        matrix->type = BLACS_MATRIX;
    else
#endif
        matrix->type = STARNEIG_MATRIX;
    matrix->dtype = dtype;
    matrix->ptr = starneig_distr_matrix_create(
        m, n, bm, bn, get_datatype(dtype), distr);

    return matrix;
}

size_t STARNEIG_MATRIX_M(const matrix_t matrix)
{
    if (matrix != NULL) {
        assert(matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX);
        return starneig_distr_matrix_get_rows(
            (const starneig_distr_matrix_t)matrix->ptr);
    }
    return 0;
}

size_t STARNEIG_MATRIX_N(const matrix_t matrix)
{
    if (matrix != NULL) {
        assert(matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX);
        return starneig_distr_matrix_get_cols(
            (const starneig_distr_matrix_t)matrix->ptr);
    }
    return 0;
}

size_t STARNEIG_MATRIX_BM(const matrix_t matrix)
{
    if (matrix != NULL) {
        assert(matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX);
        return starneig_distr_matrix_get_row_blksz(
            (const starneig_distr_matrix_t)matrix->ptr);
    }
    return 0;
}

size_t STARNEIG_MATRIX_BN(const matrix_t matrix)
{
    if (matrix != NULL) {
        assert(matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX);
        return starneig_distr_matrix_get_col_blksz(
            (const starneig_distr_matrix_t)matrix->ptr);
    }
    return 0;
}

starneig_distr_matrix_t STARNEIG_MATRIX_HANDLE(const matrix_t matrix)
{
    if (matrix != NULL) {
        assert(matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX);
        return (starneig_distr_matrix_t)matrix->ptr;
    }
    return 0;
}

starneig_distr_t STARNEIG_MATRIX_DISTR(const matrix_t matrix)
{
    if (matrix != NULL) {
        assert(matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX);
        return starneig_distr_matrix_get_distr(
            (starneig_distr_matrix_t)matrix->ptr);
    }
    return 0;
}

#ifdef STARNEIG_ENABLE_BLACS

starneig_blacs_context_t STARNEIG_BLACS_MATRIX_CONTEXT(const matrix_t matrix)
{
    assert(matrix->type == BLACS_MATRIX);

    starneig_distr_t distr = starneig_distr_matrix_get_distr(
        (starneig_distr_matrix_t)matrix->ptr);
    return starneig_distr_to_blacs_context(distr);
}

void STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
    const matrix_t matrix, starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local)
{
    assert(matrix->type == BLACS_MATRIX);

    starneig_distr_matrix_t dA = STARNEIG_MATRIX_HANDLE(matrix);
    starneig_distr_matrix_to_blacs_descr(dA, context, descr, local);
}

#endif

struct pencil_handler starneig_handler = {
    .type = STARNEIG_MATRIX,
    .copy = (matrix_copy_t) copy_starneig_matrix,
    .free = (matrix_free_t) free_starneig_matrix,
    .get_rows = STARNEIG_MATRIX_M,
    .get_cols = STARNEIG_MATRIX_N,
};

struct pencil_handler blacs_handler = {
    .type = BLACS_MATRIX,
    .copy = (matrix_copy_t) copy_starneig_matrix,
    .free = (matrix_free_t) free_starneig_matrix,
    .get_rows = STARNEIG_MATRIX_M,
    .get_cols = STARNEIG_MATRIX_N,
#ifdef STARNEIG_ENABLE_BLACS
    .gemm = mul_blacs_C_AB
#endif
};

///
/// @brief Converts a local matrix pencil to a StarNEig matrix format and
/// scatters matrices to other MPI nodes.
///
/// @param[in] sm
///         Section height.
///
/// @param[in] sn
///         Section width.
///
/// @param[in] distrib
///         Data distribution function.
///
/// @param[in] distarg
///         Data distribution function argument.
///
/// @param[in] distarg_size
///         Data distribution function argument size.
///
/// @param[in,out] pencil
///         Local matrix pencil descriptor.
///
/// @param[in] free
///         Free local pencil.
///
/// @return StarNEig formatted distributed matrix pencil.
///
static pencil_t send_pencil(
    int sm, int sn, int (distrib)(int, int, void *), void *distarg,
    size_t distarg_size, pencil_t pencil, int free)
{
    if (pencil == NULL || pencil->mat_a == NULL)
        return NULL;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    pencil_t descr = init_pencil();

    data_type_t dtype = pencil->mat_a->dtype;
    MPI_Bcast(
        &dtype, sizeof(dtype), MPI_BYTE, my_rank, MPI_COMM_WORLD);

    starneig_distr_t distr;
    if (distrib == NULL)
        distr = starneig_distr_init();
    else
        distr = starneig_distr_init_func(distrib, distarg, distarg_size);

#define CONVERT(name) \
    descr->name = NULL; \
    if (pencil->name) { \
        int defined = 1; \
        MPI_Bcast( \
            &defined, 1, MPI_INT, my_rank, MPI_COMM_WORLD); \
        int rows = LOCAL_MATRIX_M(pencil->name); \
        int cols = LOCAL_MATRIX_N(pencil->name); \
        MPI_Bcast( \
            &rows, 1, MPI_INT, my_rank, MPI_COMM_WORLD); \
        MPI_Bcast( \
            &cols, 1, MPI_INT, my_rank, MPI_COMM_WORLD); \
        MPI_Bcast( \
            &sm, 1, MPI_INT, my_rank, MPI_COMM_WORLD); \
        MPI_Bcast( \
            &sn, 1, MPI_INT, my_rank, MPI_COMM_WORLD); \
        descr->name = init_starneig_matrix(rows, cols, sm, sn, dtype, distr); \
        starneig_distr_matrix_t local = starneig_distr_matrix_create_local( \
            rows, cols, get_datatype(dtype), my_rank, \
            LOCAL_MATRIX_PTR(pencil->name), LOCAL_MATRIX_LD(pencil->name)); \
        starneig_distr_matrix_copy(local, STARNEIG_MATRIX_HANDLE(descr->name));\
        starneig_distr_matrix_destroy(local); \
    } \
    else { \
        int defined = 0; \
        MPI_Bcast( \
            &defined, 1, MPI_INT, my_rank, MPI_COMM_WORLD); \
    } \
    if (free) { \
        free_matrix_descr(pencil->name); \
        pencil->name = NULL; \
    }

    CONVERT(mat_a);
    CONVERT(mat_b);
    CONVERT(mat_q);
    CONVERT(mat_z);
    CONVERT(mat_x);
    CONVERT(mat_ca);
    CONVERT(mat_cb);

#undef CONVERT

    descr->supp = copy_supplementary(pencil->supp);
    broadcast_supplementary(my_rank, MPI_COMM_WORLD, &descr->supp);

    if (free)
        free_pencil(pencil);

    return descr;
}

///
/// @brief Creates a StarNEig matrix pencil and receives the matrices from a
/// root node.
///
/// @param[in] sender
///         Sending MPI rank.
///
/// @param[in] distrib
///         Data distribution function.
///
/// @param[in] distarg
///         Data distribution function argument.
///
/// @param[in] distarg_size
///         Data distribution function argument size.
///
/// @return StarNEig formatted distributed matrix pencil.
///
static pencil_t receive_pencil(
    int sender, int (distrib)(int, int, void *), void *distarg,
    size_t distarg_size)
{
    pencil_t descr = init_pencil();

    data_type_t dtype;
    MPI_Bcast(
        &dtype, sizeof(dtype), MPI_BYTE, sender, MPI_COMM_WORLD);

    starneig_distr_t distr;
    if (distrib == NULL)
        distr = starneig_distr_init();
    else
        distr = starneig_distr_init_func(distrib, distarg, distarg_size);

#define RECEIVE(name)  \
    { \
        int defined = 0; \
        MPI_Bcast( \
            &defined, 1, MPI_INT, sender, MPI_COMM_WORLD); \
        if (defined) { \
            int rows, cols; \
            MPI_Bcast( \
                &rows, 1, MPI_INT, sender, MPI_COMM_WORLD); \
            MPI_Bcast( \
                &cols, 1, MPI_INT, sender, MPI_COMM_WORLD); \
            int sm, sn; \
            MPI_Bcast( \
                &sm, 1, MPI_INT, sender, MPI_COMM_WORLD); \
            MPI_Bcast( \
                &sn, 1, MPI_INT, sender, MPI_COMM_WORLD); \
            descr->name = init_starneig_matrix( \
                rows, cols, sm, sn, dtype, distr); \
            starneig_distr_matrix_t local = \
                starneig_distr_matrix_create_local( \
                    rows, cols, get_datatype(dtype), sender, NULL, 0); \
            starneig_distr_matrix_copy( \
                local, STARNEIG_MATRIX_HANDLE(descr->name)); \
            starneig_distr_matrix_destroy(local); \
        } \
        else { \
            descr->name = NULL; \
        } \
    }

    RECEIVE(mat_a);
    RECEIVE(mat_b);
    RECEIVE(mat_q);
    RECEIVE(mat_z);
    RECEIVE(mat_x);
    RECEIVE(mat_ca);
    RECEIVE(mat_cb);

#undef RECEIVE

    broadcast_supplementary(sender, MPI_COMM_WORLD, &descr->supp);

    return descr;
}

///
/// @brief Creates a local matrix pencil and gathers matrices from other MPI
/// nodes.
///
/// @param[in,out] pencil  StarNEig formatted distributed matrix pencil
/// @param[in]     free    free distributed pencil
///
/// @return local matrix pencil descriptor
///
static pencil_t receive_back_pencil(pencil_t pencil, int free)
{
    if (pencil == NULL)
        return NULL;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    pencil_t descr = init_pencil();

    data_type_t dtype = pencil->mat_a->dtype;

#define RECEIVE(name) \
    descr->name = NULL; \
    if (pencil->name != NULL) { \
        int rows = starneig_distr_matrix_get_rows( \
            STARNEIG_MATRIX_HANDLE(pencil->name)); \
        int cols = starneig_distr_matrix_get_cols( \
            STARNEIG_MATRIX_HANDLE(pencil->name)); \
        descr->name = init_local_matrix(rows, cols, dtype); \
        starneig_distr_matrix_t local = starneig_distr_matrix_create_local( \
            rows, cols, get_datatype(dtype), my_rank, \
            LOCAL_MATRIX_PTR(descr->name), LOCAL_MATRIX_LD(descr->name)); \
        starneig_distr_matrix_copy( \
            STARNEIG_MATRIX_HANDLE(pencil->name), local); \
        starneig_distr_matrix_destroy(local); \
    } \
    if (free) { \
        starneig_distr_matrix_destroy(STARNEIG_MATRIX_HANDLE(pencil->name)); \
        pencil->name = NULL; \
    }

    RECEIVE(mat_a);
    RECEIVE(mat_b);
    RECEIVE(mat_q);
    RECEIVE(mat_z);
    RECEIVE(mat_x);
    RECEIVE(mat_ca);
    RECEIVE(mat_cb);

#undef RECEIVE

    descr->supp = copy_supplementary(pencil->supp);

    if (free)
        free_pencil(pencil);

    return descr;
}

///
/// @brief Send a StarNEig formatted matrix pencil back to root node.
///
/// @param[in]     receiver  receiving MPI node
/// @param[in,out] matrix    StarNEig formatted distributed matrix pencil
/// @param[in]     free      free distributed pencil
///
static void send_back_pencil(int receiver, pencil_t pencil, int free)
{
    if (pencil == NULL)
        return;

    data_type_t dtype = pencil->mat_a->dtype;

#define SEND(name) \
    if (pencil->name != NULL) { \
        int rows = starneig_distr_matrix_get_rows( \
            STARNEIG_MATRIX_HANDLE(pencil->name)); \
        int cols = starneig_distr_matrix_get_cols( \
            STARNEIG_MATRIX_HANDLE(pencil->name)); \
        starneig_distr_matrix_t local = starneig_distr_matrix_create_local( \
            rows, cols, get_datatype(dtype), receiver, NULL, 0); \
        starneig_distr_matrix_copy( \
            STARNEIG_MATRIX_HANDLE(pencil->name), local); \
        starneig_distr_matrix_destroy(local); \
    } \
    if (free) { \
        starneig_distr_matrix_destroy(STARNEIG_MATRIX_HANDLE(pencil->name)); \
        pencil->name = NULL; \
    }

    SEND(mat_a);
    SEND(mat_b);
    SEND(mat_q);
    SEND(mat_z);
    SEND(mat_x);
    SEND(mat_ca);
    SEND(mat_cb);

#undef SEND

    if (free)
        free_pencil(pencil);
}

starneig_distr_matrix_t copy_distr_matrix(starneig_distr_matrix_t descr)
{
    if (descr == NULL)
        return NULL;

    starneig_distr_matrix_t new = starneig_distr_matrix_create(
        starneig_distr_matrix_get_rows(descr),
        starneig_distr_matrix_get_cols(descr),
        starneig_distr_matrix_get_row_blksz(descr),
        starneig_distr_matrix_get_col_blksz(descr),
        starneig_distr_matrix_get_elemsize(descr),
        starneig_distr_matrix_get_distr(descr));

    starneig_distr_matrix_copy(descr, new);

    return new;
}

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Symmetric data distribution.
///
///  0 1 2 3 4 ...
///  1 2 3 4 5 ...
///  2 3 4 5 6 ...
///  3 4 5 6 7 ...
///  4 5 6 7 8 ...
///  . . . . . .
///
static int symmetric_distrib(int i, int j, void *arg)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    return (i % world_size + j) % world_size;
}

///
/// @brief Two-dimensional block cyclic data distribution
///
///  0 1 0 1 0 1 ...
///  2 3 2 3 2 3 ...
///  0 1 0 1 0 1 ...
///  2 3 2 3 2 3 ...
///  . . . . . . .
///
static int block_cyclic_distrib(int i, int j, void *arg)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int sp = ceil(sqrt(world_size));
    while (world_size % sp != 0)
        sp++;

    return (i*sp + j%sp) % world_size;
}

///
/// @brief Generalized two-dimensional block cyclic data distribution
///
///  0 1 0 1 0 1 ...
///  2 3 2 3 2 3 ...
///  0 1 0 1 0 1 ...
///  2 3 2 3 2 3 ...
///  . . . . . . .
///
static int generalized_block_cyclic_distrib(int i, int j, void *arg)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int sp = ceil(sqrt(world_size));

    return (i*sp + j%sp) % world_size;
}

///
/// @brief Data distributions.
///
static const struct data_distr_t data_distr[] = {
    { .name = "default",
        .desc = "Default data distribution",
        .func = NULL
    },
    { .name = "bcd",
        .desc = "Two-dimensional block cyclic distribution",
        .func = &block_cyclic_distrib
    },
    { .name = "gbcd",
        .desc = "Generalized two-dimensional block cyclic data distribution",
        .func = &generalized_block_cyclic_distrib
    },
    { .name = "symmetric",
        .desc = "Symmetric data distribution",
        .func = &symmetric_distrib
    }
};

PRINT_AVAIL(print_avail_data_distr, "  Available data distributions:",
    name, desc, data_distr, 0)

READ_FROM_ARGV(read_data_distr, struct data_distr_t const,
    name, data_distr, 0)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void local_starneig_print_usage(int argc, char * const *argv)
{
    printf(
        "  --data-distr (data distribution) -- Data distribution\n"
        "  --section-height [default,(num)] -- Section height\n"
        "  --section-width [default,(num)] -- Section width\n"
    );

    print_avail_data_distr();
}

static void local_starneig_print_args(int argc, char * const *argv)
{
    struct data_distr_t const *data_distr = read_data_distr(
        "--data-distr", argc, argv, NULL);
    printf(" --data-distr %s", data_distr->name);

    print_multiarg("--section-height", argc, argv, "default", NULL);
    print_multiarg("--section-width", argc, argv, "default", NULL);
}

static int local_starneig_check_args(int argc, char * const *argv, int *argr)
{
    if (read_data_distr("--data-distr", argc, argv, argr) == NULL) {
        fprintf(stderr, "Invalid data distribution.\n");
        return -1;
    }

    struct multiarg_t section_height = read_multiarg(
        "--section-height", argc, argv, argr, "default", NULL);

    if (section_height.type == MULTIARG_INVALID ||
    (section_height.type == MULTIARG_INT && section_height.int_value < 8)) {
        fprintf(stderr, "Invalid section height.\n");
        return -1;
    }

    struct multiarg_t section_width = read_multiarg(
        "--section-width", argc, argv, argr, "default", NULL);

    if (section_width.type == MULTIARG_INVALID ||
    (section_width.type == MULTIARG_INT && section_width.int_value < 8)) {
        fprintf(stderr, "Invalid section width.\n");
        return -1;
    }

    return 0;
}

static int local_starneig_convert(
    int argc, char * const *argv, struct hook_data_env *env)
{
    struct data_distr_t const *data_distr = read_data_distr(
        "--data-distr", argc, argv, NULL);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    env->format = HOOK_DATA_FORMAT_PENCIL_STARNEIG;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;

    starneig_node_init(threads_get_workers(), 0,
        STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    if (my_rank == 0) {
        struct multiarg_t section_height = read_multiarg(
            "--section-height", argc, argv, NULL, "default", NULL);

        struct multiarg_t section_width = read_multiarg(
            "--section-width", argc, argv, NULL, "default", NULL);

        int sm;
        if (section_height.type == MULTIARG_STR)
            sm = -1;
        else
            sm = section_height.int_value;

        int sn;
        if (section_width.type == MULTIARG_STR)
            sn = -1;
        else
            sn = section_width.int_value;

        env->data = send_pencil(
            sm, sn, data_distr->func, NULL, 0, env->data, 1);
    }
    else {
        env->data = receive_pencil(0, data_distr->func, NULL, 0);
    }

    starneig_node_finalize();

    return 0;
}

static int starneig_local_convert(
    int argc, char * const *argv, struct hook_data_env *env)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    env->format = HOOK_DATA_FORMAT_PENCIL_LOCAL;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;

    starneig_node_init(threads_get_workers(), 0,
        STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    if (my_rank == 0) {
        env->data = receive_back_pencil(env->data, 1);
    }
    else {
        send_back_pencil(0, env->data, 1);
        env->data = NULL;
    }

    starneig_node_finalize();

    return 0;
}

const struct hook_data_converter local_starneig_converter = {
    .from = HOOK_DATA_FORMAT_PENCIL_LOCAL,
    .to = HOOK_DATA_FORMAT_PENCIL_STARNEIG,
    .print_usage = local_starneig_print_usage,
    .print_args = local_starneig_print_args,
    .check_args = local_starneig_check_args,
    .convert = local_starneig_convert
};

const struct hook_data_converter starneig_local_converter = {
    .from = HOOK_DATA_FORMAT_PENCIL_STARNEIG,
    .to = HOOK_DATA_FORMAT_PENCIL_LOCAL,
    .convert = starneig_local_convert
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void local_blacs_print_usage(int argc, char * const *argv)
{
    printf(
        "  --section-height [default,(num)] -- Section height\n"
        "  --section-width [default,(num)] -- Section width\n"
    );
}

static void local_blacs_print_args(int argc, char * const *argv)
{
    print_multiarg("--section-height", argc, argv, "default", NULL);
    print_multiarg("--section-width", argc, argv, "default", NULL);
}

static int local_blacs_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t section_height = read_multiarg(
        "--section-height", argc, argv, argr, "default", NULL);

    if (section_height.type == MULTIARG_INVALID ||
    (section_height.type == MULTIARG_INT && section_height.int_value < 8)) {
        fprintf(stderr, "Invalid section height.\n");
        return -1;
    }

    struct multiarg_t section_width = read_multiarg(
        "--section-width", argc, argv, argr, "default", NULL);

    if (section_width.type == MULTIARG_INVALID ||
    (section_width.type == MULTIARG_INT && section_width.int_value < 8)) {
        fprintf(stderr, "Invalid section width.\n");
        return -1;
    }

    return 0;
}

static int local_blacs_convert(
    int argc, char * const *argv, struct hook_data_env *env)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    env->format = HOOK_DATA_FORMAT_PENCIL_BLACS;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;

    starneig_node_init(threads_get_workers(), 0,
        STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    if (my_rank == 0) {
        struct multiarg_t section_height = read_multiarg(
            "--section-height", argc, argv, NULL, "default", NULL);

        struct multiarg_t section_width = read_multiarg(
            "--section-width", argc, argv, NULL, "default", NULL);

        int sm;
        if (section_height.type == MULTIARG_STR)
            sm = -1;
        else
            sm = section_height.int_value;

        int sn;
        if (section_width.type == MULTIARG_STR)
            sn = -1;
        else
            sn = section_width.int_value;

        env->data = send_pencil(sm, sn, NULL, NULL, 0, env->data, 1);
    }
    else {
        env->data = receive_pencil(0, NULL, NULL, 0);
    }

    starneig_node_finalize();

    return 0;
}

static int blacs_local_convert(
    int argc, char * const *argv, struct hook_data_env *env)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    env->format = HOOK_DATA_FORMAT_PENCIL_LOCAL;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;

    starneig_node_init(threads_get_workers(), 0,
        STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    if (my_rank == 0) {
        env->data = receive_back_pencil(env->data, 1);
    }
    else {
        send_back_pencil(0, env->data, 1);
        env->data = NULL;
    }

    starneig_node_finalize();

    return 0;
}

const struct hook_data_converter local_blacs_converter = {
    .from = HOOK_DATA_FORMAT_PENCIL_LOCAL,
    .to = HOOK_DATA_FORMAT_PENCIL_BLACS,
    .print_usage = local_blacs_print_usage,
    .print_args = local_blacs_print_args,
    .check_args = local_blacs_check_args,
    .convert = local_blacs_convert
};

const struct hook_data_converter blacs_local_converter = {
    .from = HOOK_DATA_FORMAT_PENCIL_BLACS,
    .to = HOOK_DATA_FORMAT_PENCIL_LOCAL,
    .convert = blacs_local_convert
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void blacs_starneig_print_usage(int argc, char * const *argv)
{
    init_helper_print_usage(
        "blacs-starneig-", INIT_HELPER_STARNEIG_PENCIL, argc, argv);
}

static void blacs_starneig_print_args(int argc, char * const *argv)
{
    init_helper_print_args(
        "blacs-starneig-", INIT_HELPER_STARNEIG_PENCIL, argc, argv);
}

static int blacs_starneig_check_args(int argc, char * const *argv, int *argr)
{
    return init_helper_check_args(
        "blacs-starneig-", INIT_HELPER_STARNEIG_PENCIL, argc, argv, argr);
}

static int blacs_starneig_convert(
    int argc, char * const *argv, struct hook_data_env *env)
{
    env->format = HOOK_DATA_FORMAT_PENCIL_STARNEIG;
    pencil_t pencil = env->data;

    init_helper_t helper = init_helper_init_hook(
        "blacs-starneig-", env->format,
        STARNEIG_MATRIX_M(pencil->mat_a), STARNEIG_MATRIX_N(pencil->mat_a),
        pencil->mat_a->dtype, argc, argv);

    starneig_node_init(threads_get_workers(), 0,
        STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

#define CONVERT(mat_x)
    if (pencil->mat_x != NULL) { \
        matrix_t tmp = init_matrix( \
            STARNEIG_MATRIX_M(pencil->mat_x), \
            STARNEIG_MATRIX_N(pencil->mat_x), \
            helper); \
        starneig_distr_matrix_copy( \
            STARNEIG_MATRIX_HANDLE(pencil->mat_x), \
            STARNEIG_MATRIX_HANDLE(tmp)); \
        free_matrix_descr(pencil->mat_x); \
        pencil->mat_x = tmp; \
    }

    CONVERT(mat_a);
    CONVERT(mat_b);
    CONVERT(mat_q);
    CONVERT(mat_z);
    CONVERT(mat_x);
    CONVERT(mat_ca);
    CONVERT(mat_cb);

#undef CONVERT

    starneig_node_finalize();
    init_helper_free(helper);

    return 0;
}

const struct hook_data_converter blacs_starneig_converter = {
    .from = HOOK_DATA_FORMAT_PENCIL_BLACS,
    .to = HOOK_DATA_FORMAT_PENCIL_STARNEIG,
    .print_usage = blacs_starneig_print_usage,
    .print_args = blacs_starneig_print_args,
    .check_args = blacs_starneig_check_args,
    .convert = blacs_starneig_convert
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void starneig_blacs_print_usage(int argc, char * const *argv)
{
    init_helper_print_usage(
        "starneig-blacs-", INIT_HELPER_BLACS_PENCIL, argc, argv);
}

static void starneig_blacs_print_args(int argc, char * const *argv)
{
    init_helper_print_args(
        "starneig-blacs-", INIT_HELPER_BLACS_PENCIL, argc, argv);
}

static int starneig_blacs_check_args(int argc, char * const *argv, int *argr)
{
    return init_helper_check_args(
        "starneig-blacs-", INIT_HELPER_BLACS_PENCIL, argc, argv, argr);
}

static int starneig_blacs_convert(
    int argc, char * const *argv, struct hook_data_env *env)
{
    env->format = HOOK_DATA_FORMAT_PENCIL_BLACS;
    pencil_t pencil = env->data;

    init_helper_t helper = init_helper_init_hook(
        "starneig-blacs-", env->format,
        STARNEIG_MATRIX_M(pencil->mat_a), STARNEIG_MATRIX_N(pencil->mat_a),
        pencil->mat_a->dtype, argc, argv);

    starneig_node_init(threads_get_workers(), 0,
        STARNEIG_HINT_DM | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

#define CONVERT(mat_x)
    if (pencil->mat_x != NULL) { \
        matrix_t tmp = init_matrix( \
            STARNEIG_MATRIX_M(pencil->mat_x), \
            STARNEIG_MATRIX_N(pencil->mat_x), \
            helper); \
        starneig_distr_matrix_copy( \
            STARNEIG_MATRIX_HANDLE(pencil->mat_x), \
            STARNEIG_MATRIX_HANDLE(tmp)); \
        free_matrix_descr(pencil->mat_x); \
        pencil->mat_x = tmp; \
    }

    CONVERT(mat_a);
    CONVERT(mat_b);
    CONVERT(mat_q);
    CONVERT(mat_z);
    CONVERT(mat_x);
    CONVERT(mat_ca);
    CONVERT(mat_cb);

#undef CONVERT

    starneig_node_finalize();
    init_helper_free(helper);

    return 0;
}

const struct hook_data_converter starneig_blacs_converter = {
    .from = HOOK_DATA_FORMAT_PENCIL_STARNEIG,
    .to = HOOK_DATA_FORMAT_PENCIL_BLACS,
    .print_usage = starneig_blacs_print_usage,
    .print_args = starneig_blacs_print_args,
    .check_args = starneig_blacs_check_args,
    .convert = starneig_blacs_convert
};
