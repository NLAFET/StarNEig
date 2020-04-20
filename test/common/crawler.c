///
/// @file This file contains a matrix crawler object that is used to initialize
/// an validate the input and output matrices
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
#include "crawler.h"
#include "common.h"
#include "local_pencil.h"
#include "threads.h"
#include <stdlib.h>
#include <stdio.h>
#include <starneig/node.h>
#ifdef STARNEIG_ENABLE_MPI
#include "starneig_pencil.h"
#include <starneig/distr_helpers.h>
#include <mpi.h>
#endif

#define MAX_MATRICES 10
#define BUFFER_SIZE 1000000000

static int extract_local(int max_matrices, va_list vl, matrix_t *matrices)
{
    int matrix_count = 0;

    matrix_t val = va_arg(vl, matrix_t);
    while (val != NULL) {
        if (matrix_count < max_matrices) {
            matrices[matrix_count++] = val;
            val = va_arg(vl, matrix_t);
        }
        else {
            fprintf(stderr,
                "The matrix crawler reached the maximum number of matrices "
                "to crawl.\n");
            abort();
        }
    }

    return matrix_count;
}

static void crawl_panel_local(
    crawler_access_t access, crawler_func_t func, void *arg, va_list vl)
{
    matrix_t matrices[MAX_MATRICES];
    int matrix_count = extract_local(
        sizeof(matrices)/sizeof(matrices[0]), vl, matrices);

    if (matrix_count == 0)
        return;

    size_t lds[MAX_MATRICES];
    double *ptrs[MAX_MATRICES];

    for (int i = 0; i < matrix_count; i++) {
        lds[i] = LOCAL_MATRIX_LD(matrices[i]);
        ptrs[i] = LOCAL_MATRIX_PTR(matrices[i]);
    }

    func(0, LOCAL_MATRIX_N(matrices[0]), LOCAL_MATRIX_M(matrices[0]),
        LOCAL_MATRIX_N(matrices[0]), matrix_count, lds, (void **) ptrs, arg);

    init_prand(prand());
}

static void crawl_hpanel_local(
    crawler_access_t access, crawler_func_t func, void *arg, va_list vl)
{
    matrix_t matrices[MAX_MATRICES];
    int matrix_count = extract_local(
        sizeof(matrices)/sizeof(matrices[0]), vl, matrices);

    if (matrix_count == 0)
        return;

    size_t lds[MAX_MATRICES];
    double *ptrs[MAX_MATRICES];

    for (int i = 0; i < matrix_count; i++) {
        lds[i] = LOCAL_MATRIX_LD(matrices[i]);
        ptrs[i] = LOCAL_MATRIX_PTR(matrices[i]);
    }

    func(0, LOCAL_MATRIX_M(matrices[0]), LOCAL_MATRIX_M(matrices[0]),
        LOCAL_MATRIX_N(matrices[0]), matrix_count, lds, (void **) ptrs, arg);

    init_prand(prand());
}

static void crawl_diag_window_local(
    crawler_access_t access, crawler_func_t func, void *arg,
    va_list vl)
{
    matrix_t matrices[MAX_MATRICES];
    int matrix_count = extract_local(
        sizeof(matrices)/sizeof(matrices[0]), vl, matrices);

    if (matrix_count == 0)
        return;

    size_t lds[MAX_MATRICES];
    double *ptrs[MAX_MATRICES];

    for (int i = 0; i < matrix_count; i++) {
        lds[i] = LOCAL_MATRIX_LD(matrices[i]);
        ptrs[i] = LOCAL_MATRIX_PTR(matrices[i]);
    }

    func(0, MIN(LOCAL_MATRIX_M(matrices[0]), LOCAL_MATRIX_N(matrices[0])),
        LOCAL_MATRIX_M(matrices[0]), LOCAL_MATRIX_N(matrices[0]), matrix_count,
        lds, (void **) ptrs, arg);

    init_prand(prand());
}

#ifdef STARNEIG_ENABLE_MPI

static int extract_starneig(
    int max_matrices, va_list vl, starneig_distr_matrix_t *matrices)
{
    int matrix_count = 0;

    matrix_t val = va_arg(vl, matrix_t);
    while (val != NULL) {
        if (matrix_count < max_matrices) {
            matrices[matrix_count++] = val->ptr;
            val = va_arg(vl, matrix_t);
        }
        else {
            fprintf(stderr,
                "The matrix crawler reached the maximum number of matrices "
                "to crawl.\n");
            abort();
        }
    }

    return matrix_count;
}

static void crawl_panel_starneig(
    crawler_access_t access, crawler_func_t func, void *arg, va_list vl)
{
    starneig_distr_matrix_t matrices[MAX_MATRICES];
    int matrix_count = extract_starneig(
        sizeof(matrices)/sizeof(matrices[0]), vl, matrices);

    if (matrix_count == 0)
        return;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int initialized = starneig_node_initialized();
    if (!initialized)
        starneig_node_init(MAX(1, threads_get_workers()/2), 0,
            threads_get_fast_dm() | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    int m = starneig_distr_matrix_get_rows(matrices[0]);
    int n = starneig_distr_matrix_get_cols(matrices[0]);
    size_t elemsize = starneig_distr_matrix_get_elemsize(matrices[0]);
    int panel_width = MAX(128, MIN(n, ((BUFFER_SIZE/(m*elemsize))/128)*128));

    //
    // allocate local buffers
    //

    size_t lds[MAX_MATRICES] = { 0 };
    double *ptrs[MAX_MATRICES] = { 0 };
    starneig_distr_matrix_t handles[MAX_MATRICES] = { 0 };
    for (int i = 0; i < matrix_count; i++) {
        if (my_rank == 0)
            ptrs[i] = alloc_matrix(m, panel_width, elemsize, &lds[i]);
        // TODO: Add support for other data types
        handles[i] = starneig_distr_matrix_create_local(
            m, panel_width, STARNEIG_REAL_DOUBLE, 0, ptrs[i], lds[i]);
    }

    int offset = 0;
    while (offset < n) {
        int width = MIN(panel_width, n-offset);

        //
        // gather
        //

        if (access == CRAWLER_R || access == CRAWLER_RW) {
            for (int j = 0; j < matrix_count; j++)
                starneig_distr_matrix_copy_region(
                    0, offset, 0, 0, m, width, matrices[j], handles[j]);
        }

        //
        // process
        //

        int progress;
        if (my_rank == 0)
            progress = func(
                offset, width, m, n, matrix_count, lds, (void **) ptrs, arg);
        starneig_mpi_broadcast(0, sizeof(progress), &progress);

        if (progress == -1)
            goto cleanup;

        //
        // scatter
        //

        if (access == CRAWLER_W || access == CRAWLER_RW) {
            for (int j = 0; j < matrix_count; j++)
                starneig_distr_matrix_copy_region(
                    0, 0, 0, offset, m, progress, handles[j], matrices[j]);
        }

        offset += progress;
    }

cleanup:

    for (int i = 0; i < matrix_count; i++) {
        starneig_distr_matrix_destroy(handles[i]);
        free_matrix(ptrs[i]);
    }

    unsigned seed = prand();
    starneig_mpi_broadcast(0, sizeof(seed), &seed);
    init_prand(seed);

    if (!initialized)
        starneig_node_finalize();
}

static void crawl_hpanel_starneig(
    crawler_access_t access, crawler_func_t func, void *arg, va_list vl)
{
    starneig_distr_matrix_t matrices[MAX_MATRICES];
    int matrix_count = extract_starneig(
        sizeof(matrices)/sizeof(matrices[0]), vl, matrices);

    if (matrix_count == 0)
        return;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int initialized = starneig_node_initialized();
    if (!initialized)
        starneig_node_init(MAX(1, threads_get_workers()/2), 0,
            threads_get_fast_dm() | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    int m = starneig_distr_matrix_get_rows(matrices[0]);
    int n = starneig_distr_matrix_get_cols(matrices[0]);
    size_t elemsize = starneig_distr_matrix_get_elemsize(matrices[0]);
    int panel_height = MAX(128, MIN(m, ((BUFFER_SIZE/(n*elemsize))/128)*128));

    //
    // allocate local buffers
    //

    size_t lds[MAX_MATRICES] = { 0 };
    double *ptrs[MAX_MATRICES] = { 0 };
    starneig_distr_matrix_t handles[MAX_MATRICES] = { 0 };
    for (int i = 0; i < matrix_count; i++) {
        if (my_rank == 0)
            ptrs[i] = alloc_matrix(panel_height, n, elemsize, &lds[i]);
        // TODO: Add support for other data types
        handles[i] = starneig_distr_matrix_create_local(
            panel_height, n, STARNEIG_REAL_DOUBLE, 0, ptrs[i], lds[i]);
    }

    int offset = 0;
    while (offset < n) {
        int height = MIN(panel_height, m-offset);

        //
        // gather
        //

        if (access == CRAWLER_R || access == CRAWLER_RW) {
            for (int j = 0; j < matrix_count; j++)
                starneig_distr_matrix_copy_region(
                    offset, 0, 0, 0, height, n, matrices[j], handles[j]);
        }

        //
        // process
        //

        int progress;
        if (my_rank == 0)
            progress = func(
                offset, height, m, n, matrix_count, lds, (void **) ptrs, arg);
        starneig_mpi_broadcast(0, sizeof(progress), &progress);

        if (progress == -1)
            goto cleanup;

        //
        // scatter
        //

        if (access == CRAWLER_W || access == CRAWLER_RW) {
            for (int j = 0; j < matrix_count; j++)
                starneig_distr_matrix_copy_region(
                    0, 0, offset, 0, progress, n, handles[j], matrices[j]);
        }

        offset += progress;
    }

cleanup:

    for (int i = 0; i < matrix_count; i++) {
        starneig_distr_matrix_destroy(handles[i]);
        free_matrix(ptrs[i]);
    }

    unsigned seed = prand();
    starneig_mpi_broadcast(0, sizeof(seed), &seed);
    init_prand(seed);

    if (!initialized)
        starneig_node_finalize();
}

static void crawl_diag_window_starneig(
    crawler_access_t access, crawler_func_t func, void *arg,
    va_list vl)
{
    starneig_distr_matrix_t matrices[MAX_MATRICES];
    int matrix_count = extract_starneig(
        sizeof(matrices)/sizeof(matrices[0]), vl, matrices);

    if (matrix_count == 0)
        return;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int initialized = starneig_node_initialized();
    if (!initialized)
        starneig_node_init(MAX(1, threads_get_workers()/2), 0,
            threads_get_fast_dm() | STARNEIG_NO_VERBOSE | STARNEIG_FXT_DISABLE);

    int m = starneig_distr_matrix_get_rows(matrices[0]);
    int n = starneig_distr_matrix_get_cols(matrices[0]);
    size_t elemsize = starneig_distr_matrix_get_elemsize(matrices[0]);
    int size = MIN(MIN(m, n), 256);

    //
    // allocate local buffers
    //

    size_t lds[MAX_MATRICES] = { 0 };
    double *ptrs[MAX_MATRICES] = { 0 };
    starneig_distr_matrix_t handles[MAX_MATRICES] = { 0 };
    for (int i = 0; i < matrix_count; i++) {
        if (my_rank == 0)
            ptrs[i] = alloc_matrix(size, size, elemsize, &lds[i]);
        // TODO: Add support for other data types
        handles[i] = starneig_distr_matrix_create_local(
            size, size, STARNEIG_REAL_DOUBLE, 0, ptrs[i], lds[i]);
    }

    int offset = 0;
    while (offset < MIN(m, n)) {
        int _size = MIN(size, MIN(m, n)-offset);

        //
        // gather
        //

        if (access == CRAWLER_R || access == CRAWLER_RW) {
            for (int j = 0; j < matrix_count; j++)
                starneig_distr_matrix_copy_region(
                    offset, offset, 0, 0, _size, _size, matrices[j],
                    handles[j]);
        }

        //
        // process
        //

        int progress;
        if (my_rank == 0)
            progress = func(
                offset, _size, m, n, matrix_count, lds, (void **) ptrs, arg);
        starneig_mpi_broadcast(0, sizeof(progress), &progress);

        if (progress == -1)
            goto cleanup;

        //
        // scatter
        //

        if (access == CRAWLER_W || access == CRAWLER_RW) {
            for (int j = 0; j < matrix_count; j++)
                starneig_distr_matrix_copy_region(
                    0, 0, offset, offset, progress, progress, handles[j],
                    matrices[j]);
        }

        offset += progress;
    }

cleanup:

    for (int i = 0; i < matrix_count; i++) {
        starneig_distr_matrix_destroy(handles[i]);
        free_matrix(ptrs[i]);
    }

    unsigned seed = prand();
    starneig_mpi_broadcast(0, sizeof(seed), &seed);
    init_prand(seed);

    if (!initialized)
        starneig_node_finalize();
}

#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void crawl_matrices(
    crawler_access_t access, crawler_mode_t mode,
    crawler_func_t func, void *arg, size_t arg_size, ...)
{
    matrix_type_t type;
    {
        va_list vl;
        va_start(vl, arg_size);
        matrix_t val = va_arg(vl, matrix_t);
        type = val->type;
        va_end(vl);
    }

    va_list vl;
    va_start(vl, arg_size);

    if (mode == CRAWLER_PANEL) {
        switch (type) {
            case LOCAL_MATRIX:
                crawl_panel_local(access, func, arg, vl);
                break;
#ifdef STARNEIG_ENABLE_MPI
            case STARNEIG_MATRIX:
            case BLACS_MATRIX:
                crawl_panel_starneig(access, func, arg, vl);
                break;
#endif
            default:
                fprintf(stderr,
                    "The matrix crawler encountered an invalid matrix type.\n");
                abort();
        }
    }
    else if (mode == CRAWLER_HPANEL) {
        switch (type) {
            case LOCAL_MATRIX:
                crawl_hpanel_local(access, func, arg, vl);
                break;
#ifdef STARNEIG_ENABLE_MPI
            case STARNEIG_MATRIX:
            case BLACS_MATRIX:
                crawl_hpanel_starneig(access, func, arg, vl);
                break;
#endif
            default:
                fprintf(stderr,
                    "The matrix crawler encountered an invalid matrix type.\n");
                abort();
        }
    }
    else if (mode == CRAWLER_DIAG_WINDOW) {
        switch (type) {
            case LOCAL_MATRIX:
                crawl_diag_window_local(access, func, arg, vl);
                break;
#ifdef STARNEIG_ENABLE_MPI
            case STARNEIG_MATRIX:
            case BLACS_MATRIX:
                crawl_diag_window_starneig(access, func, arg, vl);
                break;
#endif
            default:
                fprintf(stderr,
                    "The matrix crawler encountered an invalid matrix type.\n");
                abort();
        }
    }
    else {
        fprintf(stderr,
            "The matrix crawler encountered an invalid crawling mode.\n");
        abort();
    }

    va_end(vl);

#ifdef STARNEIG_ENABLE_MPI
    if (0 < arg_size && (type == STARNEIG_MATRIX || type == BLACS_MATRIX)) {
        MPI_Bcast(arg, arg_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
#endif
}
