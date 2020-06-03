///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
/// @author Lars Karlsson (larsk@cs.umu.se), Umeå University
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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/distr_matrix.h>
#include <starneig/distr_helpers.h>
#include "distr_matrix_internal.h"
#include "node_internal.h"
#include "utils.h"
#include "../common/common.h"
#include "../common/node_internal.h"
#include "../common/tasks.h"
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <mpi.h>
#include <starpu.h>
#include <starpu_mpi.h>

#define DESCR_CACHE_SIZE 100

struct descr_cache_entry {
    int in_use;
    starneig_matrix_descr_t descr;
};

static struct descr_cache_entry descr_cache[DESCR_CACHE_SIZE] = { 0 };

struct block_cyclic_arg {
    int rows;
    int cols;
};

static int block_cyclic_row_func(int i, int j, void *arg)
{
    struct block_cyclic_arg *mesh = arg;
    return (i % mesh->rows) * mesh->cols + j % mesh->cols;
}

static int block_cyclic_col_func(int i, int j, void *arg)
{
    struct block_cyclic_arg *mesh = arg;
    return i % mesh->rows + (j % mesh->cols) * mesh->rows;
}

static int single_owner_func(int i, int j, void *arg)
{
    return * (int *) arg;
}

starneig_matrix_descr_t starneig_mpi_cache_convert(
    int bm, int bn, enum starneig_matrix_type fill,
    starneig_distr_matrix_t matrix, mpi_info_t mpi)
{
    starneig_matrix_descr_t descr =
        starneig_mpi_cache_convert_and_release(bm, bn, fill, matrix, mpi);
    starneig_acquire_matrix_descr(descr);

    return descr;
}

starneig_matrix_descr_t starneig_mpi_cache_convert_and_release(
    int bm, int bn, enum starneig_matrix_type fill,
    starneig_distr_matrix_t matrix, mpi_info_t mpi)
{
    if (matrix->descr != DESCR_CACHE_EMPTY &&
    descr_cache[matrix->descr].descr != NULL) {
        struct descr_cache_entry *entry = &descr_cache[matrix->descr];
        int _bm = STARNEIG_MATRIX_BM(entry->descr);
        int _bn = STARNEIG_MATRIX_BN(entry->descr);
        if (bm == _bm && bn == _bn) {
            starneig_release_matrix_descr(entry->descr);
            return entry->descr;
        }
        starneig_mpi_cache_remove(matrix);
    }

    starneig_matrix_descr_t descr = starneig_init_matrix_descr(
        matrix->rows, matrix->cols, bm, bn,
        matrix->row_blksz / bm, matrix->col_blksz / bn,
        starneig_distr_matrix_get_elemsize(matrix),
        (int (*)(int, int, void const *)) matrix->distr->func,
        matrix->distr->arg, mpi);

    for (int i = 0; i < matrix->block_count; i++)
        starneig_register_section_with_matrix_descr(
            fill,
            matrix->blocks[i].glo_row / matrix->row_blksz,
            matrix->blocks[i].glo_col / matrix->col_blksz,
            matrix->blocks[i].ld, matrix->blocks[i].ptr, descr);

    if (matrix->descr != DESCR_CACHE_EMPTY) {
        descr_cache[matrix->descr].descr = descr;
        return descr;
    }
    else {
        for (int i = 0; i < DESCR_CACHE_SIZE; i++) {
            if (!descr_cache[i].in_use) {
                descr_cache[i].in_use = 1;
                descr_cache[i].descr = descr;
                matrix->descr = i;
                return descr;
            }
        }
    }

    starneig_fatal_error(
        "Maximum number of concurrent distributed matrices reached.");
    return NULL;
}

void starneig_mpi_cache_remove(starneig_distr_matrix_t matrix)
{
    if (matrix->descr == DESCR_CACHE_EMPTY)
        return;

    if (descr_cache[matrix->descr].descr != NULL) {
        starneig_release_matrix_descr(descr_cache[matrix->descr].descr);
        starneig_unregister_matrix_descr(descr_cache[matrix->descr].descr);
        starneig_free_matrix_descr(descr_cache[matrix->descr].descr);
    }
    descr_cache[matrix->descr].in_use = 0;
    descr_cache[matrix->descr].descr = NULL;

    matrix->descr = DESCR_CACHE_EMPTY;
}

void starneig_mpi_cache_clear()
{
    for (int i = 0; i < DESCR_CACHE_SIZE; i++) {
        if (descr_cache[i].in_use && descr_cache[i].descr != NULL) {
            starneig_release_matrix_descr(descr_cache[i].descr);
            starneig_unregister_matrix_descr(descr_cache[i].descr);
            starneig_free_matrix_descr(descr_cache[i].descr);
        }
        descr_cache[i].descr = NULL;
    }
}

__attribute__ ((visibility ("default")))
starneig_distr_t starneig_distr_init()
{
    return starneig_distr_init_mesh(-1, -1, STARNEIG_ORDER_DEFAULT);
}

__attribute__ ((visibility ("default")))
starneig_distr_t starneig_distr_init_mesh(
    int rows, int cols, starneig_distr_order_t order)
{
    struct starneig_distr *distr = malloc(sizeof(struct starneig_distr));

    if (rows == -1 || cols == -1) {
        int world_size;
        MPI_Comm_size(starneig_mpi_get_comm(), &world_size);

        rows = ceil(sqrt(world_size));
        while (world_size % rows != 0)
            rows++;

        cols = world_size/rows;
    }

    if (order == STARNEIG_ORDER_DEFAULT || STARNEIG_ORDER_ROW_MAJOR) {
        distr->type = STARNEIG_DISTR_TYPE_2DBC_ROW;
        distr->func = &block_cyclic_row_func;
    }
    else {
        distr->type = STARNEIG_DISTR_TYPE_2DBC_COL;
        distr->func = &block_cyclic_col_func;
    }
    distr->arg = malloc(sizeof(struct block_cyclic_arg));
    ((struct block_cyclic_arg *)distr->arg)->rows = rows;
    ((struct block_cyclic_arg *)distr->arg)->cols = cols;
    distr->arg_size = sizeof(struct block_cyclic_arg);
    distr->rows = rows;
    distr->cols = cols;

    return distr;
}

__attribute__ ((visibility ("default")))
starneig_distr_t starneig_distr_init_func(
    int (*func)(int row, int col, void *arg), void *arg, size_t arg_size)
{
    struct starneig_distr *distr = malloc(sizeof(struct starneig_distr));

    int world_size;
    MPI_Comm_size(starneig_mpi_get_comm(), &world_size);

    distr->type = STARNEIG_DISTR_TYPE_FUNC;

    distr->func = func;
    distr->arg = NULL;
    distr->arg_size = 0;
    if (arg != NULL) {
        distr->arg = malloc(arg_size);
        memcpy(distr->arg, arg, arg_size);
        distr->arg_size = arg_size;
    }

    distr->rows = 1;
    distr->cols = world_size;

    return distr;
}

__attribute__ ((visibility ("default")))
starneig_distr_t starneig_distr_duplicate(starneig_distr_t distr)
{
    struct starneig_distr *new = malloc(sizeof(struct starneig_distr));

    new->type = distr->type;

    new->func = distr->func;
    new->arg = NULL;
    new->arg_size = 0;
    if (distr->arg != NULL) {
        new->arg = malloc(distr->arg_size);
        memcpy(new->arg, distr->arg, distr->arg_size);
        new->arg_size = distr->arg_size;
    }

    new->rows = distr->rows;
    new->cols = distr->cols;

    return new;
}

__attribute__ ((visibility ("default")))
void starneig_distr_destroy(starneig_distr_t distr)
{
    if (distr == NULL)
        return;

    free(distr->arg);
    free(distr);
}

__attribute__ ((visibility ("default")))
starneig_distr_matrix_t starneig_distr_matrix_create(
    int rows, int cols, int row_blksz, int col_blksz, starneig_datatype_t type,
    starneig_distr_t distr)
{
    struct starneig_distr_matrix *matrix =
        malloc(sizeof(struct starneig_distr_matrix));

    int my_rank;
    MPI_Comm_rank(starneig_mpi_get_comm(), &my_rank);

    int world_size;
    MPI_Comm_size(starneig_mpi_get_comm(), &world_size);

    size_t elemsize = starneig_mpi_get_elemsize(type);

    if (distr == NULL)
        matrix->distr = distr = starneig_distr_init();
    else
        matrix->distr = starneig_distr_duplicate(distr);

    matrix->rows = rows;
    matrix->cols = cols;

    if (row_blksz < 0)
        row_blksz = MAX(1024,
            divceil(divceil(rows, 4*ceil(sqrt(world_size))), 120)*120);
    if (col_blksz < 0)
        col_blksz = MAX(1024,
            divceil(divceil(cols, 4*ceil(sqrt(world_size))), 120)*120);

    matrix->row_blksz = row_blksz;
    matrix->col_blksz = col_blksz;

    if (distr->type == STARNEIG_DISTR_TYPE_2DBC_ROW ||
    distr->type == STARNEIG_DISTR_TYPE_2DBC_COL) {

        int my_row_rank, my_col_rank;
        if (distr->type == STARNEIG_DISTR_TYPE_2DBC_ROW) {
            my_row_rank = my_rank / distr->cols;
            my_col_rank = my_rank % distr->cols;
        }
        else {
            my_row_rank = my_rank / distr->rows;
            my_col_rank = my_rank % distr->rows;
        }

        // calculate how many block rows and block columns are owner locally
        int local_block_rows =
            divceil(rows - my_row_rank*row_blksz, distr->rows*row_blksz);
        int local_block_cols =
            divceil(cols - my_col_rank*col_blksz, distr->cols*col_blksz);

        matrix->block_count = local_block_rows*local_block_cols;
        matrix->blocks = malloc(
            matrix->block_count*sizeof(struct starneig_distr_block));

        starneig_verbose("Attemting to allocate %.0f MB for a local buffer.",
            1.0E-6 * local_block_rows*row_blksz *
            local_block_cols*col_blksz * elemsize);

        matrix->ptr = starneig_alloc_pinned_matrix(
            local_block_rows*row_blksz,
            local_block_cols*col_blksz,
            elemsize, &matrix->ld);

        starneig_verbose("Allocated %.0f MB for a local buffer.",
            1.0E-6 * matrix->ld*local_block_cols*col_blksz*elemsize);

        for (int j = 0; j < local_block_cols; j++) {
            for (int i = 0; i < local_block_rows; i++) {
                struct starneig_distr_block *block =
                    &matrix->blocks[j*local_block_rows+i];

                block->row_blksz = MIN(row_blksz,
                    rows - (i*distr->rows+my_row_rank)*row_blksz);
                block->col_blksz = MIN(col_blksz,
                    cols - (j*distr->cols+my_col_rank)*col_blksz);
                block->glo_row = (i*distr->rows+my_row_rank)*row_blksz;
                block->glo_col = (j*distr->cols+my_col_rank)*col_blksz;
                block->ld = matrix->ld;
                block->ptr = (double *)matrix->ptr +
                    j*col_blksz*matrix->ld + i*row_blksz;
            }
        }
    }

    if (distr->type == STARNEIG_DISTR_TYPE_FUNC) {
        // calculate the total number of block rows and block columns
        int block_rows = divceil(rows, row_blksz);
        int block_cols = divceil(cols, col_blksz);

        // calculate how many blocks are owner locally
        int block_count = 0;
        for (int i = 0; i < block_rows; i++)
            for (int j = 0; j < block_cols; j++)
                if (distr->func(i, j, distr->arg) == my_rank)
                    block_count++;

        matrix->block_count = block_count;
        matrix->blocks =
            malloc(block_count*sizeof(struct starneig_distr_block));

        matrix->ptr = starneig_alloc_pinned_matrix(
            row_blksz, block_count*col_blksz, elemsize, &matrix->ld);

        int k = 0;
        for (int i = 0; i < block_rows; i++) {
            for (int j = 0; j < block_cols; j++) {
                if (distr->func(i, j, distr->arg) == my_rank) {
                    matrix->blocks[k].row_blksz =
                        MIN(row_blksz, rows - i*row_blksz);
                    matrix->blocks[k].col_blksz =
                        MIN(col_blksz, cols - j*col_blksz);
                    matrix->blocks[k].glo_row = i*row_blksz;
                    matrix->blocks[k].glo_col = j*col_blksz;
                    matrix->blocks[k].ld = matrix->ld;
                    matrix->blocks[k].ptr =
                        (double *) matrix->ptr + k*col_blksz*matrix->ld;
                    k++;
                }
            }
        }
    }

    matrix->free_ptr = 1;
    matrix->datatype = type;
    matrix->descr = DESCR_CACHE_EMPTY;

    return matrix;
}

__attribute__ ((visibility ("default")))
starneig_distr_matrix_t starneig_distr_matrix_create_local(
    int rows, int cols, starneig_datatype_t type, int owner, double *A,
    int ldA)
{
    struct starneig_distr_matrix *matrix =
        malloc(sizeof(struct starneig_distr_matrix));

    matrix->distr =
        starneig_distr_init_func(&single_owner_func, &owner, sizeof(owner));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->row_blksz = divceil(rows, 120)*120;
    matrix->col_blksz = divceil(cols, 120)*120;
    matrix->block_count = 1;
    matrix->blocks = malloc(sizeof(struct starneig_distr_block));
    matrix->blocks->row_blksz = rows;
    matrix->blocks->col_blksz = cols;
    matrix->blocks->glo_row = 0;
    matrix->blocks->glo_col = 0;
    matrix->blocks->ptr = A;
    matrix->blocks->ld = ldA;
    matrix->ptr = A;
    matrix->ld = ldA;
    matrix->free_ptr = 0;
    matrix->datatype = type;
    matrix->descr = DESCR_CACHE_EMPTY;

    return matrix;
}

__attribute__ ((visibility ("default")))
void starneig_distr_matrix_destroy(starneig_distr_matrix_t matrix)
{
    if (matrix == NULL)
        return;

    starneig_mpi_cache_remove(matrix);

    starneig_distr_destroy(matrix->distr);
    free(matrix->blocks);
    if (matrix->free_ptr)
        starneig_free_pinned_matrix(matrix->ptr);
    free(matrix);
}

__attribute__ ((visibility ("default")))
void starneig_distr_matrix_copy(
    starneig_distr_matrix_t source, starneig_distr_matrix_t dest)
{
    CHECK_INIT();
    starneig_distr_matrix_copy_region(0, 0, 0, 0,
        starneig_distr_matrix_get_rows(dest),
        starneig_distr_matrix_get_cols(dest),
        source, dest);
}

__attribute__ ((visibility ("default")))
void starneig_distr_matrix_copy_region(
    int sr, int sc, int dr, int dc, int rows, int cols,
    starneig_distr_matrix_t source, starneig_distr_matrix_t dest)
{
    CHECK_INIT();

    int tile_size = MAX(128, ((dest->rows/100)/8)*8);
    int dest_tile_size =
        starneig_mpi_find_valid_tile_size(tile_size, dest, NULL, NULL, NULL);
    int source_tile_size =
        starneig_mpi_find_valid_tile_size(tile_size, source, NULL, NULL, NULL);

    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    mpi_info_t mpi = starneig_mpi_get_info();

    starneig_matrix_descr_t dest_descr =
        starneig_mpi_cache_convert_and_release(
            dest_tile_size, dest_tile_size, MATRIX_TYPE_FULL,
            dest, mpi);
    starneig_matrix_descr_t source_descr =
        starneig_mpi_cache_convert_and_release(
            source_tile_size, source_tile_size, MATRIX_TYPE_FULL,
            source, mpi);

    // A machine may run out of memory when copying a large matrix. This should
    // keep things under control.
    int splice = divceil(8192, dest->col_blksz)*dest->col_blksz;
    for (int i = dc/splice; i < dc+cols; i += splice) {
        int begin = MAX(dc, i);
        int end = MIN(dc+cols, i+splice);

        starneig_insert_copy_matrix(
            sr, begin + sc - dc, dr, begin, rows, end-begin,
            STARPU_MAX_PRIO, source_descr, dest_descr, mpi);

        starpu_task_wait_for_all();
        starpu_mpi_barrier(starneig_mpi_get_comm());
    }

    starneig_acquire_matrix_descr(dest_descr);
    starneig_acquire_matrix_descr(source_descr);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
}

__attribute__ ((visibility ("default")))
void starneig_distr_matrix_get_blocks(
    starneig_distr_matrix_t matrix, struct starneig_distr_block **blocks,
    int *block_count)
{
    *blocks = matrix->blocks;
    *block_count = matrix->block_count;
}

__attribute__ ((visibility ("default")))
starneig_distr_t starneig_distr_matrix_get_distr(starneig_distr_matrix_t matrix)
{
    return matrix->distr;
}

__attribute__ ((visibility ("default")))
starneig_datatype_t starneig_distr_matrix_get_datatype(
    starneig_distr_matrix_t matrix)
{
    return matrix->datatype;
}

__attribute__ ((visibility ("default")))
size_t starneig_distr_matrix_get_elemsize(starneig_distr_matrix_t matrix)
{
    return starneig_mpi_get_elemsize(matrix->datatype);
}

__attribute__ ((visibility ("default")))
int starneig_distr_matrix_get_rows(starneig_distr_matrix_t matrix)
{
    return matrix->rows;
}

__attribute__ ((visibility ("default")))
int starneig_distr_matrix_get_cols(starneig_distr_matrix_t matrix)
{
    return matrix->cols;
}

__attribute__ ((visibility ("default")))
int starneig_distr_matrix_get_row_blksz(starneig_distr_matrix_t matrix)
{
    return matrix->row_blksz;
}

__attribute__ ((visibility ("default")))
int starneig_distr_matrix_get_col_blksz(starneig_distr_matrix_t matrix)
{
    return matrix->col_blksz;
}
