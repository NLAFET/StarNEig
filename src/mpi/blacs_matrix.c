///
/// @file
///
/// @brief This file contains data types and functions for BLACS formatted
/// distributed matrices.
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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/blacs_matrix.h>
#include <starneig/blacs_helpers.h>
#include "distr_matrix_internal.h"
#include "../common/node_internal.h"
#include <math.h>

__attribute__ ((visibility ("default")))
starneig_blacs_context_t starneig_distr_to_blacs_context(starneig_distr_t distr)
{
    if (distr->type != STARNEIG_DISTR_TYPE_2DBC_ROW &&
    distr->type != STARNEIG_DISTR_TYPE_2DBC_COL)
        starneig_fatal_error(
            "Only two-dimensional block cyclic distributions can be "
            "converted to a BLACS format.");

    starneig_blacs_context_t default_context =
        starneig_blacs_get(0, STARNEIG_BLACS_GET_DEFAULT_CONTEXT);

    if (distr->type == STARNEIG_DISTR_TYPE_2DBC_ROW)
        return starneig_blacs_gridinit(
            default_context, "Row-major", distr->rows, distr->cols);
    else
        return starneig_blacs_gridinit(
            default_context, "Column-major", distr->rows, distr->cols);
}

__attribute__ ((visibility ("default")))
starneig_distr_t starneig_blacs_context_to_distr(
    starneig_blacs_context_t context)
{
    int rows, cols, row, col;
    starneig_blacs_gridinfo(context, &rows, &cols, &row, &col);

    int prow, pcol;
    starneig_blacs_pcoord(context, 1, &prow, &pcol);
    if (0 < pcol)
        return starneig_distr_init_mesh(rows, cols, STARNEIG_ORDER_COL_MAJOR);
    else
        return starneig_distr_init_mesh(rows, cols, STARNEIG_ORDER_ROW_MAJOR);
}

__attribute__ ((visibility ("default")))
void starneig_blacs_create_matrix(
    int rows, int cols, int row_blksz, int col_blksz, starneig_datatype_t type,
    starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local)
{
    int my_rank, world_size;
    starneig_blacs_pinfo(&my_rank, &world_size);

    int grid_rows, grid_cols, row_rank, col_rank;
    starneig_blacs_gridinfo(
        context, &grid_rows, &grid_cols, &row_rank, &col_rank);

    if (row_blksz < 0)
        row_blksz = MAX(64, ((int)(rows / (2*sqrt(world_size))) / 120)*120);
    if (col_blksz < 0)
        col_blksz = MAX(64, ((int)(cols / (2*sqrt(world_size))) / 120)*120);

    // calculate how many rows and columns are owner locally
    int local_rows = starneig_blacs_numroc(rows, row_blksz, row_rank, 0, grid_rows);
    int local_cols = starneig_blacs_numroc(cols, col_blksz, col_rank, 0, grid_cols);

    size_t ld;
    *local = starneig_alloc_pinned_matrix(
        local_rows, local_cols, sizeof(double), &ld);

    starneig_blacs_descinit(
        descr, rows, cols, row_blksz, col_blksz, 0, 0, context, ld);
}

__attribute__ ((visibility ("default")))
void starneig_blacs_destroy_matrix(starneig_blacs_descr_t *descr, void **local)
{
    starneig_free_pinned_matrix(*local);
}

__attribute__ ((visibility ("default")))
void starneig_distr_matrix_to_blacs_descr(
    starneig_distr_matrix_t matrix, starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local)
{
    if (!starneig_distr_matrix_is_compatible_with(matrix, context))
        starneig_fatal_error(
            "The data distribution and the BLASC context are incompatible.");

    starneig_blacs_descinit(
        descr, matrix->rows, matrix->cols, matrix->row_blksz, matrix->col_blksz,
        0, 0, context, matrix->ld);

    *local = matrix->ptr;
}

__attribute__ ((visibility ("default")))
starneig_distr_matrix_t starneig_blacs_descr_to_distr_matrix(
    starneig_datatype_t type, starneig_distr_t distr,
    starneig_blacs_descr_t *descr, void *local)
{
    if (!starneig_distr_is_compatible_with(distr, descr->context))
        starneig_fatal_error(
            "The data distribution and the BLASC context are incompatible.");

    int my_rank, world_size;
    starneig_blacs_pinfo(&my_rank, &world_size);

    int grid_rows, grid_cols, row_rank, col_rank;
    starneig_blacs_gridinfo(
        descr->context, &grid_rows, &grid_cols, &row_rank, &col_rank);

    struct starneig_distr_matrix *matrix =
        malloc(sizeof(struct starneig_distr_matrix));

    matrix->distr = starneig_distr_duplicate(distr);

    int rows = matrix->rows = descr->m;
    int cols = matrix->cols = descr->n;
    int row_blksz = matrix->row_blksz = descr->sm;
    int col_blksz = matrix->col_blksz = descr->sn;

    // calculate how many rows and columns are owner locally
    int local_rows = starneig_blacs_numroc(rows, row_blksz, row_rank, 0, grid_rows);
    int local_cols = starneig_blacs_numroc(cols, col_blksz, col_rank, 0, grid_cols);

    int local_block_rows = divceil(local_rows, row_blksz);
    int local_block_cols = divceil(local_cols, col_blksz);

    matrix->block_count = local_block_rows*local_block_cols;
    matrix->blocks = malloc(
        matrix->block_count*sizeof(struct starneig_distr_block));

    matrix->ptr = local;
    matrix->ld = descr->lld;

    for (int j = 0; j < local_block_cols; j++) {
        for (int i = 0; i < local_block_rows; i++) {
            struct starneig_distr_block *block =
                &matrix->blocks[j*local_block_rows+i];

            block->row_blksz = MIN(row_blksz,
                rows - (i*grid_rows+row_rank)*row_blksz);
            block->col_blksz = MIN(col_blksz,
                cols - (j*grid_cols+col_rank)*col_blksz);
            block->glo_row = (i*grid_rows+row_rank)*row_blksz;
            block->glo_col = (j*grid_cols+col_rank)*col_blksz;
            block->ld = matrix->ld;
            block->ptr = (double *)matrix->ptr +
                j*col_blksz*matrix->ld + i*row_blksz;
        }
    }

    matrix->free_ptr = 0;
    matrix->datatype = type;

    return matrix;
}

__attribute__ ((visibility ("default")))
int starneig_distr_is_blacs_compatible(starneig_distr_t distr)
{
    return distr->type == STARNEIG_DISTR_TYPE_2DBC_ROW ||
        distr->type == STARNEIG_DISTR_TYPE_2DBC_COL;
}

__attribute__ ((visibility ("default")))
int starneig_distr_is_compatible_with(
    starneig_distr_t distr, starneig_blacs_context_t context)
{
    int grid_rows, grid_cols, row_rank, col_rank;
    starneig_blacs_gridinfo(
        context, &grid_rows, &grid_cols, &row_rank, &col_rank);

    if ((
        distr->type != STARNEIG_DISTR_TYPE_2DBC_ROW &&
        distr->type != STARNEIG_DISTR_TYPE_2DBC_COL) ||
    distr->rows != grid_rows || distr->cols != grid_cols)
        return 0;

    return 1;
}

__attribute__ ((visibility ("default")))
int starneig_distr_matrix_is_blacs_compatible(starneig_distr_matrix_t matrix)
{
    starneig_distr_t distr = starneig_distr_matrix_get_distr(matrix);
    return starneig_distr_is_blacs_compatible(distr);
}

__attribute__ ((visibility ("default")))
int starneig_distr_matrix_is_compatible_with(
    starneig_distr_matrix_t matrix, starneig_blacs_context_t context)
{
    starneig_distr_t distr = starneig_distr_matrix_get_distr(matrix);
    return starneig_distr_is_compatible_with(distr, context);
}

// deprecated
__attribute__ ((visibility ("default")))
void starneig_create_blacs_matrix(
    int rows, int cols, int row_blksz, int col_blksz, starneig_datatype_t type,
    starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local)
{
    starneig_warning("starneig_create_blacs_matrix has been deprecated.");
    starneig_blacs_create_matrix(
        rows, cols, row_blksz, col_blksz, type, context, descr, local);
}

// deprecated
__attribute__ ((visibility ("default")))
void starneig_destroy_blacs_matrix(
    starneig_blacs_descr_t *descr, void **local)
{
    starneig_warning("starneig_destroy_blacs_matrix has been deprecated.");
    starneig_blacs_destroy_matrix(descr, local);
}
