///
/// @file
///
/// @brief This file contains data types and functions for distributed matrices.
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

#ifndef STARNEIG_DISTR_MATRIX_H
#define STARNEIG_DISTR_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

#ifndef STARNEIG_ENABLE_MPI
#error "This header should be included only when STARNEIG_ENABLE_MPI is defined."
#endif

#include <stddef.h>

///
/// @defgroup starneig_dm_matrix Distributed Memory / Distributed matrices
///
/// @brief Data types and functions for distributed matrices.
///
/// @{
///

///
/// @name Data distributions
/// @{
///

struct starneig_distr;

///
/// @brief Data distribution.
///
typedef struct starneig_distr * starneig_distr_t;

///
/// @brief Process mapping order.
///
typedef enum {
    STARNEIG_ORDER_DEFAULT,     ///< Default ordering.
    STARNEIG_ORDER_ROW_MAJOR,   ///< Row-major natural ordering.
    STARNEIG_ORDER_COL_MAJOR    ///< Column-major natural ordering.
} starneig_distr_order_t;

///
/// @brief Creates a default data distribution.
///
/// @return A new data distribution.
///
starneig_distr_t starneig_distr_init();

///
/// @brief Creates a two-dimensional block cyclic data distribution.
///
/// @param[in] rows
///         The number of rows in the mesh. Can be set to -1 in
///         which case the library decides the value.
///
/// @param[in] cols
///         The number of columns in the mesh. Can be set to -1 in
///         which case the library decides the value.
///
/// @param[in] order
///         The process mapping order.
///
/// @return A new data distribution.
///
starneig_distr_t starneig_distr_init_mesh(
    int rows, int cols, starneig_distr_order_t order);

///
/// @brief Creates a distribution using a data distribution function.
///
/// The distribution function maps each block to it's owner. The function takes
/// three arguments: block's row index, blocks's column index and an optional
/// user defined argument.
///
/// @code{.c}
/// struct block_cyclic_arg {
///     int rows;
///     int cols;
/// };
///
/// int block_cyclic_func(int i, int j, void *arg)
/// {
///     struct block_cyclic_arg *mesh = (struct block_cyclic_arg *) arg;
///     return (i % mesh->rows) * mesh->cols + j % mesh->cols;
/// }

/// void func(...)
/// {
///     ...
///
///     // create a custom two-dimensional block cyclic distribution with 4 rows
///     // and 6 columns in the mesh
///     struct block_cyclic_arg arg = { .rows = 4, .cols = 6 };
///     starneig_distr_t distr =
///         starneig_distr_init_func(&block_cyclic_func, &arg, sizeof(arg));
///
///     ...
/// }
/// @endcode
///
/// @param[in] func
///         The data distribution function.
///
/// @param[in] arg
///         An optional data distribution function argument.
///
/// @param[in] arg_size
///         The size of the optional data distribution function argument.
///
/// @return A new data distribution.
///
starneig_distr_t starneig_distr_init_func(
    int (*func)(int row, int col, void *arg), void *arg, size_t arg_size);

///
/// @brief Duplicates a data distribution.
///
/// @param[in] distr
///         The data distribution to be duplicated.
///
/// @return A duplicated data distribution.
///
starneig_distr_t starneig_distr_duplicate(starneig_distr_t distr);

///
/// @brief Destroys a data distribution.
///
/// @param[in,out] distr
///         The data distribution to be destroyed.
///
void starneig_distr_destroy(starneig_distr_t distr);

///
/// @}
///

///
/// @name Distributed matrices
/// @{
///

struct starneig_distr_matrix;

///
/// @brief Distributed matrix.
///
typedef struct starneig_distr_matrix * starneig_distr_matrix_t;

///
/// @brief Distributed matrix element data type.
///
typedef enum {
    STARNEIG_REAL_DOUBLE  ///< Double precision real numbers.
} starneig_datatype_t;

///
/// @brief Distributed block.
///
struct starneig_distr_block {
    int row_blksz;  ///< The number of rows in the block.
    int col_blksz;  ///< The number of columns in the block.
    int glo_row;    ///< The topmost global row that belong to the block.
    int glo_col;    ///< The leftmost global column that belong to the block.
    int ld;         ///< The leading dimension of the local array.
    void *ptr;      ///< A pointer to the local array.
};

///
/// @brief Creates a distributed matrix with uninitialized matrix elements.
///
/// @code{.c}
/// // create a m X n double-precision real matrix that is distributed in a
/// // two-dimensional block cyclic fashion in bm X bn blocks
/// starneig_distr_t distr = starneig_distr_init();
/// starneig_distr_matrix_t dA =
///     starneig_distr_matrix_create(m, n, bm, bn, STARNEIG_REAL_DOUBLE, distr);
/// @endcode
///
/// @attention StarNEig library is designed to use much larger distributed
/// blocks than ScaLAPACK. Selecting a too small distributed block size will be
/// detrimental to the performance.
///
/// @param[in] rows
///         The number of (global) rows in the matrix.
///
/// @param[in] cols
///         The number of (global) columns in the matrix.
///
/// @param[in] row_blksz
///         The number of rows in a distribution block. Can be set to -1 in
///         which case the library decides the value.
///
/// @param[in] col_blksz
///         The number of columns in a distribution block. Can be set to -1 in
///         which case the library decides the value.
///
/// @param[in] type
///         The matrix element data type.
///
/// @param[in] distr
///         The data distribution. Can be left to NULL in which case the library
///         decides the distribution.
///
/// @return A new distributed matrix.
///
starneig_distr_matrix_t starneig_distr_matrix_create(
    int rows, int cols, int row_blksz, int col_blksz, starneig_datatype_t type,
    starneig_distr_t distr);

///
/// @brief Creates a single-owner distributed matrix from a local matrix.
///
///  This creates a wrapper. The contents of the local matrix may be
///  modified by the functions that use the wrapper. The
///  starneig_distr_matrix_destroy() function does not free the local matrix.
///
/// @code{.c}
/// int m = 1000, n = 1000;
/// double *A = NULL; size_t ldA = 0;
///
/// // rank 3 initialized the local matrix
/// if (my_rank = 3) {
///     A = initialize_matrix(m, n, &ldA);
/// }
///
/// // all ranks initialize the distributed matrix
/// starneig_distr_matrix_t lA = starneig_distr_matrix_create_local(
///     m, n, STARNEIG_REAL_DOUBLE, 3, A, ldA);
/// @endcode
///
/// @param[in] rows
///         The number of rows in the matrix.
///
/// @param[in] cols
///         The number of columns in the matrix.
///
/// @param[in] type
///         Matrix element data type.
///
/// @param[in] owner
///         MPI rank that owns the distributed matrix.
///
/// @param[in] A
///         A pointer to the local matrix. This argument is ignored the
///         calling rank is not the same as the owner.
///
/// @param[in] ldA
///         The leading dimension of the local matrix. This argument is ignored
///         the calling rank is not the same as the owner.
///
/// @return A new distributed matrix.
///
starneig_distr_matrix_t starneig_distr_matrix_create_local(
    int rows, int cols, starneig_datatype_t type, int owner, double *A,
    int ldA);

///
/// @brief Destroys a distributed matrix.
///
/// @param[in,out] matrix
///         The distributed matrix to be destroyed.
///
void starneig_distr_matrix_destroy(starneig_distr_matrix_t matrix);

///
/// @brief Copies the contents of a distributed matrix to a second distributed
/// matrix.
///
/// @param[in] source
///         The source matrix.
///
/// @param[out] dest
///         The destination matrix.
///
void starneig_distr_matrix_copy(
    starneig_distr_matrix_t source, starneig_distr_matrix_t dest);

///
/// @brief Copies region of a distributed matrix to a second distributed
/// matrix.
///
/// @param[in] sr
///         The first source matrix row to be copied.
///
/// @param[in] sc
///         The first source matrix column to be copied.
///
/// @param[in] dr
///         The first destination matrix row.
///
/// @param[in] dc
///         The first destination matrix column.
///
/// @param[in] rows
///         The number of rows to copy.
///
/// @param[in] cols
///         The number of columns to copy.
///
/// @param[in] source
///         The source matrix.
///
/// @param[out] dest
///         The destination matrix.
///
void starneig_distr_matrix_copy_region(
    int sr, int sc, int dr, int dc, int rows, int cols,
    starneig_distr_matrix_t source, starneig_distr_matrix_t dest);

///
/// @}
///

///
/// @name Query functions
/// @{
///

///
/// @brief Returns the locally owned distributed blocks.
///
/// @attention A user is allowed to modify the contents of the locally owned
/// blocks but the the returned array itself should not be modified.
///
/// @param[in] matrix
///         The distributed matrix.
///
/// @param[out] blocks
///         An array that contains all locally owned distributed blocks.
///
/// @param[out] num_blocks
///         The total number of locally owned distributed blocks.
///
void starneig_distr_matrix_get_blocks(
    starneig_distr_matrix_t matrix, struct starneig_distr_block **blocks,
    int *num_blocks);

///
/// @brief Returns the distribution that is associated with a distributed
/// matrix.
///
/// @attention The distributed matrix maintains the ownership of the returned
/// data distribution. A user must duplicate the data distribution if
/// necessary.
///
/// @param[in] matrix
///             The distributed matrix.
///
/// @return The associated distribution.
///
starneig_distr_t starneig_distr_matrix_get_distr(
    starneig_distr_matrix_t matrix);

///
/// @brief Returns the matrix element data type.
///
/// @param[in] matrix
///            The distributed matrix.
///
/// @return The matrix element data type.
///
starneig_datatype_t starneig_distr_matrix_get_datatype(
    starneig_distr_matrix_t matrix);

///
/// @brief Returns the matrix element size.
///
/// @param[in] matrix
///             The distributed matrix.
///
/// @return The matrix element size.
///
size_t starneig_distr_matrix_get_elemsize(starneig_distr_matrix_t matrix);

///
/// @brief Returns the number of (global) rows.
///
/// @param[in] matrix
///             The distributed matrix.
///
/// @return The number of (global) rows.
///
int starneig_distr_matrix_get_rows(starneig_distr_matrix_t matrix);

///
/// @brief Returns the number of (global) columns.
///
/// @param[in] matrix
///             The distributed matrix.
///
/// @return The number of (global) columns.
///
int starneig_distr_matrix_get_cols(starneig_distr_matrix_t matrix);

///
/// @brief Returns the number of rows in a distribution block.
///
/// @param[in] matrix
///             The distributed matrix.
///
/// @return The number of rows in a distribution block.
///
int starneig_distr_matrix_get_row_blksz(starneig_distr_matrix_t matrix);

///
/// @brief Returns the number of columns in a distribution block.
///
/// @param[in] matrix
///             The distributed matrix.
///
/// @return The number of columns in a distribution block.
///
int starneig_distr_matrix_get_col_blksz(starneig_distr_matrix_t matrix);

///
/// @}
///

///
/// @}
///

// deprecated
void starneig_broadcast(int root, size_t size, void *buffer);

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_DISTR_MATRIX_H
