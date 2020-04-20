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

#ifndef STARNEIG_MPI_DISTR_MATRIX_INTERNAL_H
#define STARNEIG_MPI_DISTR_MATRIX_INTERNAL_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/distr_matrix.h>
#include "../common/matrix.h"
#include <stddef.h>

///
/// @brief Descriptor cache entry index.
///
typedef int descr_cache_entry_t;

#define DESCR_CACHE_EMPTY -1

///
/// @brief Data distribution structure
///
struct starneig_distr {
    /// The data distribution type.
    enum {
        /// Artibraty function defined distribution.
        STARNEIG_DISTR_TYPE_FUNC,
        /// Two-dimensional block cyclic distribution, row-major ordering.
        STARNEIG_DISTR_TYPE_2DBC_ROW,
        /// Two-dimensional block cyclic distribution, column-major ordering.
        STARNEIG_DISTR_TYPE_2DBC_COL
    } type;
    /// The data distribution function.
    int (*func)(int row, int col, void *arg);
    /// The data distribution function argument.
    void *arg;
    /// The data distribution function argument size.
    size_t arg_size;
    /// The number of rows in the 2DBC mesh.
    int rows;
    /// The number of columns in the 2DBC mesh.
    int cols;
};

///
/// @brief Distributed matrix struc.
///
struct starneig_distr_matrix {
    /// The associated data distribution.
    struct starneig_distr *distr;
    /// The number of (global) rows in the matrix.
    int rows;
    /// The number of (global) columns in the matrix.
    int cols;
    /// The number of rows in a distributed block.
    int row_blksz;
    /// The number of columns in a distributed block.
    int col_blksz;
    /// The number of locally owned distributed blocks
    int block_count;
    /// The locally owned distributed blocks.
    struct starneig_distr_block *blocks;
    /// A pointer to the local buffer.
    void *ptr;
    /// The leading dimension of the local buffer.
    size_t ld;
    /// If non-zero, the local buffer gets freed when the matrix is destroyed.
    int free_ptr;
    /// The matrix element data type.
    starneig_datatype_t datatype;
    /// Descriptor cache entry.
    descr_cache_entry_t descr;
};

///
/// @brief Returns the matrix element size of a given matrix element data type.
///
/// @brief[in] type
///         The matrix element data type.
///
/// @return The matrix element size.
///
static inline size_t starneig_mpi_get_elemsize(starneig_datatype_t type)
{
    return sizeof(double);
}

///
/// @brief Converts a distributed matrix to a matrix descriptor and adds in to
/// the descriptor cache. The data handles ARE acquired.
///
/// @param[in] bm
///         The number of rows in a tile.
///
/// @param[in] bn
///         The number of columns in a tile.
///
/// @param[in] fill
///         The matrix fill mode.
///
/// @param[in,out] matrix
///         The distributed matrix.
///
/// @param[in,out]
///         The MPI info.
///
/// @return The matching entry in the descriptor cache.
///
starneig_matrix_descr_t starneig_mpi_cache_convert(
    int bm, int bn, enum starneig_matrix_type fill,
    starneig_distr_matrix_t matrix, mpi_info_t mpi);

///
/// @brief Converts a distributed matrix to a matrix descriptor and adds in to
/// the descriptor cache. The data handles are NOT acquired.
///
/// @param[in] bm
///         The number of rows in a tile.
///
/// @param[in] bn
///         The number of columns in a tile.
///
/// @param[in] fill
///         The matrix fill mode.
///
/// @param[in,out] matrix
///         The distributed matrix.
///
/// @param[in,out]
///         The MPI info.
///
/// @return The matching entry in the descriptor cache.
///
starneig_matrix_descr_t starneig_mpi_cache_convert_and_release(
    int bm, int bn, enum starneig_matrix_type fill,
    starneig_distr_matrix_t matrix, mpi_info_t mpi);

///
/// @brief Removes a distributed matrix from the descriptor cache.
///
/// @param[in] matrix
///         The distributed matrix.
///
void starneig_mpi_cache_remove(starneig_distr_matrix_t matrix);

///
/// @brief Clears the descriptor cache.
///
void starneig_mpi_cache_clear();

#endif // STARNEIG_MPI_DISTR_MATRIX_INTERNAL_H
