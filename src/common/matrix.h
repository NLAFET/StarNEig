///
/// @file
///
/// @brief This file contains the definition of a matrix descriptor that is used
/// throughout the all components of the library.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
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

#ifndef STARNEIG_COMMON_MATRIX_H
#define STARNEIG_COMMON_MATRIX_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include <starpu.h>

///
/// @brief Matrix type enumerator.
///
enum starneig_matrix_type {
    MATRIX_TYPE_FULL,             ///< full matrix
    MATRIX_TYPE_UPPER_HESSENBERG, ///< upper Hessenberg matrix
    MATRIX_TYPE_UPPER_TRIANGULAR  ///< upper triangular matrix
};

///
/// @brief Matrix descriptor.
///
///  A m x n matrix is spliced into sm x sn sections which are further spliced
///  into bm x bn tiles as shown below:
///
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  ################################################################# ^  ^  ^
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   # bm |  |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   # |  |  |
///  #---+---+---+---#---+---+---+---#---+---+---+---#---+---+---+---# '  |  |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #    sm |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #    |  |
///  #---+---+---+---#---+---+---+---#---+---+---+---#---+---+---+---#    |  |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #    |  |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #    |  |
///  #################################################################    '  |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #---+---+---+---#---+---+---+---#---+---+---+---#---+---+---+---#       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       m
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #---+---+---+---#---+---+---+---#---+---+---+---#---+---+---+---#       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #################################################################       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #---+---+---+---#---+---+---+---#---+---+---+---#---+---+---+---#       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #---+---+---+---#---+---+---+---#---+---+---+---#---+---+---+---#       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #   |   |   |   #   |   |   |   #   |   |   |   #   |   |   |   #       |
///  #################################################################       '
///  <bn >
///  <----- sn ------>
///  <------------------------------ n ------------------------------>
///
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
typedef struct starneig_matrix_descr * starneig_matrix_t;

///
/// @brief Matrix distribution function for a matrix that has a single owner.
///
/// @param[in] i
///         Section row index.
///
/// @param[in] j
///         Section column index.
///
/// @param[in] ptr
///         Distribution function argument (int-pointer to the owner).
///
/// @return Section owner.
///
int starneig_single_owner_matrix_descr(int i, int j, void const *ptr);

///
/// @brief Creates an empty matrix descriptor.
///
/// @param[in] m
///         Matrix height (row count).
///
/// @param[in] n
///         Matrix width (column count).
///
/// @param[in] bm
///         Tile height (row count).
///
/// @param[in] bn
///         Tile width (column count).
///
/// @param[in] sbm
///         Section height (tile count).
///
/// @param[in] sbn
///         Section width (tile count).
///
/// @param[in] elemsize
///         Element size.
///
/// @param[in] distrib
///         Distribution function.
///
/// @param[in] distrib
///         Distribution function argument.
///
/// @param[in,out] tag_offset
///         MPI info.
///
/// @return New matrix descriptor.
///
starneig_matrix_t starneig_matrix_init(
    int m, int n, int bm, int bn, int sbm, int sbn, size_t elemsize,
    int (*distrib)(int, int, void const *), void const *distarg,
    mpi_info_t mpi);

///
/// @brief Creates a matrix descriptor and registers a matrix with it.
///
/// @param[in] type
///         Matrix type.
///
/// @param[in] m
///         Matrix height (row count).
///
/// @param[in] n
///         Matrix width (column count).
///
/// @param[in] bm
///         Tile height (row count).
///
/// @param[in] bn
///         Tile width (column count).
///
/// @param[in]   sbm
///         Section height (tile count).
///
/// @param[in] sbn
///         Section width (tile count).
///
/// @param[in] ld
///         First dimension of the matrix.
///
/// @param[in] elemsize
///         Matrix element size.
///
/// @param[in] distrib
///         Distribution function.
///
/// @param[in] distarg
///         Distribution function argument.
///
/// @param[in,out] mat
///         Pointer to the matrix.
///
/// @param[in,out] tag_offset
///         MPI info.
///
/// @return New matrix descriptor.
///
starneig_matrix_t starneig_matrix_register(
    enum starneig_matrix_type type, int m, int n, int bm, int bn, int sbm,
    int sbn, int ld, size_t elemsize, int (*distrib)(int, int, void const *),
    void const *distarg, void *mat, mpi_info_t mpi);

///
/// @brief Takes a previously initialized matrix descriptor and acquires all
/// registered StarPU resources to main memory.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_acquire(starneig_matrix_t descr);

///
/// @brief Takes a previously initialized matrix descriptor and releases all
/// registered StarPU resources from main memory.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_release(starneig_matrix_t descr);

///
/// @brief Takes a previously initialized matrix descriptor and unregister all
/// registered StarPU resources.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_unregister(starneig_matrix_t descr);

///
/// @brief Frees a previously initialized matrix descriptor.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_free(starneig_matrix_t descr);

///
/// @brief Registers a tile with a matrix descriptor
///
/// @param[in] i
///         Tile's row index.
///
/// @param[in] j
///         Tile's column index.
///
/// @param[in] handle
///         Tile handle.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_set_tile(int i, int j,
    starpu_data_handle_t handle, starneig_matrix_t descr);

///
/// @brief Returns a tile from a matrix descriptor.
///
/// @param[in] i
///         Tile's row index.
///
/// @param[in] j
///         Tile's column index.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
/// @return The tile handle.
///
starpu_data_handle_t starneig_matrix_get_tile(
    int i, int j, starneig_matrix_t descr);

///
/// @brief Returns an element from a matrix descriptor.
///
/// @param[in] i
///         row index.
///
/// @param[in] j
///         column index.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
/// @param[in,out] tag_offset
///         MPI info.
///
/// @return The tile handle.
///
starpu_data_handle_t starneig_matrix_get_elem(
    int i, int j, starneig_matrix_t descr, mpi_info_t mpi);

///
/// @brief Registers a section with a matrix descriptor.
///
/// @param[in] type
///         Matrix type.
///
/// @param[in] i
///         Section's row index.
///
/// @param[in] j
///         Section's column index.
///
/// @param[in] ld
///         First dimension of the matrix.
///
/// @param[in] mat
///         Pointer to the matrix.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_register_section(
    enum starneig_matrix_type type, int i, int j, int ld, void *mat,
    starneig_matrix_t descr);

///
/// @brief Returns the owner of a given tile.
///
/// @param[in] i
///         Tile's row index.
///
/// @param[in] j
///         Tile's column index.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Owner's MPI rank.
///
int starneig_matrix_get_tile_owner(
    int i, int j, const starneig_matrix_t descr);

///
/// @brief Returns the owner of a given matrix element.
///
/// @param[in] i
///         row index.
///
/// @param[in] j
///         column index.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Owner's MPI rank.
///
int starneig_matrix_get_elem_owner(
    int i, int j, const starneig_matrix_t descr);

///
/// @brief Checks whether the current MPI rank is involved with a section of a
/// distributed matrix.
///
/// @param[in] rbegin
///         First row that belongs to the section.
///
/// @param[in] rend
///         Last row that belongs to the section + 1.
///
/// @param[in] cbegin
///         First column that belongs to the section.
///
/// @param[in] cend
///         Last row that belong to the section + 1.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return 1 if the MPI rank is involved, 0 otherwise.
///
int starneig_matrix_involved_with_section(
    int rbegin, int rend, int cbegin, int cend,
    const starneig_matrix_t descr);

///
/// @brief Flushes a section of a distributed matrix.
///
/// @param[in] rbegin
///         First row that belongs to the section.
///
/// @param[in] rend
///         Last row that belongs to the section + 1.
///
/// @param[in] cbegin
///         First column that belongs to the section.
///
/// @param[in] cend
///         Last row that belong to the section + 1.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_flush_section(
    int rbegin, int rend, int cbegin, int cend, starneig_matrix_t descr);

///
/// @brief Prefetches a section of a distributed matrix.
///
/// @param[in] rbegin
///         First row that belongs to the section.
///
/// @param[in] rend
///         Last row that belongs to the section + 1.
///
/// @param[in] cbegin
///         First column that belongs to the section.
///
/// @param[in] cend
///         Last row that belong to the section + 1.
///
/// @param[in] node
///         The memory node.
///
/// @param[in] async
///         Asynchronicity flag.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_matrix_prefetch_section(
    int rbegin, int rend, int cbegin, int cend, int node, int async,
    const starneig_matrix_t descr);

///
/// @brief Returns the first row that belongs to the (sub)matrix.
///
/// @param[in] descr
///         Matrix Descriptor
///
/// @return First row that belongs to the (sub)matrix.
///
int STARNEIG_MATRIX_RBEGIN(const starneig_matrix_t descr);

///
/// @brief Returns the last row that belongs to the (sub)matrix + 1.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Last row that belongs to the (sub)matrix + 1.
///
int STARNEIG_MATRIX_REND(const starneig_matrix_t descr);

///
/// @brief Returns the first column that belongs to the (sub)matrix.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return First column that belongs to the (sub)matrix.
///
int STARNEIG_MATRIX_CBEGIN(const starneig_matrix_t descr);

///
/// @brief Returns the last column that belongs to the (sub)matrix + 1.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Last column that belongs to the (sub)matrix + 1.
///
int STARNEIG_MATRIX_CEND(const starneig_matrix_t descr);

///
/// @brief Returns the height of the (sub)matrix.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return (Sub)matrix height.
///
int STARNEIG_MATRIX_M(const starneig_matrix_t descr);

///
/// @brief Returns the width of the (sub)matrix.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return (Sub)matrix width.
///
int STARNEIG_MATRIX_N(const starneig_matrix_t descr);

///
/// @brief Returns the tile height.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Tile height.
///
int STARNEIG_MATRIX_BM(const starneig_matrix_t descr);

/// @brief Returns the tile width.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Tile width.
///
int STARNEIG_MATRIX_BN(const starneig_matrix_t descr);

///
/// @brief Returns the section height in tiles.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section height in tiles.
///
int STARNEIG_MATRIX_SBM(const starneig_matrix_t descr);
///
/// @brief Returns the section width in tiles.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section width in tiles.
///
int STARNEIG_MATRIX_SBN(const starneig_matrix_t descr);

///
/// @brief Returns the section height.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section height.
///
int STARNEIG_MATRIX_SM(const starneig_matrix_t descr);

///
/// @brief Returns the section width.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section width.
///
int STARNEIG_MATRIX_SN(const starneig_matrix_t descr);

///
/// @brief Returns the element size.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return element size.
///
size_t STARNEIG_MATRIX_ELEMSIZE(const starneig_matrix_t descr);

///
/// @brief Checks whether the matrix is distributed.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Non-zero is the matrix is distributed, zero otherwise.
///
int STARNEIG_MATRIX_DISTRIBUTED(const starneig_matrix_t descr);

///
/// @brief Returns the tile row index that matches a given row.
///
/// @param[in] row
///         Row.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Tile row index that matches a given row.
///
int STARNEIG_MATRIX_TILE_IDX(int row, const starneig_matrix_t descr);

///
/// @brief Returns the tile column index that matches a given column.
///
/// @param[in] column
///         Column.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Tile column index that matches a given column.
///
int STARNEIG_MATRIX_TILE_IDY(int column, const starneig_matrix_t descr);

///
/// @brief Cuts the matrix vertically from the nearest tile boundary (rounded
/// upwards).
///
/// @param[in] row
///         Cutting point candidate.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Vertical cutting point that follows the underlying tile boundaries.
///
int starneig_matrix_cut_ver_up(int row, const starneig_matrix_t descr);

///
/// @brief Cuts the matrix vertically from the nearest tile boundary (rounded
/// downwards).
///
/// @param[in] row
///         Cutting point candidate.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Vertical cutting point that follows the underlying tile boundaries.
///
int starneig_matrix_cut_ver_down(int row, const starneig_matrix_t descr);

///
/// @brief Cuts the matrix horizontally from the nearest tile boundary (rounded
/// left).
///
/// @param[in] column
///         Cutting point candidate.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Horizontally cutting point that follows the underlying tile
/// boundaries.
///
int starneig_matrix_cut_hor_left(int column, const starneig_matrix_t descr);

///
/// @brief Cuts the matrix horizontally from the nearest tile boundary (rounded
/// right).
///
/// @param[in] column
///         Cutting point candidate.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Horizontally cutting point that follows the underlying tile
/// boundaries.
///
int starneig_matrix_cut_hor_right(int column, const starneig_matrix_t descr);

#endif
