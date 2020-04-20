///
/// @file
///
/// @brief This file contains the definition of a matrix descriptor that is used
/// throughout the all components of the library.
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
struct starneig_matrix_descr {
    int rbegin;                           ///< first row
    int rend;                             ///< last row + 1
    int cbegin;                           ///< first column
    int cend;                             ///< last column + 1
    int bm;                               ///< tile height (row count)
    int bn;                               ///< tile width (column count)
    int sbm;                              ///< section height (tile row count)
    int sbn;                              ///< section width (tile column count)
    int elemsize;                         ///< element size
    int tm_count;                         ///< number of tile rows
    int tn_count;                         ///< number of tile columns
#ifdef STARNEIG_ENABLE_MPI
    int tag_offset;                       ///< tag offset
    int **owners;                         ///< section owners (MPI ranks)
#endif
    starpu_data_handle_t **tiles;         ///< tiles
    struct starneig_matrix_descr *parent; ///< parent matrix
    enum {
        STARNEIG_MATRIX_ROOT = 0,         ///< the matrix is a root matrix
        STARNEIG_MATRIX_SUB_MATRIX,       ///< the matrix is a submatrix
    } mode;                               ///< matrix descriptor mode
#ifdef STARNEIG_ENABLE_EVENTS
    char event_label;
    int event_enabled;
    int event_roffset;
    int event_coffset;
#endif
};

///
/// @brief Matrix descriptor data type.
///
typedef struct starneig_matrix_descr * starneig_matrix_descr_t;

///
/// @brief Returns the first row that belongs to the (sub)matrix.
///
/// @param[in] descr
///         Matrix Descriptor
///
/// @return First row that belongs to the (sub)matrix.
///
static inline int STARNEIG_MATRIX_RBEGIN(const starneig_matrix_descr_t descr)
{
    return descr->rbegin;
}

///
/// @brief Returns the last row that belongs to the (sub)matrix + 1.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Last row that belongs to the (sub)matrix + 1.
///
static inline int STARNEIG_MATRIX_REND(const starneig_matrix_descr_t descr)
{
    return descr->rend;
}

///
/// @brief Returns the first column that belongs to the (sub)matrix.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return First column that belongs to the (sub)matrix.
///
static inline int STARNEIG_MATRIX_CBEGIN(const starneig_matrix_descr_t descr)
{
    return descr->cbegin;
}

///
/// @brief Returns the last column that belongs to the (sub)matrix + 1.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Last column that belongs to the (sub)matrix + 1.
///
static inline int STARNEIG_MATRIX_CEND(const starneig_matrix_descr_t descr)
{
    return descr->cend;
}

///
/// @brief Returns the height of the (sub)matrix.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return (Sub)matrix height.
///
static inline int STARNEIG_MATRIX_M(const starneig_matrix_descr_t descr)
{
    return descr->rend - descr->rbegin;
}

///
/// @brief Returns the width of the (sub)matrix.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return (Sub)matrix width.
///
static inline int STARNEIG_MATRIX_N(const starneig_matrix_descr_t descr)
{
    return descr->cend - descr->cbegin;
}

///
/// @brief Returns the tile height.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Tile height.
///
static inline int STARNEIG_MATRIX_BM(const starneig_matrix_descr_t descr)
{
    return descr->bm;
}

/// @brief Returns the tile width.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Tile width.
///
static inline int STARNEIG_MATRIX_BN(const starneig_matrix_descr_t descr)
{
    return descr->bn;
}

///
/// @brief Returns the section height in tiles.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section height in tiles.
///
static inline int STARNEIG_MATRIX_SBM(const starneig_matrix_descr_t descr)
{
    return descr->sbm;
}

///
/// @brief Returns the section width in tiles.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section width in tiles.
///
static inline int STARNEIG_MATRIX_SBN(const starneig_matrix_descr_t descr)
{
    return descr->sbn;
}

///
/// @brief Returns the section height.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section height.
///
static inline int STARNEIG_MATRIX_SM(const starneig_matrix_descr_t descr)
{
    return descr->sbm*descr->bm;
}

///
/// @brief Returns the section width.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Section width.
///
static inline int STARNEIG_MATRIX_SN(const starneig_matrix_descr_t descr)
{
    return descr->sbn*descr->bn;
}

///
/// @brief Returns the element size.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return element size.
///
static inline size_t STARNEIG_MATRIX_ELEMSIZE(
    const starneig_matrix_descr_t descr)
{
    return descr->elemsize;
}

///
/// @brief Checks whether the matrix is distributed.
///
/// @param[in] descr
///         Matrix descriptor.
///
/// @return Non-zero is the matrix is distributed, zero otherwise.
///
static inline int STARNEIG_MATRIX_DISTRIBUTED(
    const starneig_matrix_descr_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return 0 <= descr->tag_offset;
#else
    return 0;
#endif
}

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
static inline int STARNEIG_MATRIX_TILE_IDX(
    int row, const starneig_matrix_descr_t descr)
{
    return (STARNEIG_MATRIX_RBEGIN(descr) + row) / STARNEIG_MATRIX_BM(descr);
}

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
static inline int STARNEIG_MATRIX_TILE_IDY(
    int column, const starneig_matrix_descr_t descr)
{
    return (STARNEIG_MATRIX_CBEGIN(descr) + column) / STARNEIG_MATRIX_BN(descr);
}

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
static inline int starneig_matrix_cut_vectically_up(
    int row, const starneig_matrix_descr_t descr)
{
    int rbegin = STARNEIG_MATRIX_RBEGIN(descr);
    int bm = STARNEIG_MATRIX_BM(descr);
    int m = STARNEIG_MATRIX_M(descr);

    return MAX(0, MIN(m, ((rbegin + row) / bm) * bm - rbegin));
}

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
static inline int starneig_matrix_cut_vectically_down(
    int row, const starneig_matrix_descr_t descr)
{
    int rbegin = STARNEIG_MATRIX_RBEGIN(descr);
    int bm = STARNEIG_MATRIX_BM(descr);
    int m = STARNEIG_MATRIX_M(descr);

    return MAX(0, MIN(m, divceil(rbegin + row, bm) * bm - rbegin));
}

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
static inline int starneig_matrix_cut_horizontally_left(
    int column, const starneig_matrix_descr_t descr)
{
    int cbegin = STARNEIG_MATRIX_CBEGIN(descr);
    int bn = STARNEIG_MATRIX_BN(descr);
    int n = STARNEIG_MATRIX_N(descr);

    return MAX(0, MIN(n, ((cbegin + column) / bn) * bn - cbegin));
}

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
static inline int starneig_matrix_cut_horizontally_right(
    int column, const starneig_matrix_descr_t descr)
{
    int cbegin = STARNEIG_MATRIX_CBEGIN(descr);
    int bn = STARNEIG_MATRIX_BN(descr);
    int n = STARNEIG_MATRIX_N(descr);

    return MAX(0, MIN(n, divceil(cbegin + column, bn) * bn - cbegin));
}

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
static inline int starneig_single_owner_matrix_descr(
    int i, int j, void const *ptr)
{
    return *((int *) ptr);
}

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
starneig_matrix_descr_t starneig_init_matrix_descr(
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
starneig_matrix_descr_t starneig_register_matrix_descr(
    enum starneig_matrix_type type, int m, int n, int bm, int bn, int sbm,
    int sbn, int ld, size_t elemsize, int (*distrib)(int, int, void const *),
    void const *distarg, void *mat, mpi_info_t mpi);

///
/// @brief Creates a matrix descriptor that encapsulates a sub-matrix.
///
/// @param[in] rbegin
///         First row that belongs to the sub-matrix.
///
/// @param[in] rend
///         Last row that belong to the sub-matrix + 1.
///
/// @param[in] cbegin
///         First column that belongs to the sub-matrix.
///
/// @param[in] cend
///         Last column that belongs to the sub-matrix + 1.
///
/// @param[in,out] descr
///         Parent matrix descriptor.
///
/// @return New matrix descriptor.
///
starneig_matrix_descr_t starneig_create_sub_matrix_descr(
    int rbegin, int rend, int cbegin, int cend, starneig_matrix_descr_t descr);

///
/// @brief Takes a previously initialized matrix descriptor and acquires all
/// registered StarPU resources to main memory.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_acquire_matrix_descr(starneig_matrix_descr_t descr);

///
/// @brief Takes a previously initialized matrix descriptor and releases all
/// registered StarPU resources from main memory.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_release_matrix_descr(starneig_matrix_descr_t descr);

///
/// @brief Takes a previously initialized matrix descriptor and unregister all
/// registered StarPU resources.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_unregister_matrix_descr(starneig_matrix_descr_t descr);

///
/// @brief Frees a previously initialized matrix descriptor.
///
/// @param[in,out] descr
///         Matrix descriptor.
///
void starneig_free_matrix_descr(starneig_matrix_descr_t descr);

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
void starneig_register_tile_with_matrix_descr(int i, int j,
    starpu_data_handle_t handle, starneig_matrix_descr_t descr);

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
starpu_data_handle_t starneig_get_tile_from_matrix_descr(
    int i, int j, starneig_matrix_descr_t descr);

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
starpu_data_handle_t starneig_get_elem_from_matrix_descr(
    int i, int j, starneig_matrix_descr_t descr, mpi_info_t mpi);

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
void starneig_register_section_with_matrix_descr(
    enum starneig_matrix_type type, int i, int j, int ld, void *mat,
    starneig_matrix_descr_t descr);

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
int starneig_get_tile_owner_matrix_descr(
    int i, int j, const starneig_matrix_descr_t descr);

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
int starneig_get_elem_owner_matrix_descr(
    int i, int j, const starneig_matrix_descr_t descr);

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
int starneig_involved_with_part_of_matrix_descr(
    int rbegin, int rend, int cbegin, int cend,
    const starneig_matrix_descr_t descr);

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
void starneig_flush_section_matrix_descr(
    int rbegin, int rend, int cbegin, int cend, starneig_matrix_descr_t descr);

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
void starneig_prefetch_section_matrix_descr(
    int rbegin, int rend, int cbegin, int cend, int node, int async,
    const starneig_matrix_descr_t descr);

#endif
