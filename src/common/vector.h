///
/// @file
///
/// @brief This file contains the definition of a vector descriptor.
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

#ifndef STARNEIG_COMMON_VECTOR_H
#define STARNEIG_COMMON_VECTOR_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include <starpu.h>

///
/// @brief Vector descriptor structure.
///
struct starneig_vector_descr {
    int rbegin;                             ///< first row
    int rend;                               ///< last row + 1
    int bm;                                 ///< tile height (row count)
    int elemsize;                           ///< element size
    int tm_count;                           ///< number of tile rows
    starpu_data_handle_t *tiles;            ///< data tiles
#ifdef STARNEIG_ENABLE_MPI
    int tag_offset;                         ///< tag offset
    int *owners;                            ///< section owners (MPI ranks)
#endif
    struct starneig_vector_descr *parent;   ///< parent descriptor
    enum {
        STARNEIG_VECTOR_ROOT,               ///< root descriptor (no parent)
        STARNEIG_VECTOR_SUB_VECTOR,         ///< sub-vector descriptor
    } mode;                                 ///< descriptor mode
};

typedef struct starneig_vector_descr * starneig_vector_descr_t;

static inline int STARNEIG_VECTOR_RBEGIN(const starneig_vector_descr_t descr)
{
    return descr->rbegin;
}

static inline int STARNEIG_VECTOR_REND(const starneig_vector_descr_t descr)
{
    return descr->rend;
}

static inline int STARNEIG_VECTOR_M(const starneig_vector_descr_t descr)
{
    return descr->rend - descr->rbegin;
}

static inline int STARNEIG_VECTOR_BM(const starneig_vector_descr_t descr)
{
    return descr->bm;
}

static inline size_t STARNEIG_VECTOR_ELEMSIZE(
    const starneig_vector_descr_t descr)
{
    return descr->elemsize;
}

static inline int STARNEIG_VECTOR_DISTRIBUTED(
    const starneig_vector_descr_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return 0 <= descr->tag_offset;
#else
    return 0;
#endif
}

static inline int STARNEIG_VECTOR_TILE_IDX(
    int row, const starneig_vector_descr_t descr)
{
    return (STARNEIG_VECTOR_RBEGIN(descr) + row) / STARNEIG_VECTOR_BM(descr);
}

static inline int STARNEIG_VECTOR_IN_TILE_IDX(
    int row, int tile, const starneig_vector_descr_t descr)
{
    return
        STARNEIG_VECTOR_RBEGIN(descr) + row - tile * STARNEIG_VECTOR_BM(descr);
}

static inline int STARNEIG_VECTOR_EXT_IDX(
    int tile, int row, starneig_vector_descr_t descr)
{
    return
        tile * STARNEIG_VECTOR_BM(descr) + STARNEIG_VECTOR_RBEGIN(descr) + row;
}

///
/// @brief Cuts the vector from the nearest tile boundary (rounded upwards).
///
/// @param[in]  row    cutting point candidate
/// @param[in]  descr  vector descriptor
///
/// @return cutting point that follows the underlying tile boundaries
///
static inline int starneig_vector_cut_up(
    int row, const starneig_vector_descr_t descr)
{
    int rbegin = STARNEIG_VECTOR_RBEGIN(descr);
    int bm = STARNEIG_VECTOR_BM(descr);

    return MAX(0, ((rbegin + row) / bm) * bm - rbegin);
}

///
/// @brief Cuts the vector from the nearest tile boundary (rounded downwards).
///
/// @param[in]  row    cutting point candidate
/// @param[in]  descr  vector descriptor
///
/// @return cutting point that follows the underlying tile boundaries
///
static inline int starneig_vector_cut_down(
    int row, const starneig_vector_descr_t descr)
{
    int rbegin = STARNEIG_VECTOR_RBEGIN(descr);
    int bm = STARNEIG_VECTOR_BM(descr);

    return MAX(0, divceil(rbegin + row, bm) * bm - rbegin);
}

static inline int starneig_single_owner_vector_descr(int i, void const *ptr)
{
    return *((int const *) ptr);
}

///
/// @brief Creates an empty vector descriptor structure.
///
/// @param[in] m - vector height (row count)
/// @param[in] bm - tile height (row count)
/// @param[in] elemsize - element size
/// @param[in] distrib - distribution function
/// @param[in] distarg - distribution function argument
/// @param[in,out] mpi  MPI info
///
/// @return a pointer to the new vector descriptor structure
///
starneig_vector_descr_t starneig_init_vector_descr(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, mpi_info_t mpi);

///
/// @brief Creates a vector descriptor structure and registers a vector with it.
///
/// @param[in] m - vector height (row count)
/// @param[in] bm - tile height (row count)
/// @param[in] elemsize - element size
/// @param[in] distrib - distribution function
/// @param[in] distarg - distribution function argument
/// @param[in,out] vec - pointer to the vector
/// @param[in,out] mpi  MPI info
///
/// @return a pointer to the new vector descriptor structure
///
starneig_vector_descr_t starneig_register_vector_descr(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, void *vec, mpi_info_t mpi);

///
/// @brief Creates a sub-vector descriptor structure.
///
/// @param[in]    begin   first row that belongs to the sub-vector
/// @param[in]    end     last row that belong to the sub-vector + 1
/// @param[in,out] descr   parent vector descriptor structure
///
/// @return new sub-vector descriptor structure
///
starneig_vector_descr_t starneig_create_sub_vector_descr(
    int begin, int end, starneig_vector_descr_t descr);

///
/// @brief Takes a previously initialized vector descriptor structure and
/// unregister all registered StarPU resources.
///
/// @param[in,out] descr - pointer to the vector descriptor structure
///
void starneig_unregister_vector_descr(starneig_vector_descr_t descr);

///
/// @brief Frees a previously initialized vector descriptor structure.
///
/// @param[in,out] descr - pointer to the vector descriptor structure
///
void starneig_free_vector_descr(starneig_vector_descr_t descr);

///
/// @brief Registers a tile with a vector descriptor structure
///
/// @param[in] i - tile's row index
/// @param[in] handle - tile handle
/// @param[in,out] descr - pointer to the vector descriptor structure
///
void starneig_register_tile_with_vector_descr(
    int i, starpu_data_handle_t handle, starneig_vector_descr_t descr);

///
/// @brief Returns a tile from a vector descriptor structure.
///
/// @param[in] i - tile's row index
/// @param[in,out] descr - pointer to the vector descriptor structure
///
/// @return the tile handle
///
starpu_data_handle_t starneig_get_tile_from_vector_descr(
    int i, starneig_vector_descr_t descr);

///
/// @brief Gathers the contents of a vector descriptor structure to a node.
///
/// $param[in] root - root node
/// $param[in] descr - pointer to the vector descriptor structure
///
void starneig_gather_vector_descr(int root, starneig_vector_descr_t descr);

///
/// @brief Scatters the contents of a vector descriptor structure from a node.
///
/// $param[in] root - root node
/// $param[in] descr - pointer to the vector descriptor structure
///
void starneig_scatter_vector_descr(int root, starneig_vector_descr_t descr);

///
/// @brief Gathers the contents of a vector descriptor structure to a node.
///
/// $param[in] root   root node
/// $param[in] begin  first row to be gathered
/// $param[in] end    last row to be gathered + 1
/// $param[in] descr  pointer to the vector descriptor structure
///
void starneig_gather_segment_vector_descr(
    int root, int begin, int end, starneig_vector_descr_t descr);

///
/// @brief Scatters the contents of a vector descriptor structure from a node.
///
/// $param[in] root   root node
/// $param[in] begin  first row to be gathered
/// $param[in] end    last row to be gathered + 1
/// $param[in] descr  pointer to the vector descriptor structure
///
void starneig_scatter_segment_vector_descr(
    int root, int begin, int end, starneig_vector_descr_t descr);

///
/// @brief Returns the owner of a given data tile.
///
/// @param[in]     i  tile's row index
/// @param[in] descr  pointer to the vector descriptor structure
///
int starneig_get_tile_owner_vector_descr(int i, starneig_vector_descr_t descr);

///
/// @brief Returns the owner of a given vector element.
///
/// @param[in]     i  row index
/// @param[in] descr  pointer to the vector descriptor structure
///
int starneig_get_elem_owner_vector_descr(int i, starneig_vector_descr_t descr);

///
/// @brief Checks whether the current MPI rank is involved with a section of a
/// distributed vector.
///
/// @param[in] begin  first row that belongs to the section
/// @param[in] end    last row that belongs to the section + 1
///
/// @return 1 if the MPI rank is involved, 0 otherwise
///
int starneig_involved_with_part_of_vector_descr(
    int begin, int end, starneig_vector_descr_t descr);

#endif
