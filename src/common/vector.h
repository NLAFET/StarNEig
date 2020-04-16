///
/// @file
///
/// @brief This file contains the definition of a vector descriptor.
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

#ifndef STARNEIG_COMMON_VECTOR_H
#define STARNEIG_COMMON_VECTOR_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include <starpu.h>

///
/// @brief Vector descriptor.
///
typedef struct starneig_vector_descr * starneig_vector_t;

///
/// @brief Single owner vector disctribution function.
///
int starneig_vector_single_owner_func(int i, void const *ptr);

///
/// @brief Creates an empty vector descriptor.
///
/// @param[in] m
///         Vector height (row count).
///
/// @param[in] bm
///         Tile height (row count).
///
/// @param[in] elemsize
///         Vector element size.
///
/// @param[in] distrib
///         Distribution function.
///
/// @param[in] distarg
///         Distribution function argument.
///
/// @param[in,out] mpi
///         MPI info.
///
/// @return New empty vector descriptor.
///
starneig_vector_t starneig_vector_init(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, mpi_info_t mpi);

///
/// @brief Creates a vector descriptor and registers a vector with it.
///
/// @param[in] m
///         Vector height (row count).
///
/// @param[in] bm
///         Tile height (row count).
///
/// @param[in] elemsize
///         Vector element size.
///
/// @param[in] distrib
///         Distribution function.
///
/// @param[in] distarg
///         Distribution function argument.
///
/// @param[in] vec
///         Pointer to the vector.
///
/// @param[in,out] mpi
///         MPI info.
///
/// @return New vector descriptor.
///
starneig_vector_t starneig_vector_register(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, void *vec, mpi_info_t mpi);

///
/// @brief Takes a previously initialized vector descriptor and unregister all
/// registered StarPU resources.
///
/// @param[in,out] descr
///         Vector descriptor.
///
void starneig_vector_unregister(starneig_vector_t descr);

///
/// @brief Frees a previously initialized vector descriptor.
///
/// @param[in,out] descr
///         Vector descriptor.
///
void starneig_vector_free(starneig_vector_t descr);

///
/// @brief Returns the first row that belongs to a vector descriptor.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return First row that belongs to the vector descriptor.
///
int starneig_vector_get_rbegin(const starneig_vector_t descr);

///
/// @brief Returns the last row that belongs to a vector descriptor.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Last row that belongs to the vector descriptor + 1.
///
int starneig_vector_get_rend(const starneig_vector_t descr);

///
/// @brief Returns the length of a vector descriptor.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Length of the vector descriptor.
///
int starneig_vector_get_rows(const starneig_vector_t descr);

///
/// @brief Returns the length of a vector descriptor tile.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Length of the vector descriptor tile.
///
int starneig_vector_get_tile_size(const starneig_vector_t descr);

///
/// @brief Returns the element size of a vector descriptor.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Element size of the vector descriptor.
///
size_t starneig_vector_get_elemsize(const starneig_vector_t descr);

///
/// @brief Checks whether a vector descriptor is distributed.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Non-zero is the vector descriptor is distributed.
///
int starneig_vector_is_distributed(const starneig_vector_t descr);

///
/// @brief Return the tile row index of the tile that contains a given row.
///
/// @param[in] row
///         Row index.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return The tile row index of the tile that contains a given row.
///
int starneig_vector_get_tile_idx(
    int row, const starneig_vector_t descr);

///
/// @brief Return the in-tile row index of a row.
///
/// @param[in] row
///         Row index.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return The in-tile row index of a row.
///
int starneig_vector_get_in_tile_idx(
    int row, int tile, const starneig_vector_t descr);

///
/// @brief Return the extern row index of a row given the tile and in-tile row
/// indeces.
///
/// @param[in] tile
///         Tile row index.
///
/// @param[in] row
///         In-tile row index.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return The extern row index of a row.
///
int starneig_vector_get_ext_idx(
    int tile, int row, starneig_vector_t descr);

///
/// @brief Registers a tile with a vector descriptor.
///
/// @param[in] i
///         Tile row index.
///
/// @param[in] handle
///         Tile data handle.
///
/// @param[in,out] descr
///         Vector descriptor.
///
void starneig_vector_set_tile(
    int i, starpu_data_handle_t handle, starneig_vector_t descr);

///
/// @brief Returns a tile from a vector descriptor.
///
/// @param[in] i
///         Tile row index.
///
/// @param[in,out] descr
///         Vector descriptor.
///
/// @return Tile handle.
///
starpu_data_handle_t starneig_vector_get_tile(
    int i, starneig_vector_t descr);

///
/// @brief Gathers the content of a vector descriptor to a node.
///
/// @param[in] node
///         MPI node.
///
/// @param[in] descr
///         Vector descriptor.
///
void starneig_vector_gather(int root, starneig_vector_t descr);

///
/// @brief Scatters the content of a vector descriptor from a node.
///
/// @param[in] node
///         MPI node.
///
/// @param[in] descr
///         Vector descriptor.
///
void starneig_vector_scatter(int node, starneig_vector_t descr);

///
/// @brief Gathers a section of a vector descriptor to a node.
///
/// $param[in] node
///         MPI node.
///
/// @param[in] begin
///         First row to be gathered.
///
/// @param[in] end
///         Last row to be gathered + 1.
///
/// $param[in] descr
///         Vector descriptor.
///
void starneig_vector_gather_section(
    int node, int begin, int end, starneig_vector_t descr);

///
/// @brief Scatters a section of a vector descriptor from a node.
///
/// $param[in] node
///         MPI node.
///
/// @param[in] begin
///         First row to be scattered.
///
/// @param[in] end
///         Last row to be scattered + 1.
///
/// $param[in] descr
///         Vector descriptor.
///
void starneig_vector_scatter_section(
    int node, int begin, int end, starneig_vector_t descr);

///
/// @brief Returns the owner of a given tile.
///
/// @param[in] i
///         Tile row index.
///
/// $param[in] descr
///         Vector descriptor.
///
int starneig_vector_get_tile_owner(int i, starneig_vector_t descr);

///
/// @brief Returns the owner of a given vector element.
///
/// @param[in] i
///         Row index.
///
/// $param[in] descr
///         Vector descriptor.
///
int starneig_vector_get_elem_owner(int i, starneig_vector_t descr);

///
/// @brief Checks whether the current MPI rank is involved with a section of a
/// distributed vector.
///
/// @param[in] begin  first row that belongs to the section
/// @param[in] end    last row that belongs to the section + 1
///
/// @return 1 if the MPI rank is involved, 0 otherwise
///
int starneig_vector_involved_with_section(
    int begin, int end, starneig_vector_t descr);

///
/// @brief Cuts the vector from the nearest tile boundary (rounded upwards).
///
/// @param[in] row
///         Cutting point candidate.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Cutting point that follows the underlying tile boundaries.
///
int starneig_vector_cut_up(int row, const starneig_vector_t descr);

///
/// @brief Cuts the vector from the nearest tile boundary (rounded downwards).
///
/// @param[in] row
///         Cutting point candidate.
///
/// @param[in] descr
///         Vector descriptor.
///
/// @return Cutting point that follows the underlying tile boundaries.
///
int starneig_vector_cut_down(int row, const starneig_vector_t descr);

#endif
