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

#ifndef STARNEIG_COMMON_UTILS_H
#define STARNEIG_COMMON_UTILS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "math.h"
#include "vector.h"
#include "matrix.h"

///
/// @brief Checks whether a given matrix has valid dimensions.
///
/// @param[in] n          desired matrix dimension
/// @param[in] tile_size  desired tile size
/// @param[in] descr      matrix descriptor
///
/// @return non-zero if the dimensions are valid, zero otherwise
///
int starneig_is_valid_matrix(
    int n, int tile_size, const starneig_matrix_descr_t descr);

///
/// @brief Calculates a valid update task width/height
///
/// @param[in] n - matrix dimension
/// @param[in] bn - tile size
/// @param[in] sbn - section size (in tiles)
/// @param[in] world_size - MPI world size
/// @param[in] worker_count - worker count (per MPI rank)
///
/// @return valid update task width/height
///
int starneig_calc_update_size(
    int n, int bn, int sbn, int world_size, int worker_count);

///
/// @brief Registers a vector descriptor that has a matching tile layout to a
/// given matrix descriptor.
///
/// @param[in] descr - matrix descriptor
/// @param[in] elemsize - vector element size
/// @param[in,out] vec - pointer to the vector
/// @param[in,out] mpi  MPI info
///
/// @returns a vector descriptor that has a matching tile layout
///
starneig_vector_descr_t starneig_init_matching_vector_descr(
    const starneig_matrix_descr_t descr, size_t elemsize, void *vec,
    mpi_info_t mpi);

///
/// @brief Checks whether the sub-diagonal entries are non-zero.
///
/// @param[in] descr - matrix descriptor structure
/// @param[in,out] mpi  MPI info
///
/// @return locations of the non-zero sub-diagonals
///
starneig_vector_descr_t starneig_extract_subdiagonals(
    starneig_matrix_descr_t descr, mpi_info_t mpi);

///
/// @brief Acquires the whole vector descriptor and returns a local copy of it's
/// contents.
///
/// @param[in] descr - vector descriptor structure
///
/// @return local copy
///
void * starneig_acquire_vector_descr(starneig_vector_descr_t descr);

#endif
