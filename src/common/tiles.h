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

#ifndef STARNEIG_COMMON_TILES_H
#define STARNEIG_COMMON_TILES_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "vector.h"
#include "matrix.h"
#include <starpu.h>

///
/// @brief Packing mode flag data type.
///
typedef unsigned packing_mode_flag_t;

///
/// @brief Empty packing mode flag.
///
static const packing_mode_flag_t PACKING_MODE_DEFAULT               = 0;

///
/// @brief Assumes that the matrix is in the upper Hessenberg form.
///
static const packing_mode_flag_t PACKING_MODE_UPPER_HESSENBERG      = 1;

///
/// @brief Assumes that the matrix in in the upper triangular form.
///
static const packing_mode_flag_t PACKING_MODE_UPPER_TRIANGULAR      = 2;

///
/// @brief Calls starpu_data_unregister_submit when the packing helper is freed.
///
static const packing_mode_flag_t PACKING_MODE_SUBMIT_UNREGISTER     = 4;

///
/// @brief Window packing information structure.
///
struct packing_info {
    packing_mode_flag_t flag; ///< the packing mode flag
    size_t elemsize;          ///< the matrix element size
    int bm;                   ///< the tile height
    int bn;                   ///< the tile width
    int rbegin;               ///< the first row
    int rend;                 ///< the last row + 1
    int cbegin;               ///< the first column
    int cend;                 ///< the last column + 1
    int m;                    ///< the number of rows in the matrix
    int n;                    ///< the number of columns in the matrix
    int roffset;              ///< row offset from the beginning of the matrix
    int coffset;              ///< column offset rom the beginning of the matrix
    int handles;              ///< the total number of handles
#ifdef STARNEIG_ENABLE_EVENTS
    char event_label;
    int event_enabled;
    int event_roffset;
    int event_coffset;
#endif
};

///
/// @brief Range packing information structure.
///
struct range_packing_info {
    packing_mode_flag_t flag;   ///< packing mode flag
    size_t elemsize;            ///< element size
    int bm;                     ///< tile height
    int begin;                  ///< first row
    int end;                    ///< last row + 1
    int m;                      ///< the number of rows in the vector
    int offset;                 ///< offset from the beginning of the vector
    int handles;                ///< total number of handles
};

///
/// @brief Data packing helper structure.
///
struct packing_helper {
    struct starpu_data_descr *descrs;   ///< data descriptors
    packing_mode_flag_t *flags;         ///< packing mode flags
    int size;                           ///< data descriptor array size
    int count;                          ///< data descriptor count
};

///
/// @brief Initializes an empty packing info structure.
///
/// @param[out] info  returns an empty packing info structure
///
void starneig_init_empty_packing_info(struct packing_info *info);

///
/// @brief Initializes an empty range packing info structure.
///
/// @param[out] info  returns an empty range packing info structure
///
void starneig_init_empty_range_packing_info(struct range_packing_info *info);

///
/// @brief Initializes a data packing helper structure.
///
/// @return new data packing helper structure
///
struct packing_helper * starneig_init_packing_helper();

///
/// @brief Frees a previously allocated data packing helper structure.
///
/// @param[in,out] helper  data packing helper
///
void starneig_free_packing_helper(struct packing_helper *helper);

///
/// @brief Packs a data handle into a packing helper.
///
/// @param[in]     mode    access mode
/// @param[in]     handle  data handle
/// @param[in,out] helper  data packing helper
/// @param[in]     flag    packing mode flag
///
void starneig_pack_handle(
    enum starpu_data_access_mode mode, starpu_data_handle_t handle,
    struct packing_helper *helper, packing_mode_flag_t flag);

///
/// @brief Packs a scratch matrix into a packing helper.
///
/// @param[in]    m         row count
/// @param[in]    n         column count
/// @param[in]    elemsize  element size
/// @param[in,out] helper   data packing helper
///
void starneig_pack_scratch_matrix(
    int m, int n, size_t elemsize, struct packing_helper *helper);

///
/// @brief Packs a cached scratch matrix into a packing helper.
///
/// @param[in]    m         row count
/// @param[in]    n         column count
/// @param[in]    elemsize  element size
/// @param[in,out] helper   data packing helper
///
void starneig_pack_cached_scratch_matrix(
    int m, int n, size_t elemsize, struct packing_helper *helper);

///
/// @brief Packs a range into a packing helper.
///
/// @param[in]     mode    access mode
/// @param[in]     begin   first row that belongs to the range
/// @param[in]     end     last row that belongs to the range + 1
/// @param[in,out] vector  vector descriptor
/// @param[in,out] helper  data packing helper
/// @param[out]    info    returns tile packing information
/// @param[in]     flag    packing mode flag
///
void starneig_pack_range(
    enum starpu_data_access_mode mode, int begin, int end,
    starneig_vector_descr_t vector, struct packing_helper *helper,
    struct range_packing_info *info, packing_mode_flag_t flag);

///
/// @brief Packs a window into a packing helper.
///
/// @param[in]     mode    access mode
/// @param[in]     rbegin  first row that belongs to the window
/// @param[in]     rend    last row that belongs to the window + 1
/// @param[in]     cbegin  first column that belongs to the window
/// @param[in]     cend    last column that belongs to the window + 1
/// @param[in,out] matrix  matrix descriptor
/// @param[in,out] helper  data packing helper
/// @param[out]    info    returns tile packing information
/// @param[in]     flag    packing mode flag
///
void starneig_pack_window(
    enum starpu_data_access_mode mode,
    int rbegin, int rend, int cbegin, int cend,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag);

///
/// @brief Packs a diagonal window into a packing helper.
///
/// @param[in]     mode    access mode
/// @param[in]     begin   first row/column that belongs to the window
/// @param[in]     end     last row/column that belongs to the window + 1
/// @param[in,out] matrix  matrix descriptor
/// @param[in,out] helper  data packing helper
/// @param[out]    info    returns tile packing information
/// @param[in]     flag    packing mode flag
///
void starneig_pack_diag_window(
    enum starpu_data_access_mode mode, int begin, int end,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Copies a range to a continuous memory buffer.
///
/// @param[in]     packing_info  tile packing info
/// @param[in,out] in            input interfaces
/// @param[in,out] out           output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_join_range(
    struct range_packing_info const *packing_info,
    struct starpu_vector_interface **in, void *out, int reverse);

///
/// @brief Copies a window to a continuous memory buffer.
///
/// @param[in]     packing_info  tile packing info
/// @param[in]     ld            output buffer leading dimension
/// @param[in,out] in            input interfaces
/// @param[in,out] out           output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_join_window(
    struct packing_info const *packing_info, size_t ld,
    struct starpu_matrix_interface **in, void *out, int reverse);

///
/// @brief Copies a sub-window to a continuous memory buffer.
///
/// @param[in]     rbegin        first row that belongs to the sub-window
/// @param[in]     rend          last row that belongs to the sub-window + 1
/// @param[in]     cbegin        first column that belongs to the sub-window
/// @param[in]     cend          last column that belongs to the sum-window + 1
/// @param[in]     packing_info  tile packing info
/// @param[in]     ld            output buffer leading dimension
/// @param[in,out] in            input interfaces
/// @param[in,out] out           output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_join_sub_window(
    int rbegin, int rend, int cbegin, int cend,
    struct packing_info const *packing_info, size_t ld,
    struct starpu_matrix_interface **in, void *out, int reverse);

///
/// @brief Copies a diagonal window to a continuous memory buffer.
///
/// @param[in]     packing_info  tile packing info
/// @param[in]     ld            output buffer leading dimension
/// @param[in,out] in            input interfaces
/// @param[in,out] out           output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_join_diag_window(
    struct packing_info const *packing_info, size_t ld,
    struct starpu_matrix_interface **in, void *out, int reverse);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef STARNEIG_ENABLE_CUDA

///
/// @brief A structure that stores tile's address and leading dimension.
///
struct __align__(16) tile_addr {
    uintptr_t ptr;      ///< tile address (in device memory)
    int ld;             ///< tile leading dimension
};

#ifdef __NVCC__

#ifdef __cplusplus
extern "C" {
#endif

///
/// @brief Prepares to copy a range to a continuous memory buffer.
///
/// @param[in]     packing_info  tile packing info
/// @param[in,out] interfaces    input interfaces
///
/// @return device side argument buffer
///
uintptr_t* starneig_cuda_prepare_join_range(
    struct range_packing_info const *packing_info, void **in);

///
/// @brief Copies a range to a continuous memory buffer.
///
/// @param[in]     stream        CUDA stream
/// @param[in]     packing_info  tile packing info
/// @param[in]     device_args   device side argument buffer
/// @param[in,out] ptr           input/output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_cuda_join_range(
    cudaStream_t stream, struct range_packing_info const *packing_info,
    uintptr_t* device_args, void *ptr, int reverse);

///
/// @brief Prepares to copy a window to a continuous memory buffer.
///
/// @param[in]     packing_info  tile packing info
/// @param[in,out] interfaces    input interfaces
///
/// @return device side argument buffer
///
struct tile_addr * starneig_cuda_prepare_join_window(
    struct packing_info const *packing_info, void **in);

///
/// @brief Copies a window to a continuous memory buffer.
///
/// @param[in]     stream        CUDA stream
/// @param[in]     packing_info  tile packing info
/// @param[in]     device_args   device side argument buffer
/// @param[in]     ld            output buffer leading dimension
/// @param[in,out] ptr           input/output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_cuda_join_window(
    cudaStream_t stream, struct packing_info const *packing_info,
    struct tile_addr const *device_args, int ld, void *ptr, int reverse);

///
/// @brief Copies a sub-window to a continuous memory buffer.
///
/// @param[in]     rbegin        first row that belongs to the sub-window
/// @param[in]     rend          last row that belongs to the sub-window + 1
/// @param[in]     cbegin        first column that belongs to the sub-window
/// @param[in]     cend          last column that belongs to the sum-window + 1
/// @param[in]     stream        CUDA stream
/// @param[in]     packing_info  tile packing info
/// @param[in]     device_args   device side argument buffer
/// @param[in]     ld            output buffer leading dimension
/// @param[in,out] ptr           input/output buffer
/// @param[in]     reverse       reverse copy direction
///
void starneig_cuda_join_sub_window(
    int rbegin, int rend, int cbegin, int cend,
    cudaStream_t stream, struct packing_info const *packing_info,
    struct tile_addr const *device_args, int ld, void *ptr, int reverse);

void starneig_cuda_join_diag_window(
    cudaStream_t stream, struct packing_info const *packing_info,
    struct tile_addr const *device_args, int ld, void *ptr, int reverse);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // __NVCC__

#endif // STARNEIG_ENABLE_CUDA

#endif
