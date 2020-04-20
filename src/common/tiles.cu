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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "tiles.h"
#include "common.h"
#include "cuda_cleanup.h"

///
/// @brief Copies a range into a continuous memory buffer.
///
/// @param[in]     begin    first row that belongs to the range
/// @param[in]     end      last row that belongs to the range + 1
/// @param[in]     bm       tile height
/// @param[in]     dsize    element size
/// @param[in]     int      device side argument buffer
/// @param[in,out] out      output buffer
/// @param[in]     reverse  if non-zero, direction is reversed
///
static __global__ void join_range(
    int begin, int end, int bm, size_t dsize,
    uintptr_t const * __restrict__ in, uintptr_t out, int reverse)
{
    int bbegin = begin/bm;
    int bend = (end-1)/bm + 1;

    for (int i = bbegin+blockIdx.x; i < bend; i += gridDim.x) {

        // vertical bounds
        int _begin = MAX(0, begin - i*bm);
        int _end = MIN(bm, end - i*bm);
        int size = _end - _begin;

        uintptr_t _in, _out;
        if (reverse) {
            _in = out + dsize*MAX(0, i*bm-begin);
            _out = in[i] + dsize*_begin;
        }
        else {
            _in = in[i] + dsize*_begin;
            _out = out + dsize*MAX(0, i*bm-begin);
        }

        #define copy(_l, type) { \
            type const * __restrict__ __in = (type const *)_in; \
            type * __restrict__ __out = (type *)_out; \
            for (int j = threadIdx.x; j < size*_l; j += blockDim.x) \
                __out[j] = __in[j]; \
        }

        switch (dsize) {
            case sizeof(unsigned long long):
                copy(1, unsigned long long);
                break;
            case sizeof(unsigned int):
                copy(1, unsigned int);
                break;
            default:
                copy(dsize, unsigned char);
        }

        #undef copy
    }
}

///
/// @brief Copies a sub-matrix.
///
/// @param[in]  m      row count
/// @param[in]  n      column count
/// @param[in]  lda    leading dimension of the input matrix
/// @param[in]  ldb    leading dimension of the output matrix
/// @param[in]  dsize  element size
/// @param[in]  a      pointer to the input matrix
/// @param[out] b      pointer to the output matrix
///
static __device__ void copy_submatrix(
    int m, int n, int lda, int ldb, size_t dsize,
    uintptr_t const a, uintptr_t b)
{
    #define copy(_l, type) { \
        type const * __restrict__ __in = (type const *)a; \
        type * __restrict__ __out = (type *)b; \
        for (int i = blockIdx.z; i < n; i += gridDim.z) \
            for (int j = threadIdx.x; j < m*_l; j += blockDim.x) \
                __out[(size_t)i*ldb*_l+j] = __in[(size_t)i*lda*_l+j]; \
    }

    switch (dsize) {
        case sizeof(unsigned long long):
            copy(1, unsigned long long);
            break;
        case sizeof(unsigned int):
            copy(1, unsigned int);
            break;
        default:
            copy(dsize, unsigned char);
    }

    #undef copy
}

static __device__ void zero_submatrix(
    int m, int n, int lda, size_t dsize, uintptr_t a)
{
    #define zero(_l, type) { \
        type *__out = (type *)a; \
        for (int i = blockIdx.z; i < n; i += gridDim.z) \
            for (int j = threadIdx.x; j < m*_l; j += blockDim.x) \
                __out[(size_t)i*lda*_l+j] = 0; \
    }

    switch (dsize) {
        case sizeof(unsigned long long):
            zero(1, unsigned long long);
            break;
        case sizeof(unsigned int):
            zero(1, unsigned int);
            break;
        default:
            zero(dsize, unsigned char);
    }

    #undef zero
}

///
/// @brief Copies a computation window into a continuous memory buffer.
///
///                  |- cbegin              |- cend
///               +-----+-----------------------+
///               |     |     |     |     |     |
///     rbegin -> |  ###|#####|#####|#####|#.   |
///               |  ###|#####|#####|#####|#.   |
///               +-----+-----+-----+-----+-----+
///               |  ###|#####|#####|#####|#.   |
///               |  ###|#####|#####|#####|#.   |
///               |  ###|#####|#####|#####|#.   |
///               +-----+-----+-----+-----+-----+
///               |  ###|#####|#####|#####|#.   |
///       rend -> |  ...|.....|.....|.....|..   |
///               |     |     |     |     |     |
///               +-----+-----+-----+-----+-----+
///                             ||
///                             \/
///                   +-------------------+
///                   |###################|
///                   |###################|
///                   |###################|
///          out ===> |###################|
///                   |###################|
///                   |###################|
///                   |¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤|
///                   +-------------------+
///
/// @param[in]     rbegin   first row that belongs to the window
/// @param[in]     rend     last row that belongs to the window + 1
/// @param[in]     cbegin   first column that belongs to the window
/// @param[in]     cend     last column that belongs to the window + 1
/// @param[in]     bm       tile height
/// @param[in]     bn       tile width
/// @param[in]     in_ld    device side argument buffer leading dimension
/// @param[in]     out_ld   output buffer leading dimension
/// @param[in]     dsize    element size
/// @param[in]     in       device side argument buffer
/// @param[in,out] out      output buffer
/// @param[in]     reverse  if non-zero, direction is reversed
///
static __global__ void join_tiles_full(
    int rbegin, int rend, int cbegin, int cend,
    int bm, int bn, int in_ld, int out_ld, size_t dsize,
    struct tile_addr const * __restrict__ in, uintptr_t out, int reverse)
{
    int rbbegin = rbegin/bm;
    int rbend = (rend-1)/bm + 1;

    int cbbegin = cbegin/bn;
    int cbend = (cend-1)/bn + 1;

    for (int i = cbbegin+blockIdx.y; i < cbend; i += gridDim.y) {

        // vertical bounds
        int _cbegin = MAX(0, cbegin - i*bn);
        int _cend = MIN(bn, cend - i*bn);

        // output buffer column offset
        int coffset = MAX(0, i*bn-cbegin);

        for (int j = rbbegin+blockIdx.x; j < rbend; j += gridDim.x) {

            // horizontal bounds
            int _rbegin = MAX(0, rbegin - j*bm);
            int _rend = MIN(bm, rend - j*bm);

            // output buffer row offset
            int roffset = MAX(0, j*bm-rbegin);

            uintptr_t _ptr = (uintptr_t) in[i*in_ld+j].ptr;
            int _ld = in[i*in_ld+j].ld;

            if (reverse)
                copy_submatrix(
                    _rend - _rbegin, _cend - _cbegin, out_ld, _ld, dsize,
                    out + ((size_t)coffset*out_ld + roffset)*dsize,
                    _ptr + ((size_t)_cbegin*_ld + _rbegin)*dsize);
            else
                copy_submatrix(
                    _rend - _rbegin, _cend - _cbegin, _ld, out_ld, dsize,
                    _ptr + ((size_t)_cbegin*_ld + _rbegin)*dsize,
                     out + ((size_t)coffset*out_ld + roffset)*dsize);
        }
    }
}

static __global__ void join_tiles_upper_hess(
    int rbegin, int rend, int cbegin, int cend,
    int bm, int bn, int out_ld, size_t dsize,
    struct tile_addr const * __restrict__ in, uintptr_t out, int reverse)
{
    int rbbegin = rbegin/bm;
    int rbend = (rend-1)/bm + 1;

    int cbbegin = cbegin/bn;
    int cbend = (cend-1)/bn + 1;

    int tid = 0;
    int k = 0;
    for (int i = cbbegin; i < cbend; i++) {

        // vertical bounds
        int _cbegin = MAX(0, cbegin - i*bn);
        int _cend = MIN(bn, cend - i*bn);

        // output buffer column offset
        int coffset = MAX(0, i*bn-cbegin);

        for (int j = rbbegin; j < rbend; j++) {

            if (k % gridDim.x != blockIdx.x) {
                if (rbegin+j*bm <= cbegin+(i+1)*bn)
                    tid++;

                k++;
                continue;
            }

            // horizontal bounds
            int _rbegin = MAX(0, rbegin - j*bm);
            int _rend = MIN(bm, rend - j*bm);

            // output buffer row offset
            int roffset = MAX(0, j*bm-rbegin);

            if (rbegin+j*bm <= cbegin+(i+1)*bn) {

                //
                // copy
                //

                uintptr_t _ptr = (uintptr_t) in[tid].ptr;
                int _ld = in[tid].ld;

                if (reverse)
                    copy_submatrix(
                        _rend - _rbegin, _cend - _cbegin, out_ld, _ld, dsize,
                        out + ((size_t)coffset*out_ld + roffset)*dsize,
                        _ptr + ((size_t)_cbegin*_ld + _rbegin)*dsize);
                else
                    copy_submatrix(
                        _rend - _rbegin, _cend - _cbegin, _ld, out_ld, dsize,
                        _ptr + ((size_t)_cbegin*_ld + _rbegin)*dsize,
                         out + ((size_t)coffset*out_ld + roffset)*dsize);

                tid++;
            }
            else if (!reverse) {

                //
                // zero
                //

                zero_submatrix(
                    _rend - _rbegin, _cend - _cbegin, out_ld, dsize,
                     out + ((size_t)coffset*out_ld + roffset)*dsize);
            }

            k++;
        }
    }
}

static __global__ void join_tiles_upper_triag(
    int rbegin, int rend, int cbegin, int cend,
    int bm, int bn, int out_ld, size_t dsize,
    struct tile_addr const * __restrict__ in, uintptr_t out, int reverse)
{
    int rbbegin = rbegin/bm;
    int rbend = (rend-1)/bm + 1;

    int cbbegin = cbegin/bn;
    int cbend = (cend-1)/bn + 1;

    int tid = 0;
    int k = 0;
    for (int i = cbbegin; i < cbend; i++) {

        // vertical bounds
        int _cbegin = MAX(0, cbegin - i*bn);
        int _cend = MIN(bn, cend - i*bn);

        // output buffer column offset
        int coffset = MAX(0, i*bn-cbegin);

        for (int j = rbbegin; j < rbend; j++) {

            if (k % gridDim.x != blockIdx.x) {
                if (rbegin+j*bm < cbegin+(i+1)*bn)
                    tid++;

                k++;
                continue;
            }

            // horizontal bounds
            int _rbegin = MAX(0, rbegin - j*bm);
            int _rend = MIN(bm, rend - j*bm);

            // output buffer row offset
            int roffset = MAX(0, j*bm-rbegin);

            if (rbegin+j*bm < cbegin+(i+1)*bn) {

                //
                // copy
                //

                uintptr_t _ptr = (uintptr_t) in[tid].ptr;
                int _ld = in[tid].ld;

                if (reverse)
                    copy_submatrix(
                        _rend - _rbegin, _cend - _cbegin, out_ld, _ld, dsize,
                        out + ((size_t)coffset*out_ld + roffset)*dsize,
                        _ptr + ((size_t)_cbegin*_ld + _rbegin)*dsize);
                else
                    copy_submatrix(
                        _rend - _rbegin, _cend - _cbegin, _ld, out_ld, dsize,
                        _ptr + ((size_t)_cbegin*_ld + _rbegin)*dsize,
                         out + ((size_t)coffset*out_ld + roffset)*dsize);

                tid++;
            }
            else if (!reverse) {

                //
                // zero
                //

                zero_submatrix(
                    _rend - _rbegin, _cend - _cbegin, out_ld, dsize,
                     out + ((size_t)coffset*out_ld + roffset)*dsize);
            }

            k++;
        }
    }

}

static void CUDART_CB callback_free(
    cudaStream_t stream, cudaError_t status, void *arg_ptr)
{
    free(arg_ptr);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

uintptr_t* starneig_cuda_prepare_join_range(
    struct range_packing_info const *packing_info, void **in)
{
    cudaError err;

    if (packing_info->handles == 0)
        return NULL;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    uintptr_t *device_args;
    err = cudaMalloc(&device_args, packing_info->handles*sizeof(uintptr_t));
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    uintptr_t *host_args = (uintptr_t *)
        malloc(packing_info->handles*sizeof(uintptr_t));

    for (int i = 0; i < packing_info->handles; i++)
        host_args[i] = STARPU_VECTOR_GET_PTR(in[i]);

    err = cudaMemcpyAsync(device_args, host_args,
        packing_info->handles*sizeof(uintptr_t), cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    starneig_insert_cuda_free(device_args);
    cudaStreamAddCallback(stream, callback_free, host_args, 0);

    return device_args;
}

void starneig_cuda_join_range(
    cudaStream_t stream, struct range_packing_info const *packing_info,
    uintptr_t *device_args, void *ptr, int reverse)
{
    if (packing_info->handles == 0)
        return;

    int tiles = (packing_info->end-1)/packing_info->bm + 1 -
        packing_info->begin/packing_info->bm;

    join_range<<<tiles, MIN(256, packing_info->bm), 0, stream>>>(
        packing_info->begin, packing_info->end,
        packing_info->bm, packing_info->elemsize,
        device_args, (uintptr_t) ptr, reverse);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}

extern "C" struct tile_addr * starneig_cuda_prepare_join_window(
    struct packing_info const *packing_info, void **in)
{
    cudaError err;

    if (packing_info->handles == 0)
        return NULL;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    struct tile_addr *device_args;
    err = cudaMalloc(
        &device_args, packing_info->handles*sizeof(struct tile_addr));
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    struct tile_addr *host_args = (struct tile_addr *)
        malloc(packing_info->handles*sizeof(struct tile_addr));

    for (int i = 0; i < packing_info->handles; i++) {
        host_args[i].ptr = STARPU_MATRIX_GET_PTR(in[i]);
        host_args[i].ld = STARPU_MATRIX_GET_LD(in[i]);
    }

    err = cudaMemcpyAsync(device_args, host_args,
        packing_info->handles*sizeof(struct tile_addr),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    starneig_insert_cuda_free(device_args);
    cudaStreamAddCallback(stream, callback_free, host_args, 0);

    return device_args;
}

extern "C" void starneig_cuda_join_window(
    cudaStream_t stream, struct packing_info const *packing_info,
    struct tile_addr const *device_args, int ld, void *ptr, int reverse)
{
    if (packing_info->handles == 0)
        return;

    int rtiles = (packing_info->rend-1)/packing_info->bm + 1 -
        packing_info->rbegin/packing_info->bm;
    int ctiles = (packing_info->cend-1)/packing_info->bn + 1 -
        packing_info->cbegin/packing_info->bn;

    int threads = MIN(256, packing_info->bm);
    dim3 blocks(rtiles, ctiles, MAX(1, packing_info->bn/32));

    join_tiles_full<<<blocks, threads, 0, stream>>>(
        packing_info->rbegin, packing_info->rend,
        packing_info->cbegin, packing_info->cend,
        packing_info->bm, packing_info->bn,
        rtiles, ld, packing_info->elemsize,
        device_args, (uintptr_t) ptr, reverse);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}

extern "C" void starneig_cuda_join_sub_window(
    int rbegin, int rend, int cbegin, int cend,
    cudaStream_t stream, struct packing_info const *packing_info,
    struct tile_addr const *device_args, int ld, void *ptr, int reverse)
{
    if (packing_info->handles == 0)
        return;

    int rtiles = (packing_info->rend-1)/packing_info->bm + 1 -
        packing_info->rbegin/packing_info->bm;
    int ctiles = (packing_info->cend-1)/packing_info->bn + 1 -
        packing_info->cbegin/packing_info->bn;

    dim3 blocks(rtiles, ctiles);
    int threads = MIN(128, divceil(packing_info->bm,32)*32);

    join_tiles_full<<<blocks, threads, 0, stream>>>(
        packing_info->rbegin+rbegin, packing_info->rbegin+rend,
        packing_info->cbegin+cbegin, packing_info->cbegin+cend,
        packing_info->bm, packing_info->bn,
        rtiles, ld, packing_info->elemsize,
        device_args, (uintptr_t) ptr, reverse);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}

extern "C" void starneig_cuda_join_diag_window(
    cudaStream_t stream, struct packing_info const *packing_info,
    struct tile_addr const *device_args, int ld, void *ptr, int reverse)
{
    if (packing_info->handles == 0)
        return;

    if (packing_info->flag & PACKING_MODE_UPPER_HESSENBERG) {
        int threads = MIN(256, packing_info->bm);
        dim3 blocks(packing_info->handles-1, 1, MAX(1, packing_info->bn/32));
        join_tiles_upper_hess<<<blocks, threads, 0, stream>>>(
            packing_info->rbegin, packing_info->rend,
            packing_info->cbegin, packing_info->cend,
            packing_info->bm, packing_info->bn,
            ld, packing_info->elemsize, device_args, (uintptr_t) ptr, reverse);
    }
    else if (packing_info->flag & PACKING_MODE_UPPER_TRIANGULAR) {
        int threads = MIN(256, packing_info->bm);
        dim3 blocks(packing_info->handles-1, 1, MAX(1, packing_info->bn/32));
        join_tiles_upper_triag<<<blocks, threads, 0, stream>>>(
            packing_info->rbegin, packing_info->rend,
            packing_info->cbegin, packing_info->cend,
            packing_info->bm, packing_info->bn,
            ld, packing_info->elemsize, device_args, (uintptr_t) ptr, reverse);
    }
    else {
        int rtiles = (packing_info->rend-1)/packing_info->bm + 1 -
            packing_info->rbegin/packing_info->bm;
        int ctiles = (packing_info->cend-1)/packing_info->bn + 1 -
            packing_info->cbegin/packing_info->bn;

        int threads = MIN(256, packing_info->bm);
        dim3 blocks(rtiles, ctiles, MAX(1, packing_info->bn/32));
        join_tiles_full<<<blocks, threads, 0, stream>>>(
            packing_info->rbegin, packing_info->rend,
            packing_info->cbegin, packing_info->cend,
            packing_info->bm, packing_info->bn,
            rtiles, ld, packing_info->elemsize,
            device_args, (uintptr_t) ptr, reverse);
    }

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}
