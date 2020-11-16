///
/// @file
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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "cuda.h"
#include "cuda_cleanup.h"
#include "../common/common.h"
#include "../common/tiles.h"
#include <starpu.h>
#include <starpu_cublas_v2.h>

#define TILED_MATRIX_WG 32

static __constant__ __device__ double _one = 1.0;
static __constant__ __device__ double _m_one = -1.0;
static __constant__ __device__ double _zero = 0.0;

extern "C" void dlarfg_(int const *, double *, double *, int const *, double *);

///
/// @brief Custom matrix-vector multiplication CUDA kernel.
///
/// @param[in]  rbegin  first row that is included to the computation
/// @param[in]  rend    last row that is included to the computation + 1
/// @param[in]  cbegin  first column that is included to the computation
/// @param[in]  cend    last column that is included to the computation + 1
/// @param[in]  bm      tile height
/// @param[in]  bn      tile width
/// @param[in]  tiles   device side argument buffer (matrix tiles)
/// @param[in]  x       input vector
/// @param[out] y       output vector
///
static __global__ void _tiled_matrix_vector(
    int rbegin, int rend, int cbegin, int cend, int bm, int bn,
    struct tile_addr const * __restrict__ tiles, double const * __restrict__ x,
    double * __restrict__ y)
{
    __shared__ double tmp[TILED_MATRIX_WG][TILED_MATRIX_WG+1];

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = (rbegin + idx) / bm;              // tile row index
    int rid = (rbegin + idx) % bm;              // row index inside the tile row
    int rtiles = (rend-1)/bm + 1 - rbegin/bm;   // tile row count

    double v = 0.0;

    if (idx < rend - rbegin) {

        // loop over the tile columns
        int cbbegin = cbegin/bn;
        int cbend = (cend-1)/bn + 1;
        for (int i = cbbegin; i < cbend; i++) {

            // compute the correct row address inside the tile
            double const * __restrict__ ptr =
                (double const *) tiles[i*rtiles+tid].ptr + rid;
            int ld = tiles[i*rtiles+tid].ld;

            // compute the correct row address inside the input vector
            double const *_x = x+i*bn-cbegin;

            // loop over the columns in the tile (blockDim.y threads per row)
            int begin = MAX(0, cbegin - i*bn);
            int end = MIN(bn, cend - i*bn);
            for (int j = begin+threadIdx.y; j < end; j += blockDim.y)
                v += ptr[j*ld] * _x[j];
        }
    }

    // store partial sums to the shared memory
    tmp[threadIdx.x][threadIdx.y] = v;
     __syncthreads();

    // sum together the partial sums
    int active = TILED_MATRIX_WG/2;
    while (0 < active) {
        if (threadIdx.x < active)
            tmp[threadIdx.y][threadIdx.x] +=
                tmp[threadIdx.y][threadIdx.x + active];
        active /= 2;
        __syncthreads();
    }

    if (threadIdx.y == 0 && idx < rend - rbegin)
        y[idx] = tmp[threadIdx.x][0];
}

///
/// @brief Submits a custom matrix-vector multiplication CUDA kernel.
///
/// @param[in]  stream        CUDA stream
/// @param[in]  rbegin        first row that is included to the computation
/// @param[in]  rend          last row that is included to the computation + 1
/// @param[in]  cbegin        first column that is included to the computation
/// @param[in]  cend          last column that is included to the computation+1
/// @param[in]  packing_info  tile packing information
/// @param[in]  tiles         device side argument buffer (matrix tiles)
/// @param[in]  x             input vector
/// @param[out] y             output vector
///
static void tiled_matrix_vector(
    cudaStream_t stream, int rbegin, int rend, int cbegin, int cend,
    struct packing_info const *packing_info,
    struct tile_addr const *tiles, double const *x, double *y)
{
    dim3 threads(TILED_MATRIX_WG, TILED_MATRIX_WG);
    dim3 blocks(divceil(rend-rbegin, threads.x));

    _tiled_matrix_vector<<<blocks, threads, 0, stream>>>(
        packing_info->rbegin+rbegin, packing_info->rbegin+rend,
        packing_info->cbegin+cbegin, packing_info->cbegin+cend,
        packing_info->bm, packing_info->bn, tiles, x, y);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}

struct callback_args {
    int height;
    double *host;
};

///
/// @brief CUDA callback function that calls dlarfg.
///
static void CUDART_CB callback_dlarfg(
    cudaStream_t stream, cudaError_t status, void *arg_ptr)
{
    struct callback_args *args = (struct callback_args *) arg_ptr;

    double tau;
    dlarfg_(&args->height, args->host+3, args->host+4, (const int[]){ 1 },
        &tau);

    args->host[0] = tau;
    args->host[1] = -tau;
    args->host[2] = args->host[3];
    args->host[3] = 1.0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

extern "C" void starneig_hessenberg_ext_cuda_process_panel(
    void *buffers[], void *cl_args)
{
    cudaError err;

    double *one, *m_one, *zero;
    cudaGetSymbolAddress((void **)&one, _one);
    cudaGetSymbolAddress((void **)&m_one, _m_one);
    cudaGetSymbolAddress((void **)&zero, _zero);

    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y2 = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY2 = STARPU_MATRIX_GET_LD(buffers[2]);

    double *P = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldP = STARPU_MATRIX_GET_LD(buffers[3]);

    struct tile_addr *device_args =
        starneig_cuda_prepare_join_window(&packing_info, buffers+4);

    struct callback_args *args =
        (struct callback_args *) malloc(nb*sizeof(struct callback_args));

    double *host_values;
    err = cudaHostAlloc(
        &host_values, (m+3)*sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    double *_v = host_values+3;

    double *device_values;
    err = cudaMalloc(&device_values, (m+3)*sizeof(double));
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    double *tau = device_values;
    double *mtau = device_values+1;
    double *sub = device_values+2;
    double *v = device_values+3;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(handle, stream);

    starneig_cuda_join_sub_window(
        0, m, 0, nb, stream, &packing_info, device_args, ldP, P, 0);

    // loop over column in the panel
    for (int i = 0; i < nb; i++) {

        // update the current column if necessary
        if (0 < i) {

            // A <- A - Y2 * V' (update column from the right)
            cublasDgemv(handle, CUBLAS_OP_N, m, i,
                m_one, Y2, ldY2, V+i-1, ldV, one, P+i*ldP, 1);

            //
            // update column from the left
            //

            // we use the last column of T as a work space
            double *w = T+(nb-1)*ldT;

            // w <- V1' * b1 (upper part of V and column)
            cublasDcopy(handle, i, P+i*ldP, 1, w, 1);
            cublasDtrmv(handle,
                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                i, V, ldV, w, 1);

            // w <- w + V2' * b2 (lower part of V and column)
            cublasDgemv(handle, CUBLAS_OP_T, m-i, i,
                one, V+i, ldV, P+i*ldP+i, 1, one, w, 1);

            // w <- T' * w
            cublasDtrmv(handle,
                CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                i, T, ldT, w, 1);

            // b2 <- b2 - V2 * w
            cublasDgemv(handle, CUBLAS_OP_N, m-i, i,
                m_one, V+i, ldV, w, 1, one, P+i*ldP+i, 1);

            // b1 <- b1 - V1 * w
            cublasDtrmv(handle,
                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                i, V, ldV, w, 1);
            cublasDaxpy(handle, i, m_one, w, 1, P+i*ldP, 1);
        }

        //
        // form the reflector
        //

        cudaMemcpyAsync(_v, P+i*ldP+i, (m-i)*sizeof(double),
            cudaMemcpyDeviceToHost, stream);

        args[i].height = m-i;
        args[i].host = host_values;
        cudaStreamAddCallback(stream, callback_dlarfg, &args[i], 0);

        cudaMemcpyAsync(device_values, host_values, (m-i+3)*sizeof(double),
            cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(V+i*ldV+i, v, (m-i)*sizeof(double),
            cudaMemcpyDeviceToDevice, stream);

        //
        // zero the sub-diagonal elements
        //

        cudaMemcpyAsync(P+i*ldP+i, sub, sizeof(double),
            cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(P+i*ldP+i+1, 0, (m-i-1)*sizeof(double), stream);

        //
        // update Y2
        //

        // Y2(:,i) <- trailing matrix times v
        tiled_matrix_vector(stream, 0, m, i+1, n, &packing_info, device_args,
            V+i*ldV+i, Y2+i*ldY2);

        // w <- V' * v (shared result)
        cublasDgemv(handle, CUBLAS_OP_T, m-i, i,
            one, V+i, ldV, V+i*ldV+i, 1, zero, T+i*ldT, 1);

        // Y2(:,i) <- Y2(:,i) - Y * w
        cublasDgemv(handle, CUBLAS_OP_N, m, i,
            m_one, Y2, ldY2, T+i*ldT, 1, one, Y2+i*ldY2, 1);

        cublasDscal(handle, m, tau, Y2+i*ldY2, 1);

        //
        // update T
        //

        // w <- tau * w
        cublasDscal(handle, i, mtau, T+i*ldT, 1);

        // T(0:i,i) = T * w
        cublasDtrmv(handle,
            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            i, T, ldT, T+i*ldT, 1);

        cudaMemcpyAsync(T+i*ldT+i, tau, sizeof(double),
            cudaMemcpyDeviceToDevice, stream);
    }

    starneig_cuda_join_sub_window(
        0, m, 0, nb, stream, &packing_info, device_args, ldP, P, 1);

    starneig_hessenberg_ext_insert_process_panel_cleanup(
        args, host_values, device_values);
}

extern "C" void starneig_hessenberg_ext_cuda_update_trail(
    void *buffers[], void *cl_args)
{
    double *one, *m_one, *zero;
    cudaGetSymbolAddress((void **)&one, _one);
    cudaGetSymbolAddress((void **)&m_one, _m_one);
    cudaGetSymbolAddress((void **)&zero, _zero);

    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y2 = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY2 = STARPU_MATRIX_GET_LD(buffers[2]);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int nA = STARPU_MATRIX_GET_NY(buffers[3]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[3]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[4]);
    int mW = STARPU_MATRIX_GET_NX(buffers[4]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[4]);

    int max_width = MIN(nA, mW);

    struct tile_addr *device_args =
        starneig_cuda_prepare_join_window(&packing_info, buffers+5);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(handle, stream);

    for (int i = 0; i < n; i += max_width) {

        //
        // join tiles and update from the right
        //

        starneig_cuda_join_sub_window(0, m, i, MIN(n, i+max_width),
            stream, &packing_info, device_args, ldA, A, 0);

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            m, MIN(max_width, n-i), nb, m_one,
            Y2, ldY2, V+i+nb-1, ldV, one, A, ldA);

        //
        // update from the left
        //

        int width = MIN(max_width, n-i);
        if (0 < width) {
            cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, width, nb,
                one, A, ldA, zero, A, ldA, W, ldW);

            cublasDtrmm(
                handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_UNIT, width, nb, one, V, ldV, W, ldW, W, ldW);

            if (nb < m)
                cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, width, nb, m-nb,
                    one, A+nb, ldA, V+nb, ldV, one, W, ldW);

            cublasDtrmm(
                handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, width, nb, one, T, ldT, W, ldW, W, ldW);

            if (nb < m)
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m-nb, width, nb,
                    m_one, V+nb, ldV, W, ldW, one, A+nb, ldA);

            cublasDtrmm(
                handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                CUBLAS_DIAG_UNIT, width, nb, one, V, ldV, W, ldW, W, ldW);

            cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, nb, width,
                one, A, ldA, m_one, W, ldW, A, ldA);
        }

        //
        // copy tiles back
        //

        starneig_cuda_join_sub_window(0, m, i, MIN(n, i+max_width),
            stream, &packing_info, device_args, ldA, A, 1);
    }
}

extern "C" void starneig_hessenberg_ext_cuda_update_right(
    void *buffers[], void *cl_args)
{
    double *one, *m_one, *zero;
    cudaGetSymbolAddress((void **)&one, _one);
    cudaGetSymbolAddress((void **)&m_one, _m_one);
    cudaGetSymbolAddress((void **)&zero, _zero);

    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[2]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[3]);

    struct tile_addr *device_args =
        starneig_cuda_prepare_join_window(&packing_info, buffers + 4);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(handle, stream);

    starneig_cuda_join_window(stream, &packing_info, device_args, ldA, A, 0);

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, nb,
        one, A, ldA, zero, A, ldA, W, ldW);

    cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, nb, one, V, ldV, W, ldW, W, ldW);

    if (nb < n)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, nb, n-nb,
            one, A+nb*ldA, ldA, V+nb, ldV, one, W, ldW);

    cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nb, one, T, ldT, W, ldW, W, ldW);

    if (nb < n)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n-nb, nb,
            m_one, W, ldW, V+nb, ldV, one, A+nb*ldA, ldA);

    cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T, CUBLAS_DIAG_UNIT, m, nb, one, V, ldV, W, ldW, W, ldW);

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, nb,
        one, A, ldA, m_one, W, ldW, A, ldA);

    starneig_cuda_join_window(stream, &packing_info, device_args, ldA, A, 1);
}

extern "C" void starneig_hessenberg_ext_cuda_update_left(
    void *buffers[], void *cl_args)
{
    double *one, *m_one, *zero;
    cudaGetSymbolAddress((void **)&one, _one);
    cudaGetSymbolAddress((void **)&m_one, _m_one);
    cudaGetSymbolAddress((void **)&zero, _zero);

    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[2]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[3]);

    struct tile_addr *device_args =
        starneig_cuda_prepare_join_window(&packing_info, buffers+4);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(handle, stream);

    starneig_cuda_join_window(stream, &packing_info, device_args, ldA, A, 0);

    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, nb,
        one, A, ldA, zero, A, ldA, W, ldW);

    cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, nb, one, V, ldV, W, ldW, W, ldW);

    if (nb < m)
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, nb, m-nb,
            one, A+nb, ldA, V+nb, ldV, one, W, ldW);

    cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nb, one, T, ldT, W, ldW, W, ldW);

    if (nb < m)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m-nb, n, nb,
            m_one, V+nb, ldV, W, ldW, one, A+nb, ldA);

    cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, nb, one, V, ldV, W, ldW, W, ldW);

    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nb, n,
        m_one, W, ldW, one, A, ldA, A, ldA);

    starneig_cuda_join_window(stream, &packing_info, device_args, ldA, A, 1);
}
