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
#include "cuda.h"
#include "../common/common.h"
#include "../common/tiles.h"
#include <starpu.h>
#include <starpu_cublas_v2.h>

static const double *one = (const double[]) { 1.0 };
static const double *m_one = (const double[]) { -1.0 };
static const double *zero = (const double[]) { 0.0 };

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
/// @param[in]  A       device side argument buffer (matrix tiles)
/// @param[in]  x       device side argument buffer (input vector)
/// @param[out] y       device side argument buffer (output vector)
///
static __global__ void tiled_matrix_vector(
    int rbegin, int rend, int cbegin, int cend, int bm, int bn,
    struct tile_addr const * __restrict__ A, uintptr_t const * __restrict__ x,
    uintptr_t * __restrict__ y)
{
    extern __shared__ double s[];

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = idx / bm;                         // tile row index
    int rid = idx % bm;                         // row index inside the tile row
    int rtiles = (rend-1)/bm + 1 - rbegin/bm;   // tile row count

    double v = 0.0;

    if (rbegin <= idx && idx < rend) {

        // loop over the tile columns
        int cbbegin = cbegin/bn;
        int cbend = (cend-1)/bn + 1;
        for (int i = cbbegin; i < cbend; i++) {

            // compute the correct row address inside the tile
            double const * __restrict__ ptr =
                (double const *) A[i*rtiles+tid].ptr;
            int ld = A[i*rtiles+tid].ld;

            // loop over the columns in the tile (blockDim.y threads per row)
            int begin = MAX(0, cbegin - i*bn);
            int end = MIN(bn, cend - i*bn);
            for (int j = begin+threadIdx.y; j < end; j += blockDim.y)
                v += ptr[j*ld+rid] * ((double const *) x[i])[j];
        }
    }

    // store partial sums to the shared memory
    if (0 < threadIdx.y && rbegin <= idx && idx < rend)
        s[(threadIdx.y-1)*blockDim.x+threadIdx.x] = v;
    __syncthreads();

    // sum partial sums together and store the final result
    if (threadIdx.y == 0 && rbegin <= idx && idx < rend) {
        for (int i = 0; i < blockDim.y-1; i++)
            v += s[i*blockDim.x+threadIdx.x];
        ((double *)y[tid])[rid] = v;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_hessenberg_cuda_compute_column(
    void *buffers[], void *cl_args)
{
    struct packing_info A_pi;
    struct range_packing_info v_pi, y_pi;
    starpu_codelet_unpack_args(cl_args, &A_pi, &v_pi, &y_pi);

    int k = 0;

    // involved trailing matrix tiles
    struct tile_addr *A_da =
        starneig_cuda_prepare_join_window(&A_pi, buffers + k);
    k += A_pi.handles;

    // intemediate vector interface for the trailing matrix operation
    uintptr_t *v_da = starneig_cuda_prepare_join_range(&v_pi, buffers + k);
    k += v_pi.handles;

    // intemediate vector interface from the trailing matrix operation
    uintptr_t *y_da = starneig_cuda_prepare_join_range(&y_pi, buffers + k);
    k += y_pi.handles;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    int rtiles = (A_pi.rend-1)/A_pi.bm + 1 - A_pi.rbegin/A_pi.bm;

    dim3 threads(32, MIN(32, MAX(1, (A_pi.cend-A_pi.cbegin)/16)));
    dim3 blocks(divceil(rtiles*A_pi.bm, threads.x));
    size_t shared_size = threads.x*(threads.y-1)*sizeof(double);

    tiled_matrix_vector<<<blocks, threads, shared_size, stream>>>(
        A_pi.rbegin, A_pi.rend, A_pi.cbegin, A_pi.cend, A_pi.bm, A_pi.bn,
        A_da, v_da, y_da);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}

extern "C" void starneig_hessenberg_cuda_update_trail(
    void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb, offset;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb, &offset);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY = STARPU_MATRIX_GET_LD(buffers[2]);

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
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    for (int i = 0; i < n; i += max_width) {

        //
        // join tiles and update from the right
        //

        starneig_cuda_join_sub_window(0, m, i, MIN(n, i+max_width),
            stream, &packing_info, device_args, ldA, A, 0);

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            m, MIN(max_width, n-i), nb, m_one,
            Y, ldY, V+offset+i+nb-1, ldV, one, A, ldA);

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

extern "C" void starneig_hessenberg_cuda_update_right(
    void *buffers[], void *cl_args)
{
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
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
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

extern "C" void starneig_hessenberg_cuda_update_left(
    void *buffers[], void *cl_args)
{
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
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
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
