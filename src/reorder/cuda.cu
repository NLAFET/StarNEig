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
#include "lapack.h"
#include "../common/common.h"
#include "../common/tiles.h"
#include "../common/math.h"

#include <math.h>
#include <starpu.h>
#include <starpu_cublas_v2.h>

static const double one = 1.0;
static const double zero = 0.0;

static __global__ void _init_local_q(int n, int ld, double *ptr)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    for(int j = idy; j < n; j += gridDim.y*blockDim.y)
        for(int i = idx; i < n; i += gridDim.x*blockDim.x)
            ptr[j*ld+i] = i == j ? 1.0 : 0.0;
}

static void init_local_q(cudaStream_t stream, int n, int ld, double *ptr)
{
    dim3 threads(32,32);
    dim3 blocks(MIN(5, divceil(n, threads.x)), MIN(5, divceil(n, threads.y)));
    _init_local_q<<<blocks, threads, 0, stream>>>(n, ld, ptr);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}

static void left_gemm_update(cudaStream_t stream, cublasHandle_t handle,
    int rbegin, int rend, int cbegin, int cend, int ldQ, int ldA, int ldT,
    double const *Q, double *A, double *T) {

    int m = rend-rbegin;
    int n = cend-cbegin;
    int k = rend-rbegin;

    if (m == 0 || n == 0)
        return;

    cudaError err = cudaMemcpy2DAsync(
        T, ldT*sizeof(double), A+cbegin*ldA+rbegin, ldA*sizeof(double),
        (rend-rbegin)*sizeof(double), cend-cbegin, cudaMemcpyDeviceToDevice,
        stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    cublasSetStream(handle, stream);
    cublasStatus_t cublas_err = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k, &one, Q, ldQ, T, ldT, &zero, A+cbegin*ldA+rbegin, ldA);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(cublas_err);
}

static void right_gemm_update(cudaStream_t stream, cublasHandle_t handle,
    int rbegin, int rend, int cbegin, int cend, int ldQ, int ldA, int ldT,
    double const *Q, double *A, double *T) {

    int m = rend-rbegin;
    int n = cend-cbegin;
    int k = cend-cbegin;

    if (m == 0 || n == 0)
        return;

    double one = 1.0;
    double zero = 0.0;

    cudaError err = cudaMemcpy2DAsync(
        T, ldT*sizeof(double), A+cbegin*ldA+rbegin, ldA*sizeof(double),
        (rend-rbegin)*sizeof(double), cend-cbegin, cudaMemcpyDeviceToDevice,
        stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    cublasSetStream(handle, stream);
    cublasStatus_t cublas_err = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &one, T, ldT, Q, ldQ, &zero, A+cbegin*ldA+rbegin, ldA);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(cublas_err);
}

static int reorder_window(cudaStream_t stream, cublasHandle_t handle,
    int window_size, int threshold, int n, int ldQ, int ldZ, int ldA, int ldB,
    int *select, double *Q, double *Z, double *A, double *B)
{
    int ret = 0;
    cudaError err;

    int *_select = NULL;
    double *_lA = NULL; size_t ld_lA = 0;
    double *_lB = NULL; size_t ld_lB = 0;
    double *_lQ = NULL; size_t ld_lQ = 0;
    double *_lZ = NULL; size_t ld_lZ = 0;
    double *_work = NULL;

    double *lQ = NULL; size_t ldlQ = 0;
    double *lZ = NULL; size_t ldlZ = 0;
    double *vT = NULL; size_t ldvT = 0;
    double *hT = NULL; size_t ldhT = 0;
    double *qT = NULL; size_t ldqT = 0;

    int streams_created = 0;
    cudaStream_t left, right, right_q;

    int begin = 0;
    int end = 0;

    // copy eigenvalue selection vector from device memory

    err = cudaHostAlloc(&_select, n*sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
    err = cudaMemcpyAsync(
        _select, select, n*sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    // The window may already contain "tainted" tiles but even in that
    // situation the window can be partially reordered. To be able to do it we
    // need to identify the last non-tainted selected tile:
    int term = 0;
    for (int i = 0; i < n; i++) {
        if (_select[i] == 2) {
            // make sure that tainted section is marked tainted
            for (int j = i; j < n; j++)
                _select[j] = 2;
            break;
        }
        if (_select[i] == 1)
            term = i+1;
    }

    // exit if nothing can be done
    if (term < 2) {
        ret = 1;
        goto cleanup;
    }

    // allocate work space for dtgsen/dtrsen
    if (B != NULL)
        _work = (double *) malloc((7*n+16)*sizeof(double));
    else
        _work = (double *) malloc(3*n*sizeof(double));

    // make sure that the window is big enough and call
    // *_starneig_reorder_window directly if it is not
    if (n < threshold) {

        if (B != NULL) {
            ld_lA = ld_lB = ld_lQ = ld_lZ = divceil(n, 8)*8;
            err = cudaHostAlloc(
                &_lA, 4*n*ld_lA*sizeof(double), cudaHostAllocDefault);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            _lB = _lA + n*ld_lA;
            _lQ = _lB + n*ld_lB;
            _lZ = _lQ + n*ld_lQ;
        }
        else {
            ld_lA = ld_lQ = divceil(n, 8)*8;
            err = cudaHostAlloc(
                &_lA, 2*n*ld_lA*sizeof(double), cudaHostAllocDefault);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            _lQ = _lA + n*ld_lA;
        }

        // copy A matrix

        err = cudaMemcpy2DAsync(
            _lA, ld_lA*sizeof(double), A, ldA*sizeof(double),
            n*sizeof(double), n, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        // copy Q matrix

        err = cudaMemcpy2DAsync(
            _lQ, ld_lQ*sizeof(double), Q, ldQ*sizeof(double),
            n*sizeof(double), n, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        if (B != NULL) {

            // copy B matrix

            err = cudaMemcpy2DAsync(
                _lB, ld_lB*sizeof(double), B, ldB*sizeof(double),
                n*sizeof(double), n, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            // copy Z matrix

            err = cudaMemcpy2DAsync(
                _lZ, ld_lZ*sizeof(double), Z, ldZ*sizeof(double),
                n*sizeof(double), n, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        // reorder

        int m;
        if (B != NULL)
            ret = starneig_dtgsen(0, term, ld_lQ, ld_lZ, ld_lA, ld_lB, &m,
                _select, _lQ, _lZ, _lA, _lB, _work);
        else
            ret = starneig_dtrsen(
                0, term, ld_lQ, ld_lA, &m, _select, _lQ, _lA, _work);

        // store A matrix

        err = cudaMemcpy2DAsync(
            A, ldA*sizeof(double), _lA, ld_lA*sizeof(double),
            n*sizeof(double), n, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        // store Q matrix

        err = cudaMemcpy2DAsync(
            Q, ldQ*sizeof(double), _lQ, ld_lQ*sizeof(double),
            n*sizeof(double), n, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        if (B != NULL) {

            // store B matrix

            err = cudaMemcpy2DAsync(
                B, ldB*sizeof(double), _lB, ld_lB*sizeof(double),
                n*sizeof(double), n, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            // store Z matrix

            err = cudaMemcpy2DAsync(
                Z, ldZ*sizeof(double), _lZ, ld_lZ*sizeof(double),
                n*sizeof(double), n, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);
        }

        // if an error occurred, mark the whole window tainted
        if (ret != 0)
            for (int i = 0; i < n; i++)
                _select[i] = 2;

        goto cleanup;
    }

    // allocate host workspace

    if (B != NULL) {
        ld_lA = ld_lB = ld_lQ = ld_lZ = divceil(window_size, 8)*8;
        err = cudaHostAlloc(
            &_lA, 4*window_size*ld_lA*sizeof(double), cudaHostAllocDefault);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        _lB = _lA + window_size*ld_lA;
        _lQ = _lB + window_size*ld_lB;
        _lZ = _lQ + window_size*ld_lQ;
    }
    else {
        ld_lA = ld_lQ = divceil(window_size, 8)*8;
        err = cudaHostAlloc(
            &_lA, 2*window_size*ld_lA*sizeof(double), cudaHostAllocDefault);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        _lQ = _lA + window_size*ld_lA;
    }

    // device side local transformation matrices

    if (B != NULL) {
        err = cudaMallocPitch(
            &lQ, &ldlQ, window_size*sizeof(double), 2*window_size);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);
        ldlQ /= sizeof(double);

        lZ = lQ + window_size*ldlQ;
        ldlZ = ldlQ;
    }
    else {
        err = cudaMallocPitch(
            &lQ, &ldlQ, window_size*sizeof(double), window_size);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);
        ldlQ /= sizeof(double);

        lZ = lQ;
        ldlZ = ldlQ;
    }

    // device side scratch buffers for GEMM kernels

    err = cudaMallocPitch(&hT, &ldhT, window_size*sizeof(double), n);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
    ldhT /= sizeof(double);

    err = cudaMallocPitch(&vT, &ldvT, n*sizeof(double), 2*window_size);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
    ldvT /= sizeof(double);

    qT = vT + window_size*ldvT;
    ldqT = ldvT;

    // GEMM kernel streams

    cudaStreamCreate(&left);
    cudaStreamCreate(&right);
    cudaStreamCreate(&right_q);
    streams_created = 1;

    // repeat until all chains have been processed
    while (1) {

        // place the window chain
        int in_chain = 0;
        for (int i = end; in_chain < window_size/2 && i < term; i++) {
            if (_select[i]) {
                in_chain++;
                end = i+1;
            }
        }

        // quit if the chain is empty
        if (in_chain == 0)
            goto cleanup;

        // place the first window
        int first = 1;
        int wend = MIN(term, end+1);
        int wbegin = MAX(begin, wend-window_size);

        cudaEvent_t left_ready;
        cudaEventCreate(&left_ready);
        cudaEventRecord(left_ready, stream);

        cudaEvent_t right_ready;
        cudaEventCreate(&right_ready);
        cudaEventRecord(right_ready, stream);

        cudaEvent_t right_q_ready;
        cudaEventCreate(&right_q_ready);
        cudaEventRecord(right_q_ready, stream);

        // repeat until all windows in the current chain have been processed
        int in_window = 0;
        while(1) {

            // calculate window size
            int wsize = wend-wbegin;

            // the main stream should wait until all right-hand side updates
            // have finished

            cudaStreamWaitEvent(stream, right_ready, 0);
            cudaEventDestroy(right_ready);

            // copy padded window from the matrix A

            err = cudaMemcpy2DAsync(_lA, ld_lA*sizeof(double),
                A+(size_t)wbegin*ldA+wbegin, ldA*sizeof(double),
                wsize*sizeof(double), wsize, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            // resize window if necessary

            double *__lA = _lA;

            if (wbegin != begin && _lA[1] == 0.0) {
                wbegin++;
                __lA = _lA+ld_lA+1;
            }

            if (first && wend < term && _lA[(wsize-2)*ld_lA+wsize-1] == 0.0)
                wend--;

            wsize = wend-wbegin;

            // copy window from the matrix B

            if (B != NULL) {
                err = cudaMemcpy2DAsync(_lB, ld_lB*sizeof(double),
                    B+(size_t)wbegin*ldB+wbegin, ldB*sizeof(double),
                    wsize*sizeof(double), wsize, cudaMemcpyDeviceToHost,
                    stream);
                if (err != cudaSuccess)
                    STARPU_CUDA_REPORT_ERROR(err);
            }

            // reorder the window

            if (B != NULL) {
                starneig_init_local_q(wsize, ld_lQ, _lQ);
                starneig_init_local_q(wsize, ld_lZ, _lZ);

                err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess)
                    STARPU_CUDA_REPORT_ERROR(err);

                ret = starneig_dtgsen(0, wsize, ld_lQ, ld_lZ, ld_lA, ld_lB,
                    &in_window, _select+wbegin, _lQ, _lZ, __lA, _lB, _work);
            }
            else {
                starneig_init_local_q(wsize, ld_lQ, _lQ);
                ret = starneig_dtrsen(0, wsize, ld_lQ, ld_lA,
                    &in_window, _select+wbegin, _lQ, __lA, _work);
            }

            // store window

            err = cudaMemcpy2DAsync(
                A+(size_t)wbegin*ldA+wbegin, ldA*sizeof(double),
                __lA, ld_lA*sizeof(double),
                wsize*sizeof(double), wsize, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            if (B != NULL) {
                err = cudaMemcpy2DAsync(
                    B+(size_t)wbegin*ldB+wbegin, ldB*sizeof(double),
                    _lB, ld_lB*sizeof(double),
                    wsize*sizeof(double), wsize, cudaMemcpyHostToDevice,
                    stream);
                if (err != cudaSuccess)
                    STARPU_CUDA_REPORT_ERROR(err);
            }

            // the main stream should wait until all left-hand side updates
            // and Q/Z matrix updates have finished

            cudaStreamWaitEvent(stream, left_ready, 0);
            cudaEventDestroy(left_ready);

            cudaStreamWaitEvent(stream, right_q_ready, 0);
            cudaEventDestroy(right_q_ready);

            // move transformation matrices to device memory

            err = cudaMemcpy2DAsync(
                lQ, ldlQ*sizeof(double), _lQ, ld_lQ*sizeof(double),
                wsize*sizeof(double), wsize, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
                STARPU_CUDA_REPORT_ERROR(err);

            if (B != NULL) {
                err = cudaMemcpy2DAsync(
                    lZ, ldlZ*sizeof(double), _lZ, ld_lZ*sizeof(double),
                    wsize*sizeof(double), wsize, cudaMemcpyHostToDevice,
                    stream);
                if (err != cudaSuccess)
                    STARPU_CUDA_REPORT_ERROR(err);
            }

            cudaEvent_t window_ready;
            cudaEventCreate(&window_ready);
            cudaEventRecord(window_ready, stream);
            cudaStreamWaitEvent(left, window_ready, 0);
            cudaStreamWaitEvent(right, window_ready, 0);
            cudaStreamWaitEvent(right_q, window_ready, 0);
            cudaEventDestroy(window_ready);

            // apply the local transformation matrices lQ and lZ to Q and Z
            if (Q != NULL)
                right_gemm_update(right_q, handle,
                    0, MIN(term, end+1), wbegin, wend, ldlQ, ldQ, ldqT,
                    lQ, Q, qT);
            if (Z != NULL)
                right_gemm_update(right_q, handle,
                    0, MIN(term, end+1), wbegin, wend, ldlZ, ldZ, ldqT,
                    lZ, Z, qT);

            // apply the local transformation matrices lQ and lZ to A
            right_gemm_update(right, handle,
                0, wbegin, wbegin, wend, ldlZ, ldA, ldvT, lZ, A, vT);
            left_gemm_update(left, handle,
                wbegin, wend, wend, n, ldlQ, ldA, ldhT, lQ, A, hT);

            // apply the local transformation matrices lQ and lZ to Z
            if (B != NULL) {
                right_gemm_update(right, handle,
                    0, wbegin, wbegin, wend, ldlZ, ldB, ldvT, lZ, B, vT);
                left_gemm_update(left, handle,
                    wbegin, wend, wend, n, ldlQ, ldB, ldhT, lQ, B, hT);
            }

            // if an error occurred, mark the current window and everything
            // below it tainted
            if (ret != 0) {
                for (int i = wbegin; i < n; i++)
                    _select[i] = 2;
                goto cleanup;
            }

            cudaEventCreate(&left_ready);
            cudaEventRecord(left_ready, left);

            // quit if this was the topmost window in the chain
            if (wbegin == begin)
                break;

            cudaEventCreate(&right_ready);
            cudaEventRecord(right_ready, right);

            cudaEventCreate(&right_q_ready);
            cudaEventRecord(right_q_ready, right_q);

            // place the next window such that it does not split any 2-by-2
            // tiles
            first = 0;
            wend = MIN(term, wbegin + in_window);
            wbegin = MAX(begin, wend-window_size);
        }

        // the main stream should wait until all left-hand side updates
        // from the previous window chain have finished

        cudaStreamWaitEvent(stream, left_ready, 0);
        cudaEventDestroy(left_ready);

        // advance downwards
        begin += in_window;
    }

cleanup:

    err = cudaMemcpyAsync(
        select, _select, n*sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    if (streams_created) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        err = cudaStreamSynchronize(left);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        err = cudaStreamSynchronize(right);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        err = cudaStreamSynchronize(right_q);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);

        cudaStreamDestroy(left);
        cudaStreamDestroy(right);
        cudaStreamDestroy(right_q);
        cublasSetStream(handle, stream);
    }

    cudaFreeHost(_select);
    cudaFreeHost(_lA);
    free(_work);

    cudaFree(lQ);
    cudaFree(vT);
    cudaFree(hT);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_cuda_reorder_window(void *buffers[], void *cl_arg)
{
    struct packing_info packing_info_A, packing_info_B;
    struct range_packing_info packing_info_selected;
    int window_size, threshold, swaps;
    starpu_codelet_unpack_args(cl_arg,
        &packing_info_selected, &packing_info_A, &packing_info_B,
        &window_size, &threshold, &swaps);

    cudaError err;
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    int size = packing_info_A.rend - packing_info_A.rbegin;
    int general = packing_info_B.handles != 0;

    int k = 0;

    // local matrix Q
    struct starpu_matrix_interface *lQ_i =
        (struct starpu_matrix_interface *)buffers[k++];
    double *lQ_ptr = (double*) STARPU_MATRIX_GET_PTR(lQ_i);
    int lQ_ld = STARPU_MATRIX_GET_LD(lQ_i);
    init_local_q(stream, size, lQ_ld, lQ_ptr);

    // local matrix Z
    double *lZ_ptr = NULL;
    int lZ_ld = 0;
    if (general) {
        struct starpu_matrix_interface *lZ_i =
            (struct starpu_matrix_interface *)buffers[k++];
        lZ_ptr = (double*) STARPU_MATRIX_GET_PTR(lZ_i);
        lZ_ld = STARPU_MATRIX_GET_LD(lZ_i);
        init_local_q(stream, size, lZ_ld, lZ_ptr);
    }

    // local matrix A
    struct starpu_matrix_interface *lA_i =
        (struct starpu_matrix_interface *)buffers[k++];
    double *lA_ptr = (double*) STARPU_MATRIX_GET_PTR(lA_i);
    int lA_ld = STARPU_MATRIX_GET_LD(lA_i);

    // local matrix B
    double *lB_ptr = NULL;
    int lB_ld = 0;
    if (general) {
        struct starpu_matrix_interface *lB_i =
            (struct starpu_matrix_interface *)buffers[k++];
        lB_ptr = (double*) STARPU_MATRIX_GET_PTR(lB_i);
        lB_ld = STARPU_MATRIX_GET_LD(lB_i);
    }

    // eigenvalue selection bitmap

    int *selected;
    err = cudaMalloc(&selected, size*sizeof(int));
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    struct starpu_vector_interface **select_i =
        (struct starpu_vector_interface **)buffers + k;
    k += packing_info_selected.handles;

    uintptr_t* selected_ds = starneig_cuda_prepare_join_range(
        &packing_info_selected, (void **)select_i);
    starneig_cuda_join_range(
        stream, &packing_info_selected, selected_ds, selected, 0);

    // corresponding tiles from the matrix A

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + k;
    k += packing_info_A.handles;

    struct tile_addr *A_ds =
        starneig_cuda_prepare_join_window(&packing_info_A, (void **)A_i);
    starneig_cuda_join_diag_window(
        stream, &packing_info_A, A_ds, lA_ld, lA_ptr, 0);

    // corresponding tiles from the matrix B

    struct tile_addr *B_ds = NULL;
    if (general) {
        struct starpu_matrix_interface **B_i =
            (struct starpu_matrix_interface **)buffers + k;
        k += packing_info_B.handles;

        B_ds =
            starneig_cuda_prepare_join_window(&packing_info_B, (void **)B_i);
        starneig_cuda_join_diag_window(
            stream, &packing_info_B, B_ds, lB_ld, lB_ptr, 0);
    }

    // reorder
    reorder_window(stream, handle,
        window_size, threshold, size, lQ_ld, lZ_ld, lA_ld, lB_ld,
        selected, lQ_ptr, lZ_ptr, lA_ptr, lB_ptr);

    // store result

    starneig_cuda_join_range(
        stream, &packing_info_selected, selected_ds, selected, 1);

    starneig_cuda_join_diag_window(
        stream, &packing_info_A, A_ds, lA_ld, lA_ptr, 1);

    if (general)
        starneig_cuda_join_diag_window(
            stream, &packing_info_B, B_ds, lB_ld, lB_ptr, 1);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);

    cudaFree(selected);
}
