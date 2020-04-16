///
/// @file
///
/// @brief This file contains the CUDA implementations of codelets that are
/// shared among all components of the library.
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
#include "common.h"
#include "tiles.h"
#include <starpu.h>
#include <starpu_cublas_v2.h>

static const double *one = (const double[]) { 1.0 };
static const double *zero = (const double[]) { 0.0 };

extern "C" void starneig_cuda_left_gemm_update(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    // local Q matrix
    double *lq_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int lq_ld = STARPU_MATRIX_GET_LD(buffers[0]);

    // scratch buffers
    double *st1_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *st2_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    int st1_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    int st2_ld = STARPU_MATRIX_GET_LD(buffers[2]);

    struct tile_addr *device_args =
        starneig_cuda_prepare_join_window(&packing_info, buffers+3);

    // prepare for kernels
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    //
    // st1 <- X
    //

    starneig_cuda_join_window(
        stream, &packing_info, device_args, st1_ld, st1_ptr, 0);

    //
    // st2 <- Q^T * st1
    //

    int n = packing_info.rend-packing_info.rbegin;
    int m = packing_info.cend-packing_info.cbegin;
    int k = packing_info.rend-packing_info.rbegin;

    cublasStatus_t cublas_err = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, one, lq_ptr, lq_ld, st1_ptr, st1_ld, zero, st2_ptr, st2_ld);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(cublas_err);

    //
    // Y <- st2
    //

    starneig_cuda_join_window(
        stream, &packing_info, device_args, st2_ld, st2_ptr, 1);
}

extern "C" void starneig_cuda_right_gemm_update(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    starpu_codelet_unpack_args(cl_args, &packing_info);

    // local Q matrix
    double *lq_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int lq_ld = STARPU_MATRIX_GET_LD(buffers[0]);

    // scratch buffers
    double *st1_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *st2_ptr = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    int st1_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    int st2_ld = STARPU_MATRIX_GET_LD(buffers[2]);

    struct tile_addr *device_args =
        starneig_cuda_prepare_join_window(&packing_info, buffers+3);

    // prepare for kernels
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    //
    // st1 <- X
    //

    starneig_cuda_join_window(
        stream, &packing_info, device_args, st1_ld, st1_ptr, 0);

    //
    // st2 <- st1 * Q
    //

    int n = packing_info.rend-packing_info.rbegin;
    int m = packing_info.cend-packing_info.cbegin;
    int k = packing_info.cend-packing_info.cbegin;

    cublasStatus_t cublas_err = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, one, st1_ptr, st1_ld, lq_ptr, lq_ld, zero, st2_ptr, st2_ld);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(cublas_err);

    //
    // Y <- st2
    //

    starneig_cuda_join_window(
        stream, &packing_info, device_args, st2_ld, st2_ptr, 1);
}

void starneig_cuda_set_vector_to_zero(void *buffers[], void *cl_args)
{
    void *A = (void *) STARPU_VECTOR_GET_PTR(buffers[0]);
    int m = STARPU_VECTOR_GET_NX(buffers[0]);
    size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cudaError_t ret = cudaMemsetAsync(A, 0, m*elemsize, stream);
    if (ret != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(ret);
}

void starneig_cuda_add_vectors(void *buffers[], void *cl_args)
{
    double *Y = (double *) STARPU_VECTOR_GET_PTR(buffers[0]);
    double *X = (double *) STARPU_VECTOR_GET_PTR(buffers[1]);
    int m = STARPU_VECTOR_GET_NX(buffers[1]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    cublasStatus_t cublas_err = cublasDaxpy(handle, m, one, X, 1, Y, 1);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(cublas_err);
}

void starneig_cuda_set_matrix_to_zero(void *buffers[], void *cl_args)
{
    void *A = (void *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int m = STARPU_MATRIX_GET_NX(buffers[0]);
    int n = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t ldA = STARPU_MATRIX_GET_LD(buffers[0]);
    size_t elemsize = STARPU_MATRIX_GET_ELEMSIZE(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cudaError_t ret = cudaMemset2DAsync(
        A, ldA*elemsize, 0, m*elemsize, n, stream);
    if (ret != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(ret);
}

void starneig_cuda_add_matrices(void *buffers[], void *cl_args)
{
    double *Y = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    size_t ldY = STARPU_MATRIX_GET_LD(buffers[0]);

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    size_t ldX = STARPU_MATRIX_GET_LD(buffers[1]);

    int m = STARPU_MATRIX_GET_NX(buffers[1]);
    int n = STARPU_MATRIX_GET_NY(buffers[1]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    cublasStatus_t cublas_err = cublasDgeam(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, m, n, one, Y, ldY, one, X, ldX, Y, ldY);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(cublas_err);
}
