///
/// @file
///
/// @brief This file contains task definitions and task insertion function that
/// are shared among all components of the library.
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

#ifndef STARNEIG_COMMON_TASKS_H
#define STARNEIG_COMMON_TASKS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "matrix.h"
#include "vector.h"

///
/// @brief Inserts left_gemm_update task(s).
///
/// @param[in] rbegin
///         first row that belongs to the update window
///
/// @param[in] rend
///         last row that belongs to the update window + 1
///
/// @param[in] cbegin
///         first column that belongs to the update window
///
/// @param[in] cend
///         last column that belongs to the update window + 1
///
/// @param[in] splice
///         splice width
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] lQ_h
///         handle to the local Q matrix
///
/// @param[in,out] matrix
///          matrix descriptor
///
/// @param[in,out] mpi
///          MPI info
///
void starneig_insert_left_gemm_update(
    int rbegin, int rend, int cbegin, int cend, int splice, int prio,
    starpu_data_handle_t lq_h, starneig_matrix_descr_t matrix, mpi_info_t mpi);

///
/// @brief Inserts right_gemm_update task(s).
///
/// @param[in] rbegin
///         first row that belongs to the update window
///
/// @param[in] rend
///         last row that belongs to the update window + 1
///
/// @param[in] cbegin
///         first column that belongs to the update window
///
/// @param[in] cend
///         last column that belongs to the update window + 1
///
/// @param[in] splice
///         splice height
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] lQ_h
///         handle to the local Q matrix
///
/// @param[in,out] matrix
///         matrix descriptor
///
/// @param[in,out] mpi
///         MPI info
///
void starneig_insert_right_gemm_update(
    int rbegin, int rend, int cbegin, int cend, int splice, int prio,
    starpu_data_handle_t lq_h, starneig_matrix_descr_t matrix, mpi_info_t mpi);

///
/// @brief Inserts copy_matrix task(s).
///
/// @param[in] sr
///         first source matrix row to be copied
///
/// @param[in] sc
///         first source matrix column to be copied
///
/// @param[in] dr
///         first destination matrix row
///
/// @param[in] dc
///         first destination matrix column
///
/// @param[in] m
///         copy area height
///
/// @param[in] n
///         copy area width
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] source
///         source matrix
///
/// @param[in,out] dest
///         destination matrix
///
/// @param[in,out] mpi
///          MPI info
///
void starneig_insert_copy_matrix(
    int sr, int sc, int dr, int dc, int m, int n, int prio,
    starneig_matrix_descr_t source, starneig_matrix_descr_t dest,
    mpi_info_t mpi);

///
/// @brief Inserts copy_to_handle task.
///
/// @param[in] rbegin
///         The first row to be copied.
///
/// @param[in] rend
///         The last row the copied + 1.
///
/// @param[in] cbegin
///         The first column to be copied.
///
/// @param[in] cend
///         The last column the copied + 1.
///
/// @param[in] prio
///         The StarPU priority.
///
/// @param[in] source
///         The source matrix.
///
/// @param[in,out] dest
///         The destination data handle.
///
/// @param[in,out] mpi
///          The MPI info.
///
void starneig_insert_copy_matrix_to_handle(
    int rbegin, int rend, int cbegin, int cend, int prio,
    starneig_matrix_descr_t source, starpu_data_handle_t dest,
    mpi_info_t mpi);

///
/// @brief Inserts copy_from_handle task.
///
/// @param[in] rbegin
///         The first row to be copied.
///
/// @param[in] rend
///         The last row the copied + 1.
///
/// @param[in] cbegin
///         The first column to be copied.
///
/// @param[in] cend
///         The last column the copied + 1.
///
/// @param[in] prio
///         The StarPU priority.
///
/// @param[in] source
///         The source data handle.
///
/// @param[in,out] dest
///         The destination matrix.
///
/// @param[in,out] mpi
///          The MPI info.
///
void starneig_insert_copy_handle_to_matrix(
    int rbegin, int rend, int cbegin, int cend, int prio,
    starpu_data_handle_t source, starneig_matrix_descr_t dest,
    mpi_info_t mpi);

///
/// @brief Inserts set_to_identity task(s).
///
/// @param[in] prio
///         StarPU priority
///
/// @param[int,out] descr
///         matrix descriptor
///
/// @param[in,out] tag_offset
///         MPI info
///
void starneig_insert_set_to_identity(
    int prio, starneig_matrix_descr_t descr, mpi_info_t mpi);

///
/// @brief Inserts scan_diagonal task(s).
///
/// @param[in] begin
///         The first diagonal element to scan.
///
/// @param[in] end
///         The last diagonal element to scan + 1.
///
/// @param[in] mask_begin
///         The first scanning vector element to be used.
///
/// @param[in] up
///         The number of extra rows to include above `begin`.
///
/// @param[in] down
///         The number of extra rows to include below `end`-1.
///
/// @param[in] left
///         The number of extra rows to include left from `begin`.
///
/// @param[in] right
///         The number of extra rows to include right from `end`-1.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] func
///         Scanning function. The scanning function is expected to scan
///         diagonal elements that fall within a padded scanning window.
///          - return type: void
///          - arg  0: (int) number of diagonal entries to scan
///          - arg  1: (int) row offset for the first diagonal element
///          - arg  2: (int) column offset for the first diagonal element
///          - arg  3: (int) number of rows in the scanning window
///          - arg  4: (int) number of columns in the scanning window
///          - arg  5: (int) leading dimension of the matrix A
///          - arg  6: (int) leading dimension of the matrix B
///          - arg  7: (void const *) optional argument
///          - arg  8: (void const *) scanning window from the matrix A
///          - arg  9: (void const *) scanning window from the matrix B
///          - arg 10: (void **) pointers to scanning mask vectors
///
/// @param[in] arg
///         Optional argument for the scanning function.
///
/// @param[in] A
///         The matrix A descriptor.
///
/// @param[in] B
///         The matrix B descriptor.
///
// @param[in,out] mpi
///         MPI info
///
/// @param[in,out] ...
///         The scanning mask descriptors.
///
void starneig_insert_scan_diagonal(
    int begin, int end, int mask_begin,
    int up, int down, int left, int right, int prio,
    void (*func)(
        int, int, int, int, int, int, int, void const *, void const *,
        void const *, void **masks),
    void const *arg, starneig_matrix_descr_t A, starneig_matrix_descr_t B,
    mpi_info_t mpi, ...);

///
/// @brief Extracts eigenvalues from a (generalized) Schur form (A,B).
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] A
///         The matrix A descriptor.
///
/// @param[in] B
///         The matrix B descriptor.
///
/// @param[in,out] real
///         The real parts of the eigenvalues.
///
/// @param[in,out] imag
///         The imaginary parts of the eigenvalues.
///
/// @param[in,out] beta
///         The scaling factors of the eigenvalues.
///
/// @param[in,out] mpi
///         MPI info.
///
void starneig_insert_extract_eigenvalues(
    int prio,
    starneig_matrix_descr_t A, starneig_matrix_descr_t B,
    starneig_vector_descr_t real, starneig_vector_descr_t imag,
    starneig_vector_descr_t beta, mpi_info_t mpi);

///
/// @brief Initializes a matrix data handle with zeros.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in,out] handle
///         The matrix data handle.
///
void starneig_insert_set_to_zero(int prio, starpu_data_handle_t handle);

#endif
