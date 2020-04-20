///
/// @file This file contains the Hessenberg reduction specific task definitions
/// and task insertion functions.
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

#ifndef STARNEIG_HESSENBERG_TASKS_H
#define STARNEIG_HESSENBERG_TASKS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "../common/matrix.h"
#include "../common/vector.h"

///
/// @brief Prepares a column for a reduction.
///
/// @param[in] prio
///         The StarPU priority.
///
/// @param[in] i
///         The index of the current column in the panel.
///
/// @param[in] begin
///         First row that belongs to the panel.
///
/// @param[in] end
///         Last row that belongs to the panel + 1.
///
/// @param[in] Y_h
///         A handle to the Y matrix (only when 0 < i).
///
/// @param[in,out] V_h
///         A handle to the V matrix.
///
/// @param[in,out] T_h
///         A handle to the T matrix.
///
/// @param[in,out] P_h
///         A handle to the panel matrix.
///
/// @param[out] v
///         An intemediate vector interface for the trailing matrix operation.
///
void starneig_hessenberg_insert_prepare_column(
    int prio, int i, int begin, int end,
    starpu_data_handle_t Y_h, starpu_data_handle_t V_h,
    starpu_data_handle_t T_h, starpu_data_handle_t P_h,
    starneig_vector_descr_t v);

///
/// @brief Performs the trailing matrix operation.
///
/// @param[in] prio
///         The StarPU priority.
///
/// @param[in] rbegin
///         First row that belongs to the trailing matrix.
///
/// @param[in] rend
///         Last row that belongs to the trailing matrix + 1.
///
/// @param[in] cbegin
///         First column that belongs to the trailing matrix.
///
/// @param[in] cend
///         Last column that belongs to the trailing matrix + 1.
///
/// @param[in] matrix_a
///         A pointer to the A matrix descriptor structure.
///
/// @param[in] v
///         An intemediate vector interface for the trailing matrix operation.
///
/// @param[out] y
///         An intemediate vector interface from the trailing matrix operation.
///
void starneig_hessenberg_insert_compute_column(
    int prio, int rbegin, int rend, int cbegin, int cend,
    starneig_matrix_descr_t matrix_a, starneig_vector_descr_t v,
    starneig_vector_descr_t y);

///
/// @brief Finalizes a column reduction.
///
/// @param[in] prio
///         The StarPU priority.
///
/// @param[in] i
///         The index of the current column in the panel.
///
/// @param[in] begin
///         First row that belongs to the panel.
///
/// @param[in] end
///         Last row that belongs to the panel + 1.
///
/// @param[in] V_h
///         A handle to the V matrix.
///
/// @param[in,out] T_h
///         A handle to the T matrix.
///
/// @param[in,out] Y_h
///         A handle to the Y matrix.
///
/// @param[in] y
///         An intemediate vector interface from the trailing matrix operation.
///
void starneig_hessenberg_insert_finish_column(
    int prio, int i, int begin, int end,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starpu_data_handle_t Y_h, starneig_vector_descr_t y);

///
/// @brief Inserts a update_trail task.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] rbegin
///         First row that belongs to the update window.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last column that belongs to the update window + 1.
///
/// @param[in] nb
///         Panel width.
///
/// @param[in] offset
///         First column of the trailing matrix that belogns to the block
///         column.
///
/// @param[in] V_h
///         Matrix V handle.
///
/// @param[in]  T_h
///         Matrix T handle.
///
/// @param[in]  Y_h
///         Matrix T handle.
///
/// @param[in,out] matrix_a
///         Pointer to the A matrix descriptor structure.
///
/// @param[in] parallel
///         A parallel task is inserted if this variable is non-zero.
///
/// @param[in,out] mpi
///         MPI info
///
void starneig_hessenberg_insert_update_trail(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    int offset, starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starpu_data_handle_t Y_h, starneig_matrix_descr_t matrix_a, mpi_info_t mpi);

///
/// @brief Inserts a update_right task.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] rbegin
///         First row that belongs to the update window.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last column that belongs to the update window + 1.
///
/// @param[in] nb
///         Panel width.
///
/// @param[in] V_h
///         Matrix V handle.
///
/// @param[in]  T_h
///         Matrix T handle.
///
/// @param[in,out] matrix_a
///         Pointer to the A matrix descriptor structure.
///
/// @param[in,out] mpi
///             MPI info
///
void starneig_hessenberg_insert_update_right(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_descr_t matrix_a, mpi_info_t mpi);

///
/// @brief Inserts a update_left task.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] rbegin
///         First row that belongs to the update window.
///
/// @param[in] rend
///         Last row that belongs to the update window + 1.
///
/// @param[in] cbegin
///         First column that belongs to the update window.
///
/// @param[in] cend
///         Last column that belongs to the update window + 1.
///
/// @param[in] nb
///         Panel width.
///
/// @param[in] V_h
///         Matrix V handle.
///
/// @param[in]  T_h
///         Matrix T handle.
///
/// @param[in,out] matrix_a
///         Pointer to the A matrix descriptor structure.
///
/// @param[in,out] mpi
///             MPI info
///
void starneig_hessenberg_insert_update_left(
    int prio, int rbegin, int rend, int cbegin, int cend, int nb,
    starpu_data_handle_t V_h, starpu_data_handle_t T_h,
    starneig_matrix_descr_t matrix_a, mpi_info_t mpi);

#endif
