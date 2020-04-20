///
/// @file
///
/// @brief This file contains task definitions and related task insertion
/// functions that are used in the StarPU-bases QR algorithm.
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

#ifndef STARNEIG_SCHUR_TASKS_H
#define STARNEIG_SCHUR_TASKS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include "../common/matrix.h"
#include "../common/vector.h"

///
/// @brief Inserts a push_inf_top task.
///
/// @see push_inf_top_cl
///
/// @param[in] begin
///         First row that belongs to the computation window.
///
/// @param[in] end
///         Last row that belongs to the computation window + 1.
///
/// @param[in] top
///         If non-zero, then it is assumed that the window is located to the
///         top left corner of the segment and  the infinite eigenvalues are
///         deflated from the top of the window.
///
/// @param[in] bottom
///         If non-zero, then it is assumed that the window is located to the
///         bottom right corner of the segment.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] thres_inf
///         Those entries diagonal of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in,out] matrix_a
///         Matrix A descriptor.
///
/// @param[in,out] matrix_b
///         Matrix B descriptor.
///
/// @param[out] lQ_h
///         Returns a handle to the local left-hand size transformation matrix.
///
/// @param[out] lZ_h
///         Returns a handle to the local right-hand size transformation matrix.
///
/// @param[in,out] tag_offset
///         MPI info.
///
void starneig_schur_insert_push_inf_top(
    int begin, int end, int top, int bottom, int prio, double thres_inf,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *lQ_h, starpu_data_handle_t *lZ_h, mpi_info_t mpi);

///
/// @brief Inserts a push_bulges task.
///
/// @see push_bulges_cl
///
/// @param[in] begin
///         First row that belongs to the computation window.
///
/// @param[in] end
///         Last row that belongs to the computation window + 1.
///
/// @param[in] shifts_begin
///         First shift to be applied.
///
/// @param[in] shifts_end
///         Last shift to be applied + 1.
///
/// @param[in] mode
///         Bulge chasing mode.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] thres_a
///         Those entries of the matrix A that are smaller in magnitudes than
///         this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] shifts_real
///         Shift vector descriptor (real parts).
///
/// @param[in] shifts_imag
///         Shift vector descriptor (imaginary parts).
///
/// @param[in,out] matrix_a
///         Matrix A descriptor.
///
/// @param[in,out] matrix_b
///         Matrix B descriptor.
///
/// @param[out] lQ_h
///         Returns a handle to the local left-hand size transformation matrix.
///
/// @param[out] lZ_h
///         Returns a handle to the local right-hand size transformation matrix.
///
/// @param[in,out] tag_offset
///         MPI info.
///
void starneig_schur_insert_push_bulges(
    int begin, int end, int shifts_begin, int shifts_end,
    bulge_chasing_mode_t mode, int prio,
    double thres_a, double thres_b, double thres_inf,
    starneig_vector_descr_t shifts_real, starneig_vector_descr_t shifts_imag,
    starneig_vector_descr_t aftermath,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *lQ_h, starpu_data_handle_t *lZ_h, mpi_info_t mpi);

///
/// @brief Predicts the execution time of a aggressively_deflate task.
///
/// @param[in] generalized
///         Non-zero if the problem is generalized.
///
/// @param[in] window_size
///         Aggressive early deflation window size.
///
/// @return Expected task execution time in micro-seconds.
///
double starneig_predict_aggressively_deflate(int generalized, int window_size);

///
/// @brief Inserts an aggressively_deflate task.
///
/// @see aggressively_deflate_cl
///
/// @param[in] begin
///         First row that belongs to the computation window. The actual AED
///         window begins from 'begin'+1.
///
/// @param[in] end
///         Last row that belongs to the computation window + 1.
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] thres_a
///         Those entries of the matrix A that are smaller in magnitudes than
///         this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in,out] matrix_a
///         Matrix A descriptor.
///
/// @param[in,out] matrix_b
///         Matrix B descriptor.
///
/// @param[out] shifts_real
///         Shift vector descriptor (real parts).
///
/// @param[out] shifts_imag
///         Shift vector descriptor (imaginary parts).
///
/// @param[out] status_h
///         Returns a handle to a status tracking structure.
///
/// @param[out] lQ_h
///         Returns a handle to the local left-hand size transformation matrix.
///
/// @param[out] lZ_h
///         Returns a handle to the local right-hand size transformation matrix.
///
/// @param[in,out] tag_offset
///         MPI info.
///
void starneig_schur_insert_aggressively_deflate(
    int begin, int end, int prio,
    double thres_a, double thres_b, double thres_inf,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starneig_vector_descr_t shifts_real, starneig_vector_descr_t shifts_imag,
    starpu_data_handle_t *status_h, starpu_data_handle_t *lQ_h,
    starpu_data_handle_t *lZ_h, mpi_info_t mpi);

///
/// @brief Inserts a small_schur task.
///
/// @see small_schur_cl
///
/// @param[in] begin
///         First row that belongs to the computation window.
///
/// @param[in] end
///         Last row that belongs to the computation window + 1.
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] thres_a
///         Those entries of the matrix A that are smaller in magnitudes than
///         this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in,out] matrix_a
///         Matrix A descriptor.
///
/// @param[in,out] matrix_b
///         Matrix B descriptor.
///
/// @param[out] status_h
///         Returns a handle to a status tracking structure.
///
/// @param[out] lQ_h
///         Returns a handle to the local left-hand size transformation matrix.
///
/// @param[out] lZ_h
///         Returns a handle to the local right-hand size transformation matrix.
///
/// @param[in,out] tag_offset
///         MPI info.
///
void starneig_schur_insert_small_schur(
    int begin, int end, int prio,
    double thres_a, double thres_b, double thres_inf,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *status_h, starpu_data_handle_t *lQ_h,
    starpu_data_handle_t *lZ_h, mpi_info_t mpi);

///
/// @brief Inserts a small_hessenberg task.
///
/// @see small_hessenberg_cl
///
/// @param[in] begin
///         First row that belongs to the computation window.
///
/// @param[in] end
///         Last row that belongs to the computation window + 1.
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in,out] matrix_a
///         Matrix A descriptor.
///
/// @param[in,out] matrix_b
///         Matrix B descriptor.
///
/// @param[out] lQ_h
///         Returns a handle to the local left-hand size transformation matrix.
///
/// @param[out] lZ_h
///         Returns a handle to the local right-hand size transformation matrix.
///
/// @param[in,out] tag_offset
///         MPI info.
///
void starneig_schur_insert_small_hessenberg(
    int begin, int end, int prio, starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b, starpu_data_handle_t *lQ_h,
    starpu_data_handle_t *lZ_h, mpi_info_t mpi);

///
/// @brief Inserts a form_spike task.
///
/// @see form_spike_cl
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] matrix_q
///         Padded left-hand side AED transformation matrix
///
/// @param[out] base
///         Returns the spike base (the first row from the left-hand side AED
///         transformation matrix).
///
void starneig_schur_insert_form_spike(
    int prio, starneig_matrix_descr_t matrix_q, starneig_vector_descr_t *base);

///
/// @brief Inserts an embed_spike task.
///
/// @see embed_spike_cl
///
/// @param[in] end
///         Last row of the spike to be embedded + 1.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] base
///         Spike base (the first row from the left-hand size AED transformation
///         matrix).
///
/// @param[in,out] matrix_a
///          Padded AED window from the matrix A.
///
void starneig_schur_insert_embed_spike(
    int end, int prio, starneig_vector_descr_t base,
    starneig_matrix_descr_t matrix_a);

///
/// @brief Inserts a deflate task.
///
/// @see deflate_cl
///
/// @param[in] begin
///         First row that belongs to the computation window.
///
/// @param[in] end
///         Last row that belongs to the computation window + 1.
///
/// @param[in] deflate
///         If == 0, then the deflation checks are skipped and only reordering
///         is performed.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] thres_a
///         Those entries of the matrix A that are smaller in magnitudes than
///         this threshold may be set to zero.
///
/// @param[in] inducer_h
///         Spike inducer (the sub-diagonal entry to the left of the AED
///         window).
///
/// @param[in,out] status_h
///         Status tracking structure.
///
/// @param[in,out] base
///         Spike base (the first row from the left-hand side AED transformation
///         matrix).
///
/// @param[in,out] matrix_a
///         Padded AED window from the matrix A.
///
/// @param[in,out] matrix_b
///         padded AED window from the matrix B.
///
/// @param[out] lQ_h
///         Returns a handle to the local left-hand size transformation matrix.
///
/// @param[out] lZ_h
///         Returns a handle to the local right-hand size transformation matrix.
///
void starneig_schur_insert_deflate(
    int begin, int end, int deflate, int prio,
    double thres_a, starpu_data_handle_t inducer_h,
    starpu_data_handle_t status_h, starneig_vector_descr_t base,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    starpu_data_handle_t *lQ_h, starpu_data_handle_t *lZ_h);

///
/// @brief Inserts extract_shifts tasks(s).
///
/// @see extract_shifts_cl
///
/// @param[in] begin
///         First diagonal block / shift to be extracted.
///
/// @param[in] end
///         Last diagonal block / shift to be extracted + 1.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] matrix_a
///         Matrix A descriptor.
///
/// @param[in] matrix_b
///         Matrix B descriptor.
///
/// @param[in,out] real
///         Shift vector (real parts).
///
/// @param[in,out] imag
///         Shift vector (imaginary parts).
///
/// @param[in,out] mpi
///             MPI info.
///
void starneig_schur_insert_extract_shifts(
    int begin, int end, int prio, starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b, starneig_vector_descr_t real,
    starneig_vector_descr_t imag, mpi_info_t mpi);

///
/// @brief Inserts tasks that compute the Frobenius norm of a matrix.
///
/// @param[in] prio
///         StarPU priority.
///
/// @param[in] matrix
///         Matrix descriptor.
///
/// @param[in,out] mpi
///             MPI info.
///
/// @return  The Frobenius norm of the matrix.
///
starpu_data_handle_t starneig_schur_insert_compute_norm(
    int prio, starneig_matrix_descr_t matrix, mpi_info_t mpi);

#endif
