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

#ifndef STARNEIG_REORDER_TASKS_H
#define STARNEIG_REORDER_TASKS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "window.h"
#include "../common/matrix.h"
#include "../common/vector.h"

///
/// @brief Inserts a reorder_window task.
///
/// @param[in] prio
///         StarPU priority
///
/// @param[in] small_window_size
///         small window size
///
/// @param[in] small_window_threshold
///         small window threshold
///
/// @param[in,out] window
///         window structure
///
/// @param[in] selected
///         eigenvalue selection vector
///
/// @param[in,out] matrix_a
///         A matrix
///
/// @param[in,out] matrix_b
///         B matrix
///
/// @param[in,out] tag_offset
///         MPI info
///
void starneig_reorder_insert_window(
    int prio, int small_window_size, int small_window_threshold,
    struct window *window, starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_a, starneig_matrix_descr_t matrix_b,
    mpi_info_t mpi);

#endif
