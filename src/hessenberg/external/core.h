///
/// @file This file contains the Hessenberg reduction task insertion function.
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

#ifndef STARNEIG_HESSENBERG_EXT_CORE_H
#define STARNEIG_HESSENBERG_EXT_CORE_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/error.h>
#include "../common/common.h"
#include "../common/matrix.h"

///
/// @brief Inserts all Hessenberg reduction related tasks.
///
/// @param[in] panel_width
///         Panel width.
///
/// @param[in] begin
///         First row/column to be reduced.
///
/// @param[in] end
///         Last row/column to be reduced + 1.
///
/// @param[in] parallel_ctx
///         Parallel StarPU scheduling context (for panel reduction and
///         trailing matrix update tasks). This scheduling context is ignored if
///         the value is STARPU_NMAX_SCHED_CTXS.
///
/// @param[in] other_ctx
///         Regular StarPU scheduling context (for remaining update tasks).
///
/// @param[in] critical_prio
///         Panel reduction and trailing matrix update task priority.
///
/// @param[in] update_prio
///         Update tasks priority.
///
/// @param[in] misc_prio
///         Miscellaneous task priority.
///
/// @param[in,out] matrix_q
///         Matrix Q.
///
/// @param[in,out] matrix_a
///         Matrix A.
///
/// @param[in,out] tag_offset
///         MPI info
///
starneig_error_t starneig_hessenberg_ext_insert_tasks(
    int panel_width, int begin, int end,
    unsigned parallel_ctx, unsigned other_ctx,
    int critical_prio, int update_prio, int misc_prio,
    starneig_matrix_descr_t matrix_q, starneig_matrix_descr_t matrix_a,
    mpi_info_t mpi);

#endif
