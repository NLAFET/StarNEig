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

#ifndef STARNEIG_REORDER_CORE_H
#define STARNEIG_REORDER_CORE_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "../common/common.h"
#include "../common/vector.h"
#include "../common/matrix.h"
#include <starneig/expert.h>
#include <starneig/error.h>

///
/// @brief Returns "optimal" tile size for given problem size
///
/// @param[in] n
///         The matrix dimension.
///
/// @param[in] select_ratio
///         The eigenvalue selection ratio.
///
/// @return The optimal tile size.
///
int starneig_reorder_get_optimal_tile_size(int n, double select_ratio);

///
/// @brief Inserts all reordering related tasks
///
/// @param[in] conf
///         The configuration structure.
///
/// @param[in,out] selected
///         The eigenvalue selection vector.
///
/// @param[in,out] Q
///         The orthogonal matrix Q.
///
/// @param[in,out] Z
///         The orthogonal matrix Z.
///
/// @param[in,out] A
///         The Schur matrix A.
///
/// @param[in,out] B
///         The upper triangular matrix B.
///
/// @param[in,out] mpi
///         MPI info.
///
starneig_error_t starneig_reorder_insert_tasks(
    struct starneig_reorder_conf const *conf,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t Q, starneig_matrix_descr_t Z,
    starneig_matrix_descr_t A, starneig_matrix_descr_t B,
    starneig_vector_descr_t real, starneig_vector_descr_t imag,
    starneig_vector_descr_t beta,
    mpi_info_t mpi);

#endif
