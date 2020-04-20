///
/// @file
///
/// @brief This file contains the high level components of the StarPU-based QR
/// algorithm.
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

#ifndef STARNEIG_SCHUR_CORE_H
#define STARNEIG_SCHUR_CORE_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "../common/common.h"
#include "../common/matrix.h"
#include "../common/vector.h"
#include <starneig/expert.h>
#include <starneig/error.h>

///
/// @brief Inserts all Schur reduction related tasks.
///
/// @param[in] conf
///         configuration structure
///
/// @param[in,out] matrix_q
///         matrix Q descriptor
///
/// @param[in,out] matrix_z
///         matrix Z descriptor
///
/// @param[in,out] matrix_a
///         matrix A descriptor
///
/// @param[in,out] matrix_b
///         matrix B descriptor
///
/// @param[out] eigen_real
///         eigenvalues (real parts)
///
/// @param[out] eigen_imag
///         eigenvalues (imaginary parts)
///
/// @param[out] eigen_beta
///         eigenvalues (beta)
///
/// @param[in,out] tag_offset
///         MPI info
///
/// @return error code
///
starneig_error_t starneig_schur_insert_tasks(
    struct starneig_schur_conf const *conf,
    starneig_matrix_descr_t matrix_q,
    starneig_matrix_descr_t matrix_z,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    starneig_vector_descr_t eigen_real,
    starneig_vector_descr_t eigen_imag,
    starneig_vector_descr_t eigen_beta,
    mpi_info_t mpi);

#endif
