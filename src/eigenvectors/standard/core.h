///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
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

#ifndef STARNEIG_EIGENVECTORS_STD_CORE_H
#define STARNEIG_EIGENVECTORS_STD_CORE_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/error.h>
#include "../../common/common.h"
#include <starpu.h>


///
/// @brief Inserts all tasks for computing eigenvectors of the Schur matrix S.
///
starneig_error_t starneig_eigvec_std_insert_backsolve_tasks(
    int num_tiles,
    starpu_data_handle_t **S_tiles,
    starpu_data_handle_t **S_tiles_norms,
    starpu_data_handle_t *lambda_tiles,
    starpu_data_handle_t *lambda_type_tiles,
    starpu_data_handle_t **X_tiles,
    starpu_data_handle_t **scales_tiles,
    starpu_data_handle_t **Xnorms_tiles,
    starpu_data_handle_t *selected_tiles,
    starpu_data_handle_t *selected_lambda_type_tiles,
    starpu_data_handle_t *info_tiles,
    double smlnum,
    int critical_prio, int update_prio);


///
/// @brief Inserts all tasks for backtransforming the eigenvectors.
///
starneig_error_t starneig_eigvec_std_insert_backtransform_tasks(
    int *first_row, int num_tiles,
    starpu_data_handle_t **Q_tiles,
    starpu_data_handle_t **X_tiles,
    starpu_data_handle_t **Y_tiles);

#endif
