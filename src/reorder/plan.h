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

#ifndef STARNEIG_REORDER_PLAN_H
#define STARNEIG_REORDER_PLAN_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "window.h"

///
/// @brief Eigenvalue reordering plan.
///
struct plan {
    struct chain_list *begin; ///< first chain list
    struct chain_list *end;   ///< last chain list
    int longest_eff_length;   ///< longest chain (effective length)
    int total_length;         ///< combined length all chains
};

///
/// @brief Unregisters StarPU resources that are linked to a given eigenvalue
/// reordering plan.
///
/// @param[in,out] plan
///         eigenvalue reordering plan
///
void starneig_unregister_plan(struct plan *plan);

///
/// @brief Frees a previously allocated eigenvalue reordering plan and
/// unregisters the related StarPU resources.
///
/// @param[in,out] plan
///         eigenvalue reordering plan
///
void starneig_free_plan(struct plan *plan);

///
/// @brief Interface for a plan generation function.
///
/// @param[in] n
///         problem dimension
///
/// @param[in] window_size
///         window size (-1 => automatic)
///
/// @param[in] values_per_chain
///         number of selected eigenvalues per window chain (-1 => automatic)
///
/// @param[in] tile_size
///         tile size
///
/// @param[in] selected
///         eigenvalue selection bitmap
///
/// @param[in] complex_distr
///         complex eigenvalue distribution bitmap
///
/// @return eigenvalue reordering plan
///
typedef struct plan* (*plan_interface_t)(
    int n, int window_size, int values_per_chain, int tile_size,
    int *selected, int *complex_distr);

///
/// @brief Forms an one-part reordering plan.
///
/// @param[in] n                 problem dimension
/// @param[in] window_size       window size (-1 => automatic)
/// @param[in] values_per_chain  number of selected eigenvalues per window
///                              chain (-1 => automatic)
/// @param[in] tile_size         tile size
/// @param[in] selected          eigenvalue selection bitmap
/// @param[in] complex_distr     complex eigenvalue distribution bitmap
///
/// @return eigenvalue reordering plan
///
struct plan* starneig_formulate_plan(
    int n, int window_size, int values_per_chain, int tile_size,
    int *selected, int *complex_distr);

///
/// @brief Forms a multi-part reordering plan.
///
/// @param[in] n                 problem dimension
/// @param[in] window_size       window size (-1 => automatic)
/// @param[in] values_per_chain  number of selected eigenvalues per window
///                              chain (-1 => automatic)
/// @param[in] tile_size         tile size
/// @param[in] selected          eigenvalue selection bitmap
/// @param[in] complex_distr     complex eigenvalue distribution bitmap
///
/// @return eigenvalue reordering plan
///
struct plan* starneig_formulate_multiplan(
    int n, int window_size, int values_per_chain, int tile_size,
    int *selected, int *complex_distr);

#endif
