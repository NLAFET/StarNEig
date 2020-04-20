///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
/// @author Lars Karlsson (larsk@cs.umu.se), Umeå University
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

#ifndef STARNEIG_COMMON_NODE_INTERNAL_H
#define STARNEIG_COMMON_NODE_INTERNAL_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include <starneig/node.h>

#define CHECK_INIT() \
    if (!starneig_node_initialized()) { \
        starneig_fatal_error( \
            "%s(): module not initialized - call starneig_node_init() " \
            "first\n", __func__); \
    }

///
/// @brief Library mode.
///
enum starneig_mode {
    STARNEIG_MODE_OFF,  ///< No StarPU mode
    STARNEIG_MODE_SM,   ///< Shared memory mode
    STARNEIG_MODE_DM    ///< Distributed memory mode
};

///
/// @brief BLAS mode.
///
enum starneig_blas_mode {
    STARNEIG_BLAS_MODE_ORIGINAL,     ///< Original mode
    STARNEIG_BLAS_MODE_SEQUENTIAL,   ///< Sequential BLAS mode
    STARNEIG_BLAS_MODE_PARALLEL      ///< Parallel BLAS mode
};

///
/// @brief Changes the mode.
///
/// @param mode
///         New mode.
///
void starneig_node_set_mode(enum starneig_mode mode);

///
/// @brief Changes the BLAS mode.
///
/// @param mode
///         New mode.
///
void starneig_node_set_blas_mode(enum starneig_blas_mode mode);

///
/// @brief Pauses StarPU workers.
///
void starneig_node_pause_starpu();

///
/// @brief Wakes up StarPU workers.
///
void starneig_node_resume_starpu();

///
/// @brief Pauses awake StarPU workers. For (Sca)LAPACK wrappers.
///
void starneig_node_pause_awake_starpu();

///
/// @brief Wakes up "awake" StarPU workers. For (Sca)LAPACK wrappers.
///
void starneig_node_resume_awake_starpu();

#endif // STARNEIG_COMMON_NODE_INTERNAL_H
