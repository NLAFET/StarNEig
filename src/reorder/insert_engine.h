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

#ifndef STARNEIG_REORDER_INSERT_ENGINE_H
#define STARNEIG_REORDER_INSERT_ENGINE_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "plan.h"
#include "../common/matrix.h"
#include "../common/vector.h"
#include <starpu.h>

///
/// @brief Blueprint enumerator.
///
typedef enum {

    //
    // structural commands
    //

    CHAIN_LIST_BEGIN,           ///< apply following commands to all chain lists

    CHAIN_LIST_END,             ///< closes CHAIN_LIST_BEGIN

    CHAIN_FORWARD,              ///< apply following commands to all chains
                                ///< starting from the first chain in the list

    CHAIN_BACKWARD,             ///< apply following commands to all chains
                                ///< starting from the last chain in the list

    CHAIN_END,                  ///< closes CHAIN_FORWARD / CHAIN_BACKWARD

    WINDOW_BEGIN,               ///< apply following commands to all windows

    WINDOW_END,                 ///< closes WINDOW_BEGIN

    //
    // window chain commands
    //

    WINDOWS,                    ///< insert all windows in a window chain

    LEFT_UPDATES,               ///< insert all left-hand side updates that
                                ///< correspond to a window chain

    RIGHT_UPDATES,              ///< insert all right-hand side updates that
                                ///< correspond to a window chain

    REMAINING_RIGHT_UPDATES,    ///< insert all remaining right-hand side
                                ///< updates that correspond to a window chain

    Q_UPDATES,                  ///< insert all Q matrix updates that
                                ///< correspond to a window chain

    //
    // window commands
    //

    DUMMY_WINDOW,               ///< insert a window

    DUMMY_LEFT_UPDATE,          ///< insert left-hand side updates that
                                ///< correspond to a window

    DUMMY_RIGHT_UPDATE,         ///< insert right-hand side updates that
                                ///< correspond to a window

    DUMMY_Q_UPDATE,             ///< insert Q matrix updates that correspond
                                ///< to a window

    //
    // miscellaneous commands
    //

    UNREGISTER,                 ///< unregister resources that correspond to
                                ///< a chain list / a chain / a window

    END                         ///< closes the whole command chain

} blueprint_step_t;

///
/// @brief Task insertion engine configuration structure.
///
struct starneig_engine_conf_t {
    int small_window_size;      ///< small window size
    int small_window_threshold; ///< small window threshold
    int q_height;               ///< height of a single Q matrix update task
    int z_height;               ///< height of a single Z matrix update task
    int a_width;                ///< width of a single A matrix update task
    int a_height;               ///< height of a single A matrix update task
    int b_width;                ///< width of a single B matrix update task
    int b_height;               ///< height of a single B matrix update task
};

///
/// @brief Processes a plan.
///
/// @param[in] conf - configuration structure
/// @param[in] steps - blueprint
/// @param[in,out] selected - eigenvalue selection bitmap descriptor
/// @param[in,out] matrix_q - matrix Q descriptor
/// @param[in,out] matrix_z - matrix Z descriptor
/// @param[in,out] matrix_a - matrix A descriptor
/// @param[in,out] matrix_b - matrix B descriptor
/// @param[in,out] plan - plan descriptor
/// @param[in,out] mpi  MPI info
///
void starneig_process_plan(
    struct starneig_engine_conf_t const *conf,
    blueprint_step_t const *steps,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_q,
    starneig_matrix_descr_t matrix_z,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct plan *plan,
    mpi_info_t mpi);

#endif
