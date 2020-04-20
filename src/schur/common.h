///
/// @file
///
/// @brief This file contains code that is shared among all components of the
/// StarPU-bases QR algorithm.
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

#ifndef STARNEIG_SCHUR_COMMON_H
#define STARNEIG_SCHUR_COMMON_H

#include <starneig_config.h>
#include <starneig/configuration.h>

///
/// @brief Deflation and infinite eigenvalue check status.
///
typedef int bulge_chasing_aftermath_t;

#define BULGE_CHASING_AFTERMATH_NONE     0x0
#define BULGE_CHASING_AFTERMATH_DEFLATED 0x1
#define BULGE_CHASING_AFTERMATH_INFINITY 0x2

///
/// @brief Bulge chasing mode.
///
typedef enum {
    BULGE_CHASING_MODE_FULL,      ///< full bulge chasing
    BULGE_CHASING_MODE_INTRODUCE, ///< introduce and chaise bulges across window
    BULGE_CHASING_MODE_CHASE,     ///< chaise bulges across window
    BULGE_CHASING_MODE_FINALIZE   ///< chaise bulges out of window
} bulge_chasing_mode_t;

///
/// @brief AED status structure.
///
struct aed_status {
    enum {
        AED_STATUS_SUCCESS,
        AED_STATUS_FAILURE
    } status;                  ///< status
    int converged;             ///< number of computed eigenvalues
    int computed_shifts;       ///< number of successfully computed shifts
};

///
/// @brief small_schur codelet's return status.
///
struct small_schur_status {
    int converged;        ///< number of computed eigenvalues
};

///
/// @brief deflate codelet's return status.
///
struct deflate_status {
    int inherited;      ///< non-zero if the windows contains inherited blocks
    int begin;          ///< topmost undeflated diagonal block
    int end;            ///< bottommost undeflated diagonal block + 1
};

#endif
