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

#ifndef STARNEIG_MPI_NODE_INTERNAL_H
#define STARNEIG_MPI_NODE_INTERNAL_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "../common/common.h"

///
/// @brief Retuns the MPI tag offset.
///
mpi_info_t starneig_mpi_get_info();

///
/// @brief Starts StarPU-MPI in persistent mode.
///
void starneig_mpi_start_persistent_starpumpi();

///
/// @brief Stops StarPU-MPI persistent mode.
///
void starneig_mpi_stop_persistent_starpumpi();

///
/// @brief Pauses StarPU-MPI in persistent mode. For (Sca)LAPACK wrappers.
///
void starneig_mpi_pause_persistent_starpumpi();

///
/// @brief Resumes StarPU-MPI persistent mode. For (Sca)LAPACK wrappers.
///
void starneig_mpi_resume_persistent_starpumpi();

///
/// @brief Starts StarPU-MPI.
///
void starneig_mpi_start_starpumpi();

///
/// @brief Stops StarPU-MPI.
///
void starneig_mpi_stop_starpumpi();

#endif // STARNEIG_MPI_NODE_INTERNAL_H
