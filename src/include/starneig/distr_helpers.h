///
/// @file
///
/// @brief This file contains generic distributed memory interface functions.
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

#ifndef STARNEIG_DISTR_HELPERS_H
#define STARNEIG_DISTR_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

#ifndef STARNEIG_ENABLE_MPI
#error "This header should be included only when STARNEIG_ENABLE_MPI is defined."
#endif

#include <stddef.h>
#include <mpi.h>

///
/// @defgroup starneig_distr_helpers Distributed Memory / Helper functions
///
/// @brief Distributed memory helper functions.
///
/// @{
///

///
/// @name MPI communicator
/// @{
///

///
/// @brief Sets a MPI communicator for the library.
///
/// Should be called before the starneig_node_init() interface function.
///
/// @param[in] comm
///         The library MPI communicator.
///
void starneig_mpi_set_comm(MPI_Comm comm);

///
/// @brief Returns the library MPI communicator.
///
/// @return The library MPI communicator.
///
MPI_Comm starneig_mpi_get_comm();

///
/// @}
///

///
/// @name Broadcast
/// @{
///

///
/// @brief Broadcast a buffer.
///
/// @param[in] root
///         The rank that is going to broadcast the buffer.
///
/// @param[in] size
///         The size of the buffer.
///
/// @param[in,out] buffer
///         A pointer to the buffer.
///
void starneig_mpi_broadcast(int root, size_t size, void *buffer);

///
/// @brief Broadcast a buffer. Deprecated.
///
/// @deprecated The starneig_broadcast() function has been replaced with the
/// starneig_mpi_broadcast() function. This function will be removed in a
/// future release of the library.
///
void starneig_broadcast(int root, size_t size, void *buffer);

///
/// @}
///

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_DISTR_HELPERS_H
