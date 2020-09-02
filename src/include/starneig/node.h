///
/// @file
///
/// @brief This file contains interface to configure the intra-node execution
/// environment.
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

#ifndef STARNEIG_NODE_H
#define STARNEIG_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>
#include <stddef.h>
#ifdef STARNEIG_ENABLE_MPI
#include <mpi.h>
#endif

///
/// @defgroup starneig_node Intra-node execution environment
///
/// @brief Interface to configure the intra-node execution environment
///
/// @{
///

///
/// @name Library initialization flags
/// @{
///

///
/// @brief Use all resources.
///
/// Tells StarNEig to use all available CPU cores / GPUs.
///
#define STARNEIG_USE_ALL                -1

///
/// @brief Library initialization flag data type.
///
typedef unsigned starneig_flag_t;

///
/// @brief Default mode.
///
/// As a default, the library configures itself to shared memory mode.
///
#define STARNEIG_DEFAULT                0x0

///
/// @brief Shared memory mode.
///
/// Initializes the library for shared memory computation. The library will
/// automatically reconfigure itself for distributed memory computation if
/// necessary
///
#define STARNEIG_HINT_SM                0x0

///
/// @brief Distributed memory mode.
///
/// Initializes the library for distributed memory computation. The library will
/// automatically reconfigure itself for shared memory computation if
/// necessary
///
#define STARNEIG_HINT_DM                0x1

///
/// @brief No FxT traces mode.
///
/// Disables FXT traces.
///
/// @attention This flag does not work reliably with all StarPU versions.
///
#define STARNEIG_FXT_DISABLE            0x2

///
/// @brief Awake worker mode.
///
/// Keeps the StarPU worker threads awake between interface function calls.
/// Improves the performance in certain situations but can interfere with other
/// software.
///
#define STARNEIG_AWAKE_WORKERS          0x4

///
/// @brief Awake MPI worker mode.
///
/// Keeps the StarPU-MPI communication thread awake between interface function
/// calls. Improves the performance in certain situations but can interfere with
/// other software.
///
#define STARNEIG_AWAKE_MPI_WORKER       0x8

///
/// @brief Fast distributed memory mode.
///
/// Keeps the worker threads and StarPU-MPI communication thread awake between
/// interface function calls. Improves the performance in certain situations but
/// can interfere with other software.
///
#define STARNEIG_FAST_DM (STARNEIG_HINT_DM | STARNEIG_AWAKE_WORKERS | STARNEIG_AWAKE_MPI_WORKER)

///
/// @brief No verbose mode.
///
/// Disables all additional verbose messages.
///
#define STARNEIG_NO_VERBOSE             0x10

///
/// @brief No messages mode.
///
/// Disables all messages (including verbose messages).
///
#define STARNEIG_NO_MESSAGES            (STARNEIG_NO_VERBOSE | 0x20)

///
/// @}
///

///
/// @brief Initializes the intra-node execution environment.
///
/// The interface function initializes StarPU (and cuBLAS) and pauses all worker
/// The `cores` argument specifies the **total number of used CPU cores**. In
/// distributed memory mode, one CPU core is automatically allocated for the
/// StarPU-MPI communication thread. One or more CPU cores are automatically
/// allocated for GPU devices.
///
/// @param[in] cores
///         The number of cores (threads) to use per MPI rank. Can be set to
///         STARNEIG_USE_ALL in which case the library uses all available cores.
///
/// @param[in] gpus
///         The number of GPUs to use per MPI rank. Can be set to
///         STARNEIG_USE_ALL in which case the library uses all available GPUs.
///
/// @param[in] flags
///         Initialization flags.
///
void starneig_node_init(int cores, int gpus, starneig_flag_t flags);

///
/// @brief Checks whether the intra-node execution environment is initialized.
///
/// @return Non-zero if the environment is initialized, 0 otherwise.
///
int starneig_node_initialized();

///
/// @brief Returns the number of cores (threads) per MPI rank.
///
/// @return The number of cores (threads) per MPI rank.
///
int starneig_node_get_cores();

///
/// @brief Changes the number of CPUs cores (threads) to use per MPI rank.
///
/// @param cores
///         The number of CPUs to use per MPI rank.
///
void starneig_node_set_cores(int cores);

///
/// @brief Returns the number of GPUs per MPI rank.
///
/// @return The number of GPUs per MPI rank.
///
int starneig_node_get_gpus();

///
/// @brief Changes the number of GPUs to use per MPI rank.
///
/// @param gpus
///         The number of GPUs to use per MPI rank.
///
void starneig_node_set_gpus(int gpus);

///
/// @brief Deallocates resources associated with the intra-node configuration.
///
void starneig_node_finalize();

#ifdef STARNEIG_ENABLE_CUDA

///
/// @name Pinned host memory
/// @{
///

///
/// @brief Enable CUDA host memory pinning.
///
/// Should be called before any memory allocations are made.
///
void starneig_node_enable_pinning();

///
/// @brief Disables CUDA host memory pinning.
///
/// Should be called before any memory allocations are made.
///
void starneig_node_disable_pinning();

///
/// @}
///

#endif

///
/// @}
///

// deprecated
#ifdef STARNEIG_ENABLE_MPI
void starneig_mpi_set_comm(MPI_Comm comm);
MPI_Comm starneig_mpi_get_comm();
#endif

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_NODE_H
