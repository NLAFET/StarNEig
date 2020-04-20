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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "node_internal.h"
#include "distr_matrix_internal.h"
#include "../common/common.h"
#include "../common/node_internal.h"
#include <starpu.h>
#include <starpu_mpi.h>

static MPI_Comm comm = MPI_COMM_WORLD;

///
/// @brief MPI information.
///
struct mpi_info info = {
    .tag_offset = 0
};

///
/// @ StarPU-MPI mode flag.
///
static enum {
    MPI_MODE_OFF,
    MPI_MODE_ON,
    MPI_MODE_PERSISTENT
} mpi_mode = MPI_MODE_OFF;

__attribute__ ((visibility ("default")))
void starneig_mpi_set_comm(MPI_Comm _comm)
{
    comm = _comm;
}

__attribute__ ((visibility ("default")))
MPI_Comm starneig_mpi_get_comm()
{
    return comm;
}

mpi_info_t starneig_mpi_get_info()
{
    return &info;
}

void starneig_mpi_start_persistent_starpumpi()
{
    starneig_verbose("Starting StarPU-MPI in persistent mode.");

    if (mpi_mode != MPI_MODE_OFF)
        starneig_fatal_error("StarPU-MPI already initialized.");

    info.tag_offset = 0;
    int ret = starpu_mpi_init_comm(NULL, NULL, 0, comm);
    if (ret != 0)
        starneig_fatal_error("Failed to initialize StarPU-MPI.");

    mpi_mode = MPI_MODE_PERSISTENT;
}

void starneig_mpi_stop_persistent_starpumpi()
{
    starneig_verbose("Shutting down StarPU-MPI persistent mode.");

    if (mpi_mode != MPI_MODE_PERSISTENT)
        starneig_fatal_error("StarPU-MPI is not in persistent mode.");

    starneig_mpi_cache_clear();
    starpu_mpi_shutdown();

    mpi_mode = MPI_MODE_OFF;
}

void starneig_mpi_pause_persistent_starpumpi()
{
    if (mpi_mode != MPI_MODE_PERSISTENT)
        return;

    starneig_verbose("Pausing StarPU-MPI persistent mode.");

    starneig_mpi_cache_clear();
    starpu_mpi_shutdown();
}

void starneig_mpi_resume_persistent_starpumpi()
{
    if (mpi_mode != MPI_MODE_PERSISTENT)
        return;

    starneig_verbose("Resuming StarPU-MPI persistent mode.");

    int ret = starpu_mpi_init_comm(NULL, NULL, 0, comm);
    if (ret != 0)
        starneig_fatal_error("Failed to initialize StarPU-MPI.");
}

void starneig_mpi_start_starpumpi()
{
    if (mpi_mode == MPI_MODE_PERSISTENT)
        return;

    starneig_verbose("Starting StarPU-MPI.");

    if (mpi_mode != MPI_MODE_OFF)
        starneig_fatal_error("StarPU-MPI already initialized.");

    starneig_node_set_mode(STARNEIG_MODE_DM);

    info.tag_offset = 0;
    int ret = starpu_mpi_init_comm(NULL, NULL, 0, comm);
    if (ret != 0)
        starneig_fatal_error("Failed to initialize StarPU-MPI.");
    mpi_mode = MPI_MODE_ON;
}

void starneig_mpi_stop_starpumpi()
{
    if (mpi_mode == MPI_MODE_PERSISTENT)
        return;

    starneig_verbose("Shutting down StarPU-MPI.");

    if (mpi_mode != MPI_MODE_ON)
        starneig_fatal_error("StarPU-MPI is not initialized.");

    starneig_mpi_cache_clear();
    starpu_mpi_shutdown();

    mpi_mode = MPI_MODE_OFF;
}

__attribute__ ((visibility ("default")))
void starneig_mpi_broadcast(int root, size_t size, void *buffer)
{
    CHECK_INIT();

    starneig_node_set_mode(STARNEIG_MODE_DM);
    starneig_mpi_start_starpumpi();
    starneig_node_resume_starpu();

    mpi_info_t mpi = starneig_mpi_get_info();

    starpu_data_handle_t buffer_h;

    starpu_variable_data_register(
        &buffer_h, STARPU_MAIN_RAM, (uintptr_t) buffer, size);
    starpu_mpi_data_register_comm(
        buffer_h, mpi->tag_offset++, root, starneig_mpi_get_comm());

    starpu_mpi_get_data_on_all_nodes_detached(
        starneig_mpi_get_comm(), buffer_h);

    starpu_data_unregister(buffer_h);

    starpu_task_wait_for_all();
    starpu_mpi_cache_flush_all_data(starneig_mpi_get_comm());
    starpu_mpi_barrier(starneig_mpi_get_comm());

    starneig_node_pause_starpu();
    starneig_mpi_stop_starpumpi();
}

// deprecated
__attribute__ ((visibility ("default")))
void starneig_broadcast(int root, size_t size, void *buffer)
{
    starneig_warning("starneig_broadcast has been deprecated.");
    starneig_mpi_broadcast(root, size, buffer);
}
