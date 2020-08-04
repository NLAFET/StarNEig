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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "node_internal.h"
#ifdef STARNEIG_ENABLE_MPI
#include "../mpi/node_internal.h"
#include "../mpi/distr_matrix_internal.h"
#endif
#include "common.h"
#include "scratch.h"
#include <starneig/node.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <hwloc.h>
#include <starpu.h>
#ifdef MKL_SET_NUM_THREADS_LOCAL_FOUND
#include <mkl.h>
#endif
#if defined(OPENBLAS_SET_NUM_THREADS_FOUND) || \
defined(GOTO_SET_NUM_THREADS_FOUND)
#include <cblas.h>
#endif

static struct {
    /// initialization flag
    bool is_init;
    // initialization flags
    starneig_flag_t flags;
    /// library mode
    enum starneig_mode mode;
    /// blas mode
    enum starneig_blas_mode blas_mode;
    // original blas thread count
    int blas_threads_original;
    // StarPU worker bind mask
    unsigned starpu_workers_bindid[STARPU_NMAXWORKERS];
    // total number of available cpu cores
    int avail_cores;
    // total number of available gpus
    int avail_gpus;
    // number of used cpu cores
    int used_cores;
    // total number of used gpus
    int used_gpus;
} state = {
    .is_init = false,
    .flags = STARNEIG_DEFAULT,
    .mode = STARNEIG_MODE_OFF,
    .blas_mode = STARNEIG_BLAS_MODE_ORIGINAL,
    .blas_threads_original = -1,
    .avail_cores = 0,
    .avail_gpus = 0,
    .used_cores = 0,
    .used_gpus = 0
};

///
/// @brief Sets the number of BLAS threads.
///
/// @param[in] threads
///         Number of BLAS threads.
///
/// @return Previous BLAS thread count (can be 0).
///
static int set_blas_threads(int threads)
{
    starneig_verbose("Setting BLAS thread count to %d.", threads);
#ifdef MKL_SET_NUM_THREADS_LOCAL_FOUND
    return mkl_set_num_threads_local(threads);
#elif defined(OPENBLAS_SET_NUM_THREADS_FOUND)
    int old = openblas_get_num_threads();
    openblas_set_num_threads(threads);
    return old;
#elif defined(GOTO_SET_NUM_THREADS_FOUND)
    goto_set_num_threads(threads);
    return 1;
#else
    return 1;
#endif
}

///
/// @brief Sets the BLAS mode.
///
/// @param[in] mode
///         BLAS mode.
///
static void set_blas_mode(enum starneig_blas_mode mode)
{
    int old = -1;
    switch (mode) {
        case STARNEIG_BLAS_MODE_PARALLEL:
            starneig_verbose("Switching to parallel BLAS.");
            old = set_blas_threads(state.used_cores);
            break;
        case STARNEIG_BLAS_MODE_SEQUENTIAL:
            starneig_verbose("Switching to sequential BLAS.");
            old = set_blas_threads(1);
            break;
        default:
            starneig_verbose("Restoring BLAS mode.");
            if (0 <= state.blas_threads_original)
                set_blas_threads(state.blas_threads_original);
            state.blas_threads_original = -1;
    }
    if (state.blas_mode == STARNEIG_BLAS_MODE_ORIGINAL && 0 <= old)
        state.blas_threads_original = old;
    state.blas_mode = mode;
}

#ifdef MKL_SET_NUM_THREADS_LOCAL_FOUND

///
/// @brief Sets the per thread BLAS thread count to 1.
///
/// @param[in] arg
///         An unused argument.
///
static void set_worker_blas_mode(void *arg)
{
    mkl_set_num_threads_local(1);
}

#endif

///
/// @brief Reconfigures the node.
///
/// @param[in] cores
///         Number of CPU cores to use.
///
/// @param[in] gpus
///         Number of GPUs to use.
///
/// @param[in] mode
///         Library mode.
///
/// @param[in] blas_mode
///         BLAS mode.
///
#define CONFIGURE(cores, gpus, mode, blas_mode) \
    node_configure(cores, gpus, mode, blas_mode, __func__)

///
/// @brief Reconfigures the node.
///
/// @param[in] cores
///         Number of CPU cores to use.
///
/// @param[in] gpus
///         Number of GPUs to use.
///
/// @param[in] mode
///         Library mode.
///
/// @param[in] blas_mode
///         BLAS mode
///
/// @param[in] func
///         Name of the calling function.
///
static void node_configure(
    int cores, int gpus, enum starneig_mode mode,
    enum starneig_blas_mode blas_mode, char const *func)
{
#ifndef STARNEIG_ENABLE_MPI
    if (mode == STARNEIG_MODE_DM)
        starneig_fatal_error("StarPU was compiled without MPI support.");
#endif

    if (cores == state.used_cores && gpus == state.used_gpus &&
    mode == state.mode && blas_mode == state.blas_mode)
        return;

    starneig_verbose("Reconfiguring the library.");

    if (cores == state.used_cores && gpus == state.used_gpus &&
    mode == state.mode) {
        set_blas_mode(blas_mode);
        return;
    }

    //
    // shutdown StarPU
    //

    if (state.mode != STARNEIG_MODE_OFF) {
        starneig_node_resume_starpu();
#ifdef STARNEIG_ENABLE_CUDA
        if (0 < state.used_gpus) {
            starneig_verbose("Shutting down cuBLAS.");
            starpu_cublas_shutdown();
        }
#endif

        starneig_verbose("Shutting down StarPU.");

        starneig_scratch_unregister();
#ifdef STARNEIG_ENABLE_MPI
        starneig_mpi_cache_clear();
    if (state.mode == STARNEIG_MODE_DM &&
    state.flags & STARNEIG_AWAKE_MPI_WORKER)
        starneig_mpi_stop_persistent_starpumpi();
#endif

        starpu_task_wait_for_all();
        starpu_shutdown();
    }

    //
    // set the number of CPU cores
    //

    if (cores == 0)
        starneig_fatal_error("At least one CPU core must be selected.");

    if (cores < 0) {
        state.used_cores = state.avail_cores;
    }
    else {
        state.used_cores = MIN(cores, state.avail_cores);
        if (state.avail_cores < cores)
            starneig_warning(
                "Failed to acquire the desired number of CPU cores. "
                "Acquired %d.", state.used_cores);
    }

    //
    // set the number of GPUs
    //

    if (gpus < 0) {
        state.used_gpus = state.avail_gpus;
    }
    else {
#ifdef STARNEIG_ENABLE_CUDA
        state.used_gpus = MIN(gpus, state.avail_gpus);

        if (state.avail_gpus < gpus)
            starneig_warning(
                "Failed to acquire the desired number of CUDA devices. "
                "Acquired %d.", state.used_gpus);
#else
        if (0 < gpus)
            starneig_warning("StarPU was compiled without CUDA support.");
#endif
    }

    //
    // set BLAS threads
    //

    set_blas_mode(blas_mode);

    //
    // set mode
    //

    state.mode = mode;
    if (state.mode == STARNEIG_MODE_OFF)
        return;

    //
    // create StarPU configuration
    //

    starneig_verbose("Configuring StarPU.");

    struct starpu_conf conf;
    starpu_conf_init(&conf);

    int cpu_workers = state.used_cores;
    if (0 < state.used_gpus)
        cpu_workers -= state.used_gpus;
    if (state.mode == STARNEIG_MODE_DM)
        cpu_workers--;

    conf.ncpus = MAX(1, cpu_workers);
    conf.ncuda = state.used_gpus;
    conf.nopencl = 0;

//#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
    if (getenv("STARPU_WORKERS_CPUID") == NULL)
        conf.use_explicit_workers_bindid = 1;
//#endif
    memcpy(conf.workers_bindid, state.starpu_workers_bindid,
        sizeof(state.starpu_workers_bindid));

#ifdef STARNEIG_ENABLE_CUDA
    if (0 < state.used_gpus)
        conf.sched_policy_name = "dmdas";
    else
#endif
        conf.sched_policy_name = "prio";

    //
    // setup FXT
    //

    if (state.flags & STARNEIG_FXT_DISABLE) {
        starneig_verbose("Disabling FXT traces.");
        starpu_fxt_autostart_profiling(0);
    }
    else {
        char const *starpu_fxt_trace = getenv("STARPU_FXT_TRACE");
        if (starpu_fxt_trace == NULL || atoi(starpu_fxt_trace) != 0) {
            starneig_verbose("Keeping FXT traces enabled.");
            starpu_fxt_autostart_profiling(1);
        }
    }

    //
    // initialize StarPU
    //

    starneig_verbose("Starting StarPU.");

    unsigned seed = rand();
    int ret = starpu_init(&conf);
    srand(seed);

    if (ret != 0)
        starneig_fatal_error("Failed to initialize StarPU.");

    starpu_profiling_status_set(STARPU_PROFILING_ENABLE);
    starpu_malloc_set_align(64);

    //
    // initialize persistent StarPU-MPI
    //

#ifdef STARNEIG_ENABLE_MPI
    if (state.mode == STARNEIG_MODE_DM &&
    state.flags & STARNEIG_AWAKE_MPI_WORKER)
        starneig_mpi_start_persistent_starpumpi();
#endif

    //
    // cuBLAS
    //

    if (0 < state.used_gpus) {
        starneig_verbose("Initializing cuBLAS.");
        starpu_cublas_init();
    }

    //
    // configure workers
    //

#ifdef MKL_SET_NUM_THREADS_LOCAL_FOUND
    starneig_verbose(
        "MKL detected. Setting StarPU worker BLAS thread count to 1.");
    starpu_execute_on_each_worker(
        &set_worker_blas_mode, NULL, STARPU_CPU | STARPU_CUDA);
#endif

    starneig_node_pause_starpu();
}

void starneig_node_pause_starpu()
{
    if (state.flags & STARNEIG_AWAKE_WORKERS)
        return;

    starneig_verbose("Pausing StarPU workers.");
    starpu_pause();
}

void starneig_node_resume_starpu()
{
    if (state.flags & STARNEIG_AWAKE_WORKERS)
        return;

    starneig_verbose("Waking up StarPU workers.");
    starpu_resume();
}

void starneig_node_pause_awake_starpu()
{
    if (!(state.flags & STARNEIG_AWAKE_WORKERS))
        return;

    starneig_verbose("Pausing \"awake\" StarPU workers.");
    starpu_pause();
}

void starneig_node_resume_awake_starpu()
{
    if (!(state.flags & STARNEIG_AWAKE_WORKERS))
        return;

    starneig_verbose("Waking up \"awake\" StarPU workers.");
    starpu_resume();
}

__attribute__ ((visibility ("default")))
void starneig_node_init(int cores, int gpus, starneig_flag_t flags)
{
    starneig_set_message_mode(
        !(flags & STARNEIG_NO_MESSAGES), !(flags & STARNEIG_NO_VERBOSE));

    starneig_verbose("Initializing node.");

    if (state.is_init)
        starneig_fatal_error("The node is already initialized.");

    state.flags = flags;

    //
    // set up CUDA
    //

    state.avail_gpus = 0;
    state.used_gpus = 0;

#ifdef STARNEIG_ENABLE_CUDA
    // query the number of available CUDA devices
	if (cudaGetDeviceCount(&state.avail_gpus) != cudaSuccess) {
        starneig_warning("Failed to acquire CUDA device count.");
        state.avail_gpus = 0;
    }
#endif

    // query STARPU_NCUDA environment
    char *starpu_ncuda = getenv("STARPU_NCUDA");
    int num_starpu_gpus = (starpu_ncuda ? atoi(starpu_ncuda) : -1);

    if (num_starpu_gpus != -1) {
        if (state.avail_gpus < num_starpu_gpus)
        starneig_warning(
            "A conflict between STARPU_NCUDA and cudaGetDeviceCount().");
        state.avail_gpus = MIN(state.avail_gpus, num_starpu_gpus);
    }

    //
    // set up CPU cores
    //

    state.avail_cores = 0;
    state.used_cores = 0;

    // query the SLURM environment
    char *slurm = getenv("SLURM_CPUS_PER_TASK");
    const int num_slurm_cpus = (slurm ? atoi(slurm) : -1);

    // query STARPU_NCPUS environment
    char *starpu_ncpus = getenv("STARPU_NCPUS");
    int num_starpu_cpus = -1;
    if (starpu_ncpus) {
        if (0 < state.avail_gpus)
            num_starpu_cpus = atoi(starpu_ncpus) + 1;
        else
            num_starpu_cpus = atoi(starpu_ncpus);
    }

    // query hardware topology and CPU core binding mask

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    hwloc_cpuset_t res = hwloc_bitmap_alloc();

    hwloc_cpuset_t mask = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology, mask, HWLOC_CPUBIND_THREAD);

    int depth_cores = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    int num_cores = hwloc_get_nbobjs_by_depth(topology, depth_cores);

    int num_hwloc_cpus = 0;

    // iterate over all COREs
    for (int i = 0; i < num_cores && num_hwloc_cpus < STARPU_NMAXWORKERS; i++) {
        hwloc_obj_t core = hwloc_get_obj_by_depth(topology, depth_cores, i);

        // if the CORE has PUs inside it, ...
        if (core->first_child && core->first_child->type == HWLOC_OBJ_PU) {
            // iterate over them
            hwloc_obj_t pu = core->first_child;
            while (pu) {
                // if the PU is in the binding mask, ...
                hwloc_bitmap_and(res, mask, pu->cpuset);
                if (!hwloc_bitmap_iszero(res)) {
                    // add it to the worker list
                    state.starpu_workers_bindid[num_hwloc_cpus++] =
                        pu->logical_index;
                    break;
                }
                pu = pu->next_sibling;
            }
        }
        else {
            // if the CORE is in the binding mask, ...
            hwloc_bitmap_and(res, mask, core->cpuset);
            if (!hwloc_bitmap_iszero(res)) {
                // add it to the worker list
                state.starpu_workers_bindid[num_hwloc_cpus++] =
                    core->logical_index;
            }
        }
    }

    hwloc_bitmap_free(mask);
    hwloc_bitmap_free(res);
    hwloc_topology_destroy(topology);

    starneig_verbose_begin("Attached CPU cores");
    for (int i = 0; i < num_hwloc_cpus; i++)
        starneig_verbose_cont(" %d", state.starpu_workers_bindid[i]);
    starneig_verbose_cont(".\n");

    // set avail_cores

    state.avail_cores = num_hwloc_cpus;
    if (0 < num_starpu_cpus) {
        if (num_hwloc_cpus < num_starpu_cpus)
            starneig_warning(
                "A conflict between STARPU_NCPUS/STARPU_NCUDA and hwloc core "
                "binding mask.");
        state.avail_cores = MIN(state.avail_cores, num_starpu_cpus);
        if (0 < num_slurm_cpus) {
            if (num_slurm_cpus < num_starpu_cpus)
                starneig_warning(
                    "A conflict between STARPU_NCPUS/STARPU_NCUDA and "
                    "SLURM_CPUS_PER_TASK.");
            state.avail_cores = MIN(state.avail_cores, num_slurm_cpus);
        }
    }
    if (0 < num_slurm_cpus) {
        if (num_hwloc_cpus < num_slurm_cpus)
            starneig_warning(
                "A conflict between SLURM_CPUS_PER_TASK and hwloc core "
                "binding mask.");
        state.avail_cores = MIN(state.avail_cores, num_slurm_cpus);
    }

    if (state.avail_cores <= 0)
        starneig_fatal_error("Something unexpected happened.");

    state.is_init   = true;

    if (state.flags & STARNEIG_HINT_DM)
        CONFIGURE(cores, gpus, STARNEIG_MODE_DM, STARNEIG_BLAS_MODE_SEQUENTIAL);
    else
        CONFIGURE(cores, gpus, STARNEIG_MODE_SM, STARNEIG_BLAS_MODE_SEQUENTIAL);
}

__attribute__ ((visibility ("default")))
int starneig_node_initialized()
{
    return state.is_init;
}

__attribute__ ((visibility ("default")))
void starneig_node_finalize(void)
{
    CHECK_INIT();

    starneig_verbose("De-initializing node.");

    CONFIGURE(-1, -1, STARNEIG_MODE_OFF, STARNEIG_BLAS_MODE_ORIGINAL);

    starneig_set_message_mode(0, 0);

    state.avail_cores = 0;
    state.avail_gpus = 0;

    state.is_init = false;
}

__attribute__ ((visibility ("default")))
int starneig_node_get_cores(void)
{
    CHECK_INIT();
    return state.used_cores;
}

__attribute__ ((visibility ("default")))
void starneig_node_set_cores(int cores)
{
    CHECK_INIT();
    CONFIGURE(cores, starneig_node_get_gpus(), state.mode, state.blas_mode);
}

__attribute__ ((visibility ("default")))
int starneig_node_get_gpus(void)
{
    CHECK_INIT();
    return state.used_gpus;
}

__attribute__ ((visibility ("default")))
void starneig_node_set_gpus(int gpus)
{
    CHECK_INIT();
    CONFIGURE(starneig_node_get_cores(), gpus, state.mode, state.blas_mode);
}

__attribute__ ((visibility ("default")))
void starneig_node_set_mode(enum starneig_mode mode)
{
    CHECK_INIT();
    CONFIGURE(starneig_node_get_cores(), starneig_node_get_gpus(),
        mode, state.blas_mode);
}

void starneig_node_set_blas_mode(enum starneig_blas_mode blas_mode)
{
    CHECK_INIT();
    CONFIGURE(starneig_node_get_cores(), starneig_node_get_gpus(),
        state.mode, blas_mode);
}
