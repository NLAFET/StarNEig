///
/// @file This file contains OpenMP and BLAS thread count configuration
/// interface.
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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "threads.h"
#include "parse.h"
#include "omp.h"
#include <hwloc.h>
#ifdef MKL_SET_NUM_THREADS_LOCAL_FOUND
#include <mkl.h>
#endif
#if defined(OPENBLAS_SET_NUM_THREADS_FOUND) || \
defined(GOTO_SET_NUM_THREADS_FOUND)
#include <cblas.h>
#endif

static struct {
    int worker_threads;
    int blas_threads;
    int lapack_threads;
    int scalapack_threads;
} status = {
    .worker_threads = 1,
    .blas_threads = 1,
    .lapack_threads = 1,
    .scalapack_threads = 1
};

static int get_core_count()
{
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
    for (int i = 0; i < num_cores; i++) {
        hwloc_obj_t core = hwloc_get_obj_by_depth(topology, depth_cores, i);

        // if the CORE has PUs inside it, ...
        if (core->first_child && core->first_child->type == HWLOC_OBJ_PU) {
            // iterate over them
            hwloc_obj_t pu = core->first_child;
            while (pu) {
                // if the PU is in the binding mask, ...
                hwloc_bitmap_and(res, mask, pu->cpuset);
                if (!hwloc_bitmap_iszero(res)) {
                    // count the PU as a CORE
                    num_hwloc_cpus++;
                    break;
                }
                pu = pu->next_sibling;
            }
        }
        else {
            // if the CORE is in the binding mask, ...
            hwloc_bitmap_and(res, mask, core->cpuset);
            if (!hwloc_bitmap_iszero(res)) {
                // count the CORE
                num_hwloc_cpus++;
            }
        }
    }

    hwloc_bitmap_free(mask);
    hwloc_bitmap_free(res);
    hwloc_topology_destroy(topology);

    return num_hwloc_cpus;
}

static void set_blas_threads(int threads)
{
#if defined(MKL_SET_NUM_THREADS_LOCAL_FOUND)
    mkl_set_num_threads_local(threads);
#elif defined(OPENBLAS_SET_NUM_THREADS_FOUND)
    openblas_set_num_threads(threads);
#elif defined(GOTO_SET_NUM_THREADS_FOUND)
    goto_set_num_threads(threads);
#endif
}

void thread_print_usage(int argc, char * const *argv)
{
    printf(
        "  --test-workers [(num),default] -- Test program StarPU worker count\n"
        "  --blas-threads [(num),default] -- Test program BLAS thread count\n"
        "  --lapack-threads [(num),default] -- LAPACK solver thread count\n"
        "  --scalapack-threads [(num),default] -- ScaLAPACK solver thread "
        "count\n"
    );
}

void thread_print_args(int argc, char * const *argv)
{
    print_multiarg("--test-workers", argc, argv, "default", NULL);
    print_multiarg("--blas-threads", argc, argv, "default", NULL);
    print_multiarg("--lapack-threads", argc, argv, "default", NULL);
    print_multiarg("--scalapack-threads", argc, argv, "default", NULL);
}

int thread_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t worker_threads =
        read_multiarg("--test-workers", argc, argv, argr, "default", NULL);

    if (worker_threads.type == MULTIARG_INVALID ||
    (worker_threads.type == MULTIARG_INT && worker_threads.int_value < 1)) {
        fprintf(stderr, "Invalid number of StarPU worker threads.\n");
        return 1;
    }

    struct multiarg_t blas_threads =
        read_multiarg("--blas-threads", argc, argv, argr, "default", NULL);

    if (blas_threads.type == MULTIARG_INVALID ||
    (blas_threads.type == MULTIARG_INT && blas_threads.int_value < 1)) {
        fprintf(stderr, "Invalid number of BLAS threads.\n");
        return 1;
    }

    struct multiarg_t lapack_threads =
        read_multiarg("--lapack-threads", argc, argv, argr, "default", NULL);

    if (lapack_threads.type == MULTIARG_INVALID ||
    (lapack_threads.type == MULTIARG_INT && lapack_threads.int_value < 1)) {
        fprintf(stderr, "Invalid number of LAPACK threads.\n");
        return 1;
    }

    struct multiarg_t scalapack_threads =
        read_multiarg("--scalapack-threads", argc, argv, argr, "default", NULL);

    if (scalapack_threads.type == MULTIARG_INVALID ||
    (scalapack_threads.type == MULTIARG_INT &&
    scalapack_threads.int_value < 1)) {
        fprintf(stderr, "Invalid number of ScaLAPACK threads.\n");
        return 1;
    }

    return 0;
}

void threads_init(int argc, char * const *argv)
{
    struct multiarg_t worker_threads =
        read_multiarg("--test-workers", argc, argv, NULL, "default", NULL);
    if (worker_threads.type == MULTIARG_INT)
        status.worker_threads = worker_threads.int_value;
    else
        status.worker_threads = get_core_count();
    printf(
        "THREADS: Using %d StarPU worker threads during initialization and "
        "validation.\n", status.worker_threads);

    struct multiarg_t blas_threads =
        read_multiarg("--blas-threads", argc, argv, NULL, "default", NULL);
    if (blas_threads.type == MULTIARG_INT)
        status.blas_threads = blas_threads.int_value;
    else
        status.blas_threads = get_core_count();
    printf(
        "THREADS: Using %d BLAS threads during initialization and "
        "validation.\n", status.blas_threads);

    struct multiarg_t lapack_threads =
        read_multiarg("--lapack-threads", argc, argv, NULL, "default", NULL);
    if (lapack_threads.type == MULTIARG_INT)
        status.lapack_threads = lapack_threads.int_value;
    else
        status.lapack_threads = get_core_count();
    printf(
        "THREADS: Using %d BLAS threads in LAPACK solvers.\n",
        status.lapack_threads);

    struct multiarg_t scalapack_threads =
        read_multiarg("--scalapack-threads", argc, argv, NULL, "default", NULL);
    if (scalapack_threads.type == MULTIARG_INT)
        status.scalapack_threads = scalapack_threads.int_value;
    else
        status.scalapack_threads = 1;
    printf(
        "THREADS: Using %d BLAS threads in ScaLAPACK solvers.\n",
        status.scalapack_threads);

    threads_set_mode(THREADS_MODE_DEFAULT);
}

void threads_set_mode(thread_mode_t mode)
{
    switch (mode) {
        case THREADS_MODE_BLAS:
            set_blas_threads(status.blas_threads);
            break;
        case THREADS_MODE_LAPACK:
            set_blas_threads(status.lapack_threads);
            break;
        case THREADS_MODE_SCALAPACK:
            set_blas_threads(status.scalapack_threads);
            break;
        default:
            set_blas_threads(1);
    }
}

int threads_get_workers()
{
    return status.worker_threads;
}

starneig_flag_t threads_get_fast_dm()
{
    if (1 < threads_get_workers())
        return STARNEIG_FAST_DM;
    return STARNEIG_HINT_DM;
}
