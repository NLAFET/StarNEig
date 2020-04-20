///
/// @file
///
/// @brief This file contains function that frees a memory buffer using
/// cudaFree.
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
#include "cuda_cleanup.h"
#include "common.h"
#include <starpu.h>

#ifdef STARNEIG_ENABLE_CUDA

static void cleanup(void *buffers[], void *cl_args)
{
    void *p;
    starpu_codelet_unpack_args(cl_args, &p);
    cudaFree(p);
}

static struct starpu_codelet cuda_free_cl = {
    .name = "starneig_cuda_free_cl",
    .cuda_funcs = { cleanup },
    .cuda_flags = { STARPU_CUDA_ASYNC }
};

void starneig_insert_cuda_free(void *p)
{
    struct starpu_task *current_task = starpu_task_get_current();
    int prio = starpu_sched_ctx_get_max_priority(current_task->sched_ctx);

    struct starpu_task *task = starpu_task_build(
        &cuda_free_cl,
        STARPU_PRIORITY, prio,
        STARPU_EXECUTE_ON_WORKER, starpu_worker_get_id(),
        STARPU_VALUE, &p, sizeof(p), 0);

    starpu_task_declare_deps_array(task, 1, &current_task);

    STARNEIG_ASSERT(
        starpu_task_submit_to_ctx(task, current_task->sched_ctx) == 0);
}

#endif
