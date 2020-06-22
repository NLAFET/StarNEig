///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
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
#include "../common/common.h"
#include <starpu.h>

#ifdef STARNEIG_ENABLE_CUDA

static void process_panel_cleanup(void *buffers[], void *cl_args)
{
    void *p1, *p2, *p3;
    starpu_codelet_unpack_args(cl_args, &p1, &p2, &p3);
    free(p1);
    cudaFreeHost(p2);
    cudaFree(p3);
}

static struct starpu_codelet process_panel_cleanup_cl = {
    .name = "starneig_process_panel_cleanup",
    .cuda_funcs = { process_panel_cleanup },
    .cuda_flags = { STARPU_CUDA_ASYNC }
};

void starneig_hessenberg_ext_insert_process_panel_cleanup(
    void *p1, void *p2, void *p3)
{
    struct starpu_task *current_task = starpu_task_get_current();
    int prio = starpu_sched_ctx_get_max_priority(current_task->sched_ctx);

    struct starpu_task *task = starpu_task_build(
        &process_panel_cleanup_cl,
        STARPU_PRIORITY, prio,
        STARPU_EXECUTE_ON_WORKER, starpu_worker_get_id(),
        STARPU_VALUE, &p1, sizeof(p1),
        STARPU_VALUE, &p2, sizeof(p2),
        STARPU_VALUE, &p3, sizeof(p3), 0);

    starpu_task_declare_deps_array(task, 1, &current_task);

    STARNEIG_ASSERT(starpu_task_submit_to_ctx(task, current_task->sched_ctx) == 0);
}

#endif
