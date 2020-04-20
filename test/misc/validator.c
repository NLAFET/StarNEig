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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "validator.h"
#include "../common/hooks.h"
#include "../common/io.h"

static hook_solver_state_t dummy_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return env;
}

static int dummy_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int dummy_run(hook_solver_state_t state)
{
    return 0;
}

static const struct hook_solver dummy_solver = {
    .name = "dummy",
    .desc = "Dummy solver",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .prepare = &dummy_prepare,
    .finalize = &dummy_finalize,
    .run = &dummy_run
};

const struct hook_experiment_descr validator_experiment = {
    .initializers = (struct hook_initializer_t const *[])
    {
        &raw_initializer,
        0
    },
    .supplementers = (struct hook_supplementer_t const *[])
    {
        0
    },
    .solvers = (struct hook_solver const *[])
    {
        &dummy_solver,
        0
    },
    .hook_descrs = (struct hook_descr_t const *[])
    {
        (struct hook_descr_t[]) {{
            .is_enabled = 0,
            .default_mode = HOOK_MODE_NORMAL,
            .hook = &hessenberg_test
        }},
        (struct hook_descr_t[]) {{
            .is_enabled = 0,
            .default_mode = HOOK_MODE_NORMAL,
            .hook = &schur_test
        }},
        (struct hook_descr_t[]) {{
            .is_enabled = 0,
            .default_mode = HOOK_MODE_NORMAL,
            .hook = &analysis_test
        }},
        (struct hook_descr_t[]) {{
            .is_enabled = 0,
            .default_mode = HOOK_MODE_NORMAL,
            .hook = &residual_test
        }},
        &default_print_pencil_descr,
        &default_store_raw_pencil_descr,
        0
    }
};
