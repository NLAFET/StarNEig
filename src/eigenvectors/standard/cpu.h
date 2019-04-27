///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
///
/// @section LICENSE
///
/// Copyright (c) 2019, Umeå Universitet
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

#ifndef STARNEIG_EIGENVECTORS_STANDARD_CPU_H
#define STARNEIG_EIGENVECTORS_STANDARD_CPU_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starneig/error.h>
#include "typedefs.h"

void unify_scaling(int num_tiles, int *first_row, int *first_col,
    scaling_t *restrict scales,
    double *restrict X, int ldX,
    const int *restrict lambda_type, const int *restrict selected);


void starneig_cpu_bound(void *buffers[], void *cl_args);
void starneig_cpu_bound_DM(void *buffers[], void *cl_args);
void starneig_cpu_backsolve(void *buffers[], void *cl_args);
void starneig_cpu_solve(void *buffers[], void *cl_args);
void starneig_cpu_update(void *buffers[], void *cl_args);
void starneig_cpu_find_max_entry(void *buffers[], void *cl_args);
void starneig_cpu_backtransform(void *buffers[], void *cl_args);

#endif
