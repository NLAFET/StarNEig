///
/// @file
///
/// @brief This file contains the CPU implementations of codelets that are used
/// in the StarPU-bases QR algorithm.
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

#ifndef STARNEIG_SCHUR_CPU_H
#define STARNEIG_SCHUR_CPU_H

#include <starneig_config.h>
#include <starneig/configuration.h>

///
/// @prief push_inf_top codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_push_inf_top(void *buffers[], void *cl_arg);

///
/// @prief push_bulges codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_push_bulges(void *buffers[], void *cl_arg);

///
/// @prief aggressively_deflate codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_aggressively_deflate(void *buffers[], void *cl_arg);

///
/// @prief small_schur codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_small_schur(void *buffers[], void *cl_arg);

///
/// @prief small_hessenberg codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_small_hessenberg(void *buffers[], void *cl_arg);

///
/// @prief form_spike codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_form_spike(void *buffers[], void *cl_arg);

///
/// @prief embed_spike codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_embed_spike(void *buffers[], void *cl_arg);

///
/// @prief deflate codelet / CPU implementation.
///
/// @param[in,out] buffers - StarPU buffers
/// @param[in] cl_arg - StarPU arguments
///
void starneig_cpu_deflate(void *buffers[], void *cl_arg);

///
/// @prief extract_shifts codelet / CPU implementation.
///
/// @param[in,out] buffers  StarPU buffers
/// @param[in]     cl_arg   StarPU arguments
///
void starneig_cpu_extract_shifts(void *buffers[], void *cl_args);

///
/// @prief compute_norm_a codelet / CPU implementation.
///
/// @param[in,out] buffers  StarPU buffers
/// @param[in]     cl_arg   StarPU arguments
///
void starneig_cpu_compute_norm_a(void *buffers[], void *cl_args);

///
/// @prief compute_norm_b codelet / CPU implementation.
///
/// @param[in,out] buffers  StarPU buffers
/// @param[in]     cl_arg   StarPU arguments
///
void starneig_cpu_compute_norm_b(void *buffers[], void *cl_args);

#endif
