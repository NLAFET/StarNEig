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

#ifndef STARNEIG_HESSEBERG_CUDA_CLEANUP_H
#define STARNEIG_HESSEBERG_CUDA_CLEANUP_H

#include <starneig_config.h>
#include <starneig/configuration.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// @brief Cleanup function for starneig_hessenberg_ext_cuda_process_panel.
///
///  Inserts a cleanup tasks that gets executed on the same worker.
///
/// @param[inout] p1  gets freed using free
/// @param[inout] p2  gets freed using cudaFreeHost
/// @param[inout] p3  gets freed using cudaFree
///
void starneig_hessenberg_ext_insert_process_panel_cleanup(
    void *p1, void *p2, void *p3);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
