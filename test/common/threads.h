///
/// @file This file contains OpenMP and BLAS thread count configuration
/// interface.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
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

#ifndef STARNEIG_TEST_COMMOPN_THREADS_H
#define STARNEIG_TEST_COMMOPN_THREADS_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include <starneig/node.h>

///
/// @brief Thread mode.
///
typedef enum {
    THREADS_MODE_DEFAULT,   ///< Default mode.
    THREADS_MODE_BLAS       ///< BLAS mode.
} thread_mode_t;

///
/// @brief Initializes the interface.
///
/// @param[in] worker_threads
///         The number of StarPU threads to use.
///
/// @param[in] blas_threads
///         The number of BLAS threads to use.
///
void threads_init(int worker_threads, int blas_threads);

///
/// @brief Sets the threads mode.
///
/// @param[in] mode
///         The threads mode.
///
void threads_set_mode(thread_mode_t mode);

///
/// @brief Returns the number of StarPU worker threads.
///
/// @return The number of StarPU worker threads.
///
int threads_get_workers();

///
/// @brief Returns STARNEIG_FAST_DM flag if that is optimal.
///
/// @return StarNEig library initialization flag.
///
starneig_flag_t threads_get_fast_dm();

#endif // STARNEIG_TEST_COMMOPN_THREADS_H
