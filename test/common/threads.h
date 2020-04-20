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
    THREADS_MODE_BLAS,      ///< BLAS mode.
    THREADS_MODE_LAPACK,    ///< LAPACK mode.
    THREADS_MODE_SCALAPACK  ///< ScaLAPACK mode.
} thread_mode_t;

///
/// @brief Prints instructions.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
void thread_print_usage(int argc, char * const *argv);

///
/// @brief Prints command line arguments.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
void thread_print_args(int argc, char * const *argv);

///
/// @brief Checks command line arguments.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
/// @param[inout] argr
///         An array that tracks which command line arguments have been
///         processed.
///
/// @return 0 if the arguments are valid, non-zero otherwise
///
int thread_check_args(int argc, char * const *argv, int *argr);

///
/// @brief Initializes the interface.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
void threads_init(int argc, char * const *argv);

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
