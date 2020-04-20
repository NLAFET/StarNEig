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

#ifndef STARNEIG_TEST_HOOK_CONVERTER_H
#define STARNEIG_TEST_HOOK_CONVERTER_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "hook_experiment.h"

///
/// @brief Data format converter descriptor.
///
struct hook_data_converter {
    hook_data_format_t from;                ///< source format label
    hook_data_format_t to;                  ///< destination format label

    ///
    /// @brief Usage information / instruction printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_usage)(int argc, char * const *argv);

    ///
    /// @brief Command line arguments printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Command line argument validation function.
    ///
    /// @param[in]     argc  command line argument count
    /// @param[in]     argv  command line arguments
    /// @param[in,out] argr  array that tracks which command line arguments have
    ///                      been processed
    ///
    /// @return 0 if the arguments are valid, non-zero otherwise
    ///
    int (*check_args)(int argc, char * const *argv, int *argr);

    ///
    /// @brief Data conversion function.
    ///
    /// @param[in]     argc  command line argument count
    /// @param[in]     argv  command line arguments
    /// @param[in,out] env   data envelope
    ///
    /// @return 0 if the conversion was successful, non-zero otherwise
    ///
    int (*convert)(int argc, char * const *argv, struct hook_data_env *env);
};

#endif
