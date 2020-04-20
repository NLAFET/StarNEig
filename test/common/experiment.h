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

#ifndef STARNEIG_TESTS_COMMON_EXPERIMENT_H
#define STARNEIG_TESTS_COMMON_EXPERIMENT_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>

///
/// @brief Data type for generic experiment info field.
///
typedef void const * experiment_info_t;

///
/// @brief Prints instructions.
///
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[in] info - info field
///
typedef void (*experiment_print_usage_t)(
    int argc, char * const *argv, experiment_info_t info);

///
/// @brief Prints command line arguments.
///
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[in] info - info field
///
typedef void (*experiment_print_args_t)(
    int argc, char * const *argv, experiment_info_t info);

///
/// @brief Checks command line arguments.
///
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an array that tracks which command line arguments have
///                      been processed
/// @param[in] info - info field
///
/// @return 0 if the arguments are valid, non-zero otherwise
///
typedef int (*experiment_check_args_t)(
    int argc, char * const *argv, int *argr, experiment_info_t info);

///
/// @brief Executes the experiment.
///
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[in] info - info field
///
/// @return 0 if the solver was executed successfully, non-zero otherwise
///
typedef int (*experiment_run_t)(
    int argc, char * const *argv, experiment_info_t info);


///
/// @brief Experiment descriptor structure.
///
struct experiment_descr {
    char *name;                             ///< experiment name
    char *desc;                             ///< experiment description
    experiment_print_usage_t print_usage;   ///< prints usage information
    experiment_print_args_t print_args;     ///< prints command line arguments
    experiment_check_args_t check_args;     ///< checks command line arguments
    experiment_run_t run;                   ///< executes the experiment
    experiment_info_t info;                 ///< optional info field
};

#endif
