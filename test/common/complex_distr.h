///
/// @file This file contains the 2-by-2 block generator modules.
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

#ifndef STARNEIG_TEST_COMMON_COMPLEX_DISTR_H
#define STARNEIG_TEST_COMMON_COMPLEX_DISTR_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "pencil.h"

///
/// @brief 2-by-2 block generator module structure.
///
struct complex_distr {
    char const *name;   ///< module name
    char const *desc;   ///< module description

    ///
    /// @brief Prints usage information.
    ///
    void (*print_usage)();

    ///
    /// @brief Checks the command line arguments.
    ///
    /// @param[in] argc     command line argument count
    /// @param[in] argv     command line arguments
    /// @param[inout] argr  array that tracks which command line arguments have
    ///                     been processed
    ///
    /// @return 0 if the command line arguments are valid, non-zero otherwise
    ///
    int (*check_args)(int argc, char * const *argv, int *argr);

    ///
    /// @brief Prints active command line arguments.
    ///
    /// @param[in] argc     command line argument count
    /// @param[in] argv     command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Transforms a upper triangular matrix A to a real upper
    /// quasi-triangular matrix.
    ///
    /// @param[in] argc     command line argument count
    /// @param[in] argv     command line arguments
    /// @param[inout] A     A matrix
    /// @param[inout] B     B matrix (generalized case)
    ///
    /// @return 0 when the function call was successful, non-zero otherwise
    ///
    int (*init)(int argc, char * const * argv, matrix_t A, matrix_t B);
};

///
/// @brief Prints available 2-by-2 block generator modules.
///
void print_avail_complex_distr();

///
/// @brief Prints 2-by-2 block generator module usage information.
///
void print_opt_complex_distr();

///
/// @brief Reads a 2-by-2 block generator module from the command line
/// arguments.
///
/// @param[in] name     parameter name
/// @param[in] argc     command line argument count
/// @param[in] argv     command line arguments
/// @param[inout] argr  array that tracks which command line arguments have
///
/// @return matching module if one exists, NULL otherwise
///
struct complex_distr const * read_complex_distr(
    char const *name, int argc, char * const *argv, int *argr);

#endif // STARNEIG_TEST_COMMON_COMPLEX_DISTR_H
