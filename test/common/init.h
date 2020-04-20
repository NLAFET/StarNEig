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

#ifndef STARNEIG_TESTS_COMMON_INIT_H
#define STARNEIG_TESTS_COMMON_INIT_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "pencil.h"
#include "hook_experiment.h"

///
/// @brief Matrix initialization helper mode
///
typedef enum {
    INIT_HELPER_ALL,
    INIT_HELPER_STARNEIG_PENCIL,
    INIT_HELPER_BLACS_PENCIL
} init_helper_mode_t;

///
/// @brief Matrix initialization helper.
///
typedef struct init_helper * init_helper_t;

///
/// @brief Prints matrix initialization helper usage information.
///
/// @param[in] prefix
///         The argument prefix.
///
/// @param[in] mode
///         The helper mode.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
void init_helper_print_usage(
    char const *prefix, init_helper_mode_t mode, int argc, char * const *argv);

///
/// @brief Prints matrix initialization helper command line arguments.
///
/// @param[in] prefix
///         The argument prefix.
///
/// @param[in] mode
///         The helper mode.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
void init_helper_print_args(
    char const *prefix, init_helper_mode_t mode, int argc, char * const *argv);

///
/// @brief Checks matrix initialization helper command line arguments.
///
/// @param[in] prefix
///         The argument prefix.
///
/// @param[in] mode
///         The helper mode.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
/// @param[in,out] argr
///         An array that tracks which command line arguments have been
///         processed.
///
/// @return 0 if the arguments are valid, non-zero otherwise.
///
int init_helper_check_args(
    char const *prefix, init_helper_mode_t mode, int argc, char * const *argv,
    int *argr);

///
/// @brief Creates a matrix initialization helper.
///
/// @param[in] prefix
///         The argument prefix.
///
/// @param[in] format
///         The matrix pencil format. Useful with hook experiments.
///
/// @param[in] m
///         The number of rows in the matrices.
///
/// @param[in] n
///         The number of columns in the matrices.
///
/// @param[in] dtype
///         The matrix data type.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
/// @return A matrix initialization helper.
///
init_helper_t init_helper_init_hook(
    char const *prefix, hook_data_format_t format, int m, int n,
    data_type_t dtype, int argc, char * const *argv);

///
/// @brief Creates a matrix initialization helper.
///
/// @param[in] prefix
///         The argument prefix.
///
/// @param[in] type
///         The matrix type.
///
/// @param[in] m
///         The number of rows in the matrices.
///
/// @param[in] n
///         The number of columns in the matrices.
///
/// @param[in] dtype
///         The matrix data type.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
/// @return A matrix initialization helper.
///
init_helper_t init_helper_init(
    char const *prefix, matrix_type_t type, int m, int n, data_type_t dtype,
    int argc, char * const *argv);

///
/// @brief Frees a previously allocated matrix initialization helper.
///
/// @param[in,out] helper
///         The matrix initialization helper to be freed.
///
void init_helper_free(init_helper_t helper);

matrix_t init_matrix(int m, int n, init_helper_t helper);

void init_identity(matrix_t matrix);

void init_random_full(matrix_t matrix);

void init_random_fullpos(matrix_t matrix);

///
/// @brief Generates a zero matrix
///
/// @param[in] m - row count
/// @param[in] n - column count
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_zero(int m, int n, init_helper_t helper);

///
/// @brief Generates an identity matrix
///
/// @param[in] m - row count
/// @param[in] n - column count
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_identity(int m, int n, init_helper_t helper);

///
/// @brief Generates a random matrix.
///
/// @param[in] m - row count
/// @param[in] n - column count
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_random_full(int m, int n, init_helper_t helper);

///
/// @brief Generates a random matrix with positive entries.
///
/// @param[in] m - row count
/// @param[in] n - column count
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_random_fullpos(int m, int n, init_helper_t helper);

///
/// @brief Generates a random upper triangular matrix
///
/// @param[in] m - row count
/// @param[in] n - column count
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_random_uptriag(int m, int n, init_helper_t helper);

///
/// @brief Generates a random upper Hessenberg matrix
///
/// @param[in] m - row count
/// @param[in] n - column count
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_random_hessenberg(int m, int n, init_helper_t helper);

///
/// @brief Generates a Householder matrix.
///
/// @param[in] n - matrix dimension
/// @param[in] helper - matrix initialization helper
///
/// @return a pointer to an allocated matrix descriptor structure
///
matrix_t generate_random_householder(int n, init_helper_t helper);

///
/// @brief Computes C = Q A Z^T
///
/// @param[in] mat_q
///         Matrix Q.
///
/// @param[in] mat_a
///         Matrix A.
///
/// @param[in] mat_z
///         Matrix Z.
///
/// @param[out] mat_c
///         Matrix C.
///
void mul_QAZT(
    matrix_t mat_q, matrix_t mat_a, matrix_t mat_z, matrix_t *mat_c);

#endif
