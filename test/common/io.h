///
/// @file This file contains the input and output functionality of the test
/// program.
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

#ifndef STARNEIG_TESTS_COMMON_IO_H
#define STARNEIG_TESTS_COMMON_IO_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "pencil.h"
#include "init.h"

///
/// @brief Reads the dimensions of a matrix from a MTX file.
///
/// @param[in] name
///         The file name.
///
/// @param[out] m
///         The number of rows in the matrix.
///
/// @param[out] n
///         The number of columns in the matrix.
///
void read_mtx_dimensions_from_file(char const *name, int *m, int *n);

///
/// @brief Reads a matrix from a MTX file.
///
/// @param[in] name
///         The file name.
///
/// @param[in,out] helper
///         The initialization helper.
///
/// @return The matrix.
///
matrix_t read_mtx_matrix_from_file(char const *name, init_helper_t helper);

///
/// @brief Reads a submatrix from a MTX file.
///
/// @param[in] begin
///         The first row/column that belongs to the submatrix.
///
/// @param[in] end
///         The last row/column that belongs to the submatrix.
///
/// @param[in] name
///         The file name.
///
/// @param[in,out] helper
///         The initialization helper.
///
/// @return The submatrix.
///
matrix_t read_mtx_sub_matrix_from_file(
    int begin, int end, char const *name, init_helper_t helper);

///
/// @brief Writes a matrix to a file.
///
/// @param[in] name
///         The file name.
///
/// @param[in] matrix
///         The matrix.
///
void write_raw_matrix_to_file(
    char const *name, matrix_t desc);

///
/// @brief Reads the dimensions of a matrix from a file.
///
/// @param[in] name
///         The file name.
///
/// @param[out] m
///         The number of rows in the matrix.
///
/// @param[out] n
///         The number of columns in the matrix.
///
void read_raw_dimensions_from_file(char const *name, int *m, int *n);

///
/// @brief Reads a matrix from a file.
///
/// @param[in] name
///         The file name.
///
/// @param[in,out] helper
///         The initialization helper.
///
/// @return The matrix.
///
matrix_t read_raw_matrix_from_file(char const *name, init_helper_t helper);

///
/// @brief Reads a submatrix from a file.
///
/// @param[in] begin
///         The first row/column that belongs to the submatrix.
///
/// @param[in] end
///         The last row/column that belongs to the submatrix.
///
/// @param[in] name
///         The file name.
///
/// @param[in,out] helper
///         The initialization helper.
///
/// @return The submatrix.
///
matrix_t read_raw_sub_matrix_from_file(
    int begin, int end, char const *name, init_helper_t helper);

extern const struct hook_t store_raw_pencil;
extern const struct hook_descr_t default_store_raw_pencil_descr;

extern const struct hook_t store_raw_input_pencil;
extern const struct hook_descr_t default_store_raw_input_pencil_descr;

extern const struct hook_initializer_t mtx_initializer;
extern const struct hook_initializer_t raw_initializer;

#endif
