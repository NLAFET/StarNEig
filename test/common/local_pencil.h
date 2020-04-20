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

#ifndef STARNEIG_TEST_COMMON_LOCAL_PENCIL_H
#define STARNEIG_TEST_COMMON_LOCAL_PENCIL_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "pencil.h"
#include <stddef.h>
#include <assert.h>

///
/// @brief Initializes a local matrix.
///
/// @param[in] m         The number of rows in the matrix.
/// @param[in] n         The number of columns in the matrix.
/// @param[in] dtype     The matrix element data type
///
/// @return An initialized local matrix.
///
matrix_t init_local_matrix(int m, int n, data_type_t dtype);

///
/// @brief Returns the first element of a given local matrix.
///
/// @param[in] descr  local matrix descriptor
///
/// @return pointer to the first element
///
void* LOCAL_MATRIX_PTR(const matrix_t descr);

///
/// @brief Returns the row count of a given local matrix.
///
/// @param[in] descr  local matrix descriptor
///
/// @return row count
///
size_t LOCAL_MATRIX_M(const matrix_t descr);

///
/// @brief Return the column count of a given local matrix.
///
/// @param[in] descr  local matrix descriptor
///
/// @return column count
///
size_t LOCAL_MATRIX_N(const matrix_t descr);

///
/// @brief Returns the leading dimension of a given local matrix.
///
/// @param[in] descr  local matrix descriptor
///
/// @return leading dimension
///
size_t LOCAL_MATRIX_LD(const matrix_t descr);

extern struct pencil_handler local_handler;

///
/// @brief Fills the missing fields in an opaque matrix pencil object.
///
/// @param[in] pencil
///         The opaque matrix pencil object.
///
void fill_local_pencil(pencil_t pencil);

#endif
