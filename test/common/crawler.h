///
/// @file This file contains a matrix crawler object that is used to initialize
/// an validate the input and output matrices
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

#ifndef STARNEIG_TEST_COMMON_CRAWLER_H
#define STARNEIG_TEST_COMMON_CRAWLER_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "pencil.h"
#include <stdarg.h>

///
/// @brief Access mode.
///
typedef enum {
    CRAWLER_R, CRAWLER_W, CRAWLER_RW
} crawler_access_t;

///
/// @brief Crawling mode.
///
typedef enum {
    CRAWLER_PANEL,       ///< Panel-based crawling mode.
    CRAWLER_HPANEL,      ///< Horizontal panel based crawling mode.
    CRAWLER_DIAG_WINDOW  ///< Diagonal window based crawling mode.
} crawler_mode_t;

///
/// @brief Crawler function data type.
///
/// @param[in] offset
///         The offset from the beginning of the matrix.
///
/// @param[in] size
///         The size of the current panel / diagonal window.
///
/// @param[in] m
///         The number of rows in the matrix.
///
/// @param[in] n
///         The number of rows in the matrix.
///
/// @param[in] count
///         The number of matrices to crawl.
///
/// @param[in] lds
///         The leading dimensions of the local arrays.
///
/// @param[in] ptrs
///         Pointers to the local arrays.
///
/// @param[in] arg
///         An optional argument.
///
/// @return The last processed row/column + 1.
///
typedef int (*crawler_func_t)(
    int offset, int size, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg);

///
/// @brief Craws a set of matrices.
///
/// @param[in] access
///         The access mode.
///
/// @param[in] mode
///         The crawling mode.
///
/// @param[in] func
///         The crawler function.
///
/// @param[in] arg
///         An optional argument for the crawler function.
///
/// @param[in] arg_size
///         The size of the optional crawler function argument.
///
/// @param[in,out] ...
///         A list of matrices followed by a 0.
///
void crawl_matrices(
    crawler_access_t access, crawler_mode_t mode,
    crawler_func_t func, void *arg, size_t arg_size, ...);

#endif // STARNEIG_TEST_COMMON_CRAWLER_H
