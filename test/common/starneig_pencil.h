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

#ifndef STARNEIG_TEST_COMMON_STARNEIG_PENCIL_H
#define STARNEIG_TEST_COMMON_STARNEIG_PENCIL_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>

#ifndef STARNEIG_ENABLE_MPI
#error "This should be included only when STARNEIG_ENABLE_MPI is defined."
#endif

#include "pencil.h"
#include <starneig/starneig.h>
#include <stddef.h>
#include <assert.h>

///
/// @brief Initializes a StarNEig matrix.
///
/// @param[in] m         The number of rows in the matrix.
/// @param[in] n         The number of columns in the matrix.
/// @param[in] bm        The number of rows in a distributed block.
/// @param[in] bn        The number of columns in a distributed block.
/// @param[in] dtype     The matrix element data type
/// @param[in] distr     The data distribution.
///
/// @return An initialized StarNEig matrix.
///
matrix_t init_starneig_matrix(
    int m, int n, int bm, int bn, data_type_t dtype, starneig_distr_t distr);

///
/// @brief Returns the number of rows in a given StarNEig matrix.
///
/// @param[in] matrix
///         StarNEig matrix.
///
/// @return The number of rows.
///
size_t STARNEIG_MATRIX_M(const matrix_t matrix);

///
/// @brief Returns the number of columns in a given StarNEig matrix.
///
/// @param[in] matrix
///         StarNEig matrix.
///
/// @return The number of columns.
///
size_t STARNEIG_MATRIX_N(const matrix_t matrix);

///
/// @brief TODO
///
/// @param[in] matrix
///         StarNEig matrix.
///
/// @return The number of rows in a distributed block.
///
size_t STARNEIG_MATRIX_BM(const matrix_t matrix);

///
/// @brief TODO
///
/// @param[in] matrix
///         StarNEig matrix.
///
/// @return The number of columns in a distributed block.
///
size_t STARNEIG_MATRIX_BN(const matrix_t matrix);

///
/// @brief Returns a handle to the given StarNEig matrix.
///
/// @param[in] matrix
///         StarNEig matrix.
///
/// @return A handle to the given StarNEig matrix.
///
starneig_distr_matrix_t STARNEIG_MATRIX_HANDLE(const matrix_t matrix);

starneig_distr_t STARNEIG_MATRIX_DISTR(const matrix_t matrix);

#ifdef STARNEIG_ENABLE_BLACS

void STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
    const matrix_t matrix, starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local);

#endif

extern struct pencil_handler starneig_handler;
extern struct pencil_handler blacs_handler;

///
/// @brief Fills the missing fields in an opaque matrix pencil object.
///
/// @param[in] pencil
///         The opaque matrix pencil object.
///
void fill_starneig_pencil(pencil_t pencil);

///
/// @brief Data distribution descriptor.
///
struct data_distr_t {
    char *name;                            ///< name
    char *desc;                            ///< description
    int (*func)(int, int, void *);         ///< N^2 -> N function that maps a
                                           ///< section to it's owner's MPI rank
};

void print_avail_data_distr();

struct data_distr_t const * read_data_distr(
    char const *name, int argc, char * const *argv, int *argr);

extern const struct hook_data_converter local_starneig_converter;
extern const struct hook_data_converter starneig_local_converter;
extern const struct hook_data_converter local_blacs_converter;
extern const struct hook_data_converter blacs_local_converter;
extern const struct hook_data_converter starneig_blacs_converter;
extern const struct hook_data_converter blacs_starneig_converter;

#endif
