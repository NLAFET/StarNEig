///
/// @file
///
/// @brief This file contains data types and functions for BLACS formatted
/// distributed matrices.
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

#ifndef STARNEIG_BLACS_MATRIX_H
#define STARNEIG_BLACS_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

#ifndef STARNEIG_ENABLE_MPI
#error "This header should be included only when STARNEIG_ENABLE_MPI is defined."
#endif

#ifndef STARNEIG_ENABLE_BLACS
#error "This header should be included only when STARNEIG_ENABLE_BLACS is defined."
#endif

#include <starneig/distr_matrix.h>

///
/// @defgroup starneig_dm_blacs ScaLAPACK compatibility / BLACS matrices
///
/// @brief Data types and functions for BLACS formatted distributed matrices.
///
/// @{
///

///
/// @name BLACS contexts
/// @{
///

///
/// @brief BLACS context.
///
typedef int starneig_blacs_context_t;

///
/// @brief Convers a data distribution to a BLACS context.
///
///  @attention The data distribution must describe a two-dimensional block
///  cyclic distribution.
///
/// @param[in] distr
///         The data distribution.
///
/// @return The BLACS context.
///
starneig_blacs_context_t starneig_distr_to_blacs_context(
    starneig_distr_t distr);

///
/// @brief Convers a BLACS context to a data distribution.
///
/// @param[in] context
///         The BLACS context.
///
/// @return The data distribution.
///
starneig_distr_t starneig_blacs_context_to_distr(
    starneig_blacs_context_t context);

///
/// @brief Checks whether a data distribution is BLACS compatible.
///
/// @param[in] distr
///          The data distribution.
///
/// @return Non-zero if the data distribution matrix is BLACS compatible.
///
int starneig_distr_is_blacs_compatible(starneig_distr_t distr);

///
/// @brief Checks whether a data distribution is compatible with a given BLACS
/// context.
///
/// @param[in] distr
///         The data distribution.
///
/// @param[in] context
///         The BLACS context.
///
/// @return Non-zero if the data distribution compatible with the BLACS
/// context.
///
int starneig_distr_is_compatible_with(
    starneig_distr_t distr, starneig_blacs_context_t context);

///
/// @}
///

///
/// @name BLACS descriptors
/// @{
///

///
/// @brief BLACS descriptor.
///
typedef struct starneig_blacs_descr {
    /// The descriptor type.
    int type;
    /// The related BLACS context.
    starneig_blacs_context_t context;
    /// The number of (global) rows in the matrix.
    int m;
    /// The number of (global) columns in the matrix.
    int n;
    /// The number of rows in a distribution block.
    int sm;
    /// The number of columns in a distribution block.
    int sn;
    /// The process grid row over which the first row is distributed.
    int rsrc;
    /// The process grid column over which the first column is distributed.
    int csrc;
    /// The leading dimension of the local array.
    int lld;
} starneig_blacs_descr_t;

///
/// @brief Creates a BLACS matrix with uninitialized matrix elements.
///
/// @param[in] rows
///         The number of (global) rows in the matrix.
///
/// @param[in] cols
///         The number of (global) columns in the matrix.
///
/// @param[in] row_blksz
///         The number of rows in a distribution block. Can be set to -1 in
///         which case the library decides the value.
///
/// @param[in] col_blksz
///         The number of columns in a distribution block. Can be set to -1 in
///         which case the library decides the value.
///
/// @param[in] type
///         The matrix element data type.
///
/// @param[in] context
///         The BLACS context.
///
/// @param[out] descr
///         The BLACS descriptor.
///
/// @param[out] local
///         A pointer to the local array.
///
void starneig_blacs_create_matrix(
    int rows, int cols, int row_blksz, int col_blksz, starneig_datatype_t type,
    starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local);

///
/// @brief Creates a BLACS matrix with uninitialized matrix elements.
/// Deprecated.
///
/// @deprecated The starneig_create_blacs_matrix() function has been replaced
/// with the starneig_blacs_create_matrix() function. This function will be
/// removed in a future release of the library.
///
void starneig_create_blacs_matrix(
    int rows, int cols, int row_blksz, int col_blksz, starneig_datatype_t type,
    starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local);

///
/// @brief Destroyes a BLACS matrix.
///
/// @param[in,out] descr
///         The BLACS descriptor.
///
/// @param[in,out] local
///         A pointer to the local array.
///
void starneig_blacs_destroy_matrix(starneig_blacs_descr_t *descr, void **local);

///
/// @brief Destroyes a BLACS matrix. Deprecated.
///
/// @deprecated The starneig_destroy_blacs_matrix() function has been replaced
/// with the starneig_blacs_destroy_matrix() function. This function will be
/// removed in a future release of the library.
///
void starneig_destroy_blacs_matrix(starneig_blacs_descr_t *descr, void **local);

///
/// @brief Convers a distributed matrix to a BLACS descriptor and a matching
/// local array.
///
/// This function creates a wrapper object. The contents of the distributed
/// matrix may be modified by the functions that use the wrapper object.
///
/// @code{.c}
/// starneig_distr_matrix_t dA = starneig_distr_matrix_create(...);
///
/// ...
///
/// starneig_distr_t distr = starneig_distr_matrix_get_distr(A);
/// starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);
///
/// starneig_blacs_descr_t descr_a;
/// double *local_a;
/// starneig_distr_matrix_to_blacs_descr(
///     dA, context, &descr_a, (void **)&local_a);
/// @endcode
///
/// @param[in] matrix
///         The distributed matrix.
///
/// @param[in] context
///         The BLACS context. The context must have been converted from the
///         same data distribution the distributed matrix is using or vice
///         versa.
///
/// @param[out] descr
///         The BLACS descriptor.
///
/// @param[out] local
///         A pointer to the local array.
///
void starneig_distr_matrix_to_blacs_descr(
    starneig_distr_matrix_t matrix, starneig_blacs_context_t context,
    starneig_blacs_descr_t *descr, void **local);

///
/// @brief Convers a BLACS descriptor and a matching local array to a
/// distributed matrix.
///
/// This function creates a wrapper object. The contents of the local array may
/// be modified by the functions that use the wrapper object. The
/// starneig_distr_matrix_destroy() function does not de-initilize the BLACS
/// descriptor nor free the local array.
///
/// @code{.c}
/// starneig_blacs_context_t context;
/// starneig_blacs_descr_t descr_a;
/// double *local_a;
///
/// ...
///
/// starneig_distr_t distr = starneig_blacs_context_to_distr(context);
/// starneig_distr_matrix_t dA = starneig_blacs_descr_to_distr_matrix(
///     STARNEIG_REAL_DOUBLE, distr, descr_a, (void *)local_a);
/// @endcode
///
/// @param[in] type
///         The matrix element data type.
///
/// @param[in] distr
///         The data distribution. The data distribution must have been
///         converted from the same BLACS context the BLACS descriptor is using
///         or vice versa.
///
/// @param[in] descr
///         The BLACS descriptor.
///
/// @param[in] local
///         A pointer to the local array.
///
/// @return The distributed matrix.
///
starneig_distr_matrix_t starneig_blacs_descr_to_distr_matrix(
    starneig_datatype_t type, starneig_distr_t distr,
    starneig_blacs_descr_t *descr, void *local);

///
/// @brief Checks whether a distributed matrix is BLACS compatible.
///
/// @param[in] matrix
///         The distributed matrix.
///
/// @return Non-zero if the distributed matrix is BLACS compatible.
///
int starneig_distr_matrix_is_blacs_compatible(starneig_distr_matrix_t matrix);

///
/// @brief Checks whether a distributed matrix is compatible with a given BLACS
/// context.
///
/// @param[in] matrix
///         The distributed matrix.
///
/// @param[in] context
///         The BLACS context.
///
/// @return Non-zero if the distributed matrix compatible with the BLACS
/// context.
///
int starneig_distr_matrix_is_compatible_with(
    starneig_distr_matrix_t matrix, starneig_blacs_context_t context);

///
/// @}
///

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_BLACS_MATRIX_H
