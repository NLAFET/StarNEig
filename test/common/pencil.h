///
/// @file This file contains definitions of an opaque matrix object and an
/// opaque matrix pencil.
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

#ifndef STARNEIG_TEST_COMMON_PENCIL_H
#define STARNEIG_TEST_COMMON_PENCIL_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "data.h"
#include "supplementary.h"
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_matrix.h>
#endif
#include <stddef.h>
#include <stdio.h>

///
/// @brief Matrix type enumerator.
///
typedef enum {
    LOCAL_MATRIX,          ///< Local matrix.
    STARNEIG_MATRIX,       ///< StarNEig matrix.
    BLACS_MATRIX           ///< BLACS compatible StarNEig matrix.
} matrix_type_t;

///
/// @brief Opaque matrix descriptor.
///
struct matrix {
    matrix_type_t type;
    data_type_t dtype;
    void *ptr;
};

///
/// @brief Opaque matrix object.
///
typedef struct matrix * matrix_t;

///
/// @brief Matrix copy function data type.
///
typedef void * (*matrix_copy_t)(void const *);

///
/// @brief Matrix free function data type.
///
typedef void (*matrix_free_t)(void *);

///
/// @brief Matrix handler object.
///
struct pencil_handler {
    matrix_type_t type;
    void* (*copy)(void const *);
    void (*free)(void *);
    size_t (*get_rows)(matrix_t matrix);
    size_t (*get_cols)(matrix_t matrix);
    void (*gemm)(
        char const *trans_a, char const *trans_b, double alpha,
        const matrix_t mat_a, const matrix_t mat_b, double beta,
        matrix_t *mat_c);
};

///
/// @brief Frees a previously allocated opaque matrix object.
///
/// @param[in] matrix
///         The opaque matrix object.
///
void free_matrix_descr(matrix_t matrix);

///
/// @brief Copies a previously allocated opaque matrix object.
///
/// @param[in] matrix
///         The opaque matrix object.
///
/// @return A copy of the opaque matrix object.
///
matrix_t copy_matrix_descr(const matrix_t matrix);

///
/// @brief Returns the row count of a given opaque matrix object.
///
/// @param[in] matrix
///         The opaque matrix object.
///
/// @return row count
///
size_t GENERIC_MATRIX_M(const matrix_t matrix);

///
/// @brief Return the column count of a given opaque matrix object.
///
/// @param[in] matrix
///         The opaque matrix object.
///
/// @return column count
///
size_t GENERIC_MATRIX_N(const matrix_t matrix);

///
/// @brief Computes C = alpha * op(A) op(B) + beta * C
///
/// @param[in] trans_a
///         "T" => op(A) = A^T, "N" => op(A) = A
///
/// @param[in] trans_b
///         "T" => op(B) = B^T, "N" => op(B) = B
///
/// @param[in] alpha
///         Alpha.
///
/// @param[in] mat_a
///         Matrix A.
///
/// @param[in] mat_b
///         Matrix B.
///
/// @param[in] beta
///         Beta.
///
/// @param[in,out] mat_c
///         Matrix C.
///
void mul_C_AB(
    char const *trans_a, char const *trans_b, double alpha,
    const matrix_t mat_a, const matrix_t mat_b, double beta, matrix_t *mat_c);

///
/// @brief Computes the Frobenius norm of a matrix.
///
/// @param[in] mat_c
///         The matrix.
///
/// @return The requested norm.
///
double norm_C(const matrix_t mat_c);

///
/// @brief Prints an opaque matrix object.
///
/// @param[in] matrix
///         The matrix.
///
/// @param[in,out] stream
///         The printout stream.
///
void print_matrix_descr(const matrix_t matrix, FILE * stream);

///
/// @brief Opaque matrix pencil descriptor.
///
struct pencil {
    matrix_t mat_a;              ///< A matrix
    matrix_t mat_b;              ///< B matrix
    matrix_t mat_q;              ///< Q matrix
    matrix_t mat_z;              ///< Z matrix
    matrix_t mat_x;              ///< X matrix (eigenvectors)
    matrix_t mat_ca;             ///< original A matrix
    matrix_t mat_cb;             ///< original B matrix
    struct supplementary *supp;  ///< supplementary data
};

///
/// @brief Opaque matrix pencil object.
///
typedef struct pencil * pencil_t;

///
/// @brief Initialized an opaque matrix pencil object.
///
/// @return A initialized opaque matrix pencil object.
///
pencil_t init_pencil();

///
/// @brief Frees a previously allocated opaque matrix pencil object.
///
/// @param[in] pencil
///         The opaque matrix pencil object.
///
void free_pencil(pencil_t pencil);

///
/// @brief Copies a previously allocated opaque matrix pencil object.
///
/// @param[in] pencil
///         The opaque matrix pencil object.
///
/// @return A copy of the opaque matrix pencil object.
///
pencil_t copy_pencil(const pencil_t pencil);

///
/// @brief Fills the missing fields in an opaque matrix pencil object.
///
/// @param[in] pencil
///         The opaque matrix pencil object.
///
void fill_pencil(pencil_t pencil);

#endif // STARNEIG_TEST_COMMON_PENCIL_H
