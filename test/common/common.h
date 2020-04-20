///
/// @file This file contains auxiliary subroutines that are used throughout the
/// test program.
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

#ifndef STARNEIG_TEST_COMMON_COMMON_H
#define STARNEIG_TEST_COMMON_COMMON_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

///
/// @prief Returns the largest of two values.
///
/// @param[in] x
///         First value.
///
/// @param[in] y
///         Second value.
///
/// @return Largest of the two values.
///
#define MAX(x, y) ((x) > (y) ? (x) : (y))

///
/// @prief Returns the smallest of two values.
///
/// @param[in] x
///         First value.
///
/// @param[in] y
///         Second value.
///
/// @return Smallest of the two values.
///
#define MIN(x, y) ((x) < (y) ? (x) : (y))

///
/// Return the ceil of a/b
///
/// @param[in] a
///         The denominator.
///
/// @param[in] b
///         The numerator.
///
/// @return The ceil of a/b.
///
static inline int divceil(int a, int b)
{
    return (a+b-1)/b;
}

///
/// @brief Computes square of a floating-point number.
///
/// @param[in] x
///         The argument.
///
/// @return The square of the argument.
///
static inline double squ(double x)
{
    return x*x;
}

#define PRAND_MAX ((int)0x7fffffff)

///
/// @brief Initialized the internal pseudo randon number generator.
///
/// @param[in] seed
///         Seed.
///
void init_prand(unsigned int seed);

///
/// @brief Generates a randon number.
///
/// @return Random number.
///
int prand();

///
/// @brief Wrapper for BLAS DGEMM subroutine.
///
static inline void dgemm(char const *transa, char const *transb,
    int m, int n, int k, double alpha, double const *a, int lda,
    double const *b, int ldb, double beta, double *c, int ldc)
{
    extern void dgemm_(char const *, char const *, int const *, int const *,
        int const *, double const *, double const *, int const *,
        double const *, int const *, double const *, double*, int const *);

    dgemm_(transa, transb, &m, &n, &k,
        &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

///
/// @brief Prints a matrix.
///
/// @param[in] m
///         The row count.
///
/// @param[in] n
///         The column count.
///
/// @param[in] ld
///         The leading dimension.
///
/// @param[in] mat
///         A pointer to the matrix.
///
void print_matrix(int m, int n, int ld, double const *mat);

///
/// @brief Prints those matrix elements that differ from a reference matrix.
///
/// @param[in] eps
///         The threshold value.
///
/// @param[in] m
///         The row count.
///
/// @param[in] n
///         The column count.
///
/// @param[in] ld
///         The leading dimension.
///
/// @param[in] ldr
///         The reference matrix leading dimension.
///
/// @param[in] mat
///         A pointer to the matrix.
///
/// @param[in] ref
///         A pointer to the reference matrix.
///
void compare_print_matrix(
    double eps, int m, int n, int ld, int ldr,
    double const *mat, double const *ref);

void set_pinning(int value);

///
/// @brief Allocated a matrix (two-dimensional array).
///
/// @param[in] m
///         The row count.
///
/// @param[in] n
///         The column count.
///
/// @param[in]  elemsize
///         The data type size (sizeof(double), etc).
///
/// @param[out] ld
///         Returns the leading dimensions.
///
/// @return A pointer to the allocated matrix.
///
void * alloc_matrix(
    int m, int n, size_t elemsize, size_t *ld);

///
/// @brief Frees a matrix.
///
/// @param[in] matrix
///         A pointer to the destination matrix.
///
void free_matrix(void *matrix);

///
/// @brief Copies a matrix.
///
/// @param[in] m
///         The row count.
///
/// @param[in] n
///         The column count.
///
/// @param[in] lds
///         The source leading dimension.
///
/// @param[in] ldd
///         The destination leading dimension.
///
/// @param[in] elemsize
///         The data type size (sizeof(double), etc).
///
/// @param[in] source
///         A pointer to the source matrix.
///
/// @param[in] dest
///         A pointer to the destination matrix.
///
void copy_matrix(
    int m, int n, size_t lds, size_t ldd, size_t elemsize,
    void const * source, void * dest);

#endif
