///
/// @file
///
/// @brief This file contains code that is shared among all components of the
/// library.
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

#ifndef STARNEIG_COMMON_COMMON_H
#define STARNEIG_COMMON_COMMON_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define STARNEIG_ASSERT(cond) \
    if (!(cond)) { \
        fprintf(stderr, "[starneig][assert] %s:%d\n", \
            __FILE__, __LINE__); \
        abort(); \
    }

#define STARNEIG_ASSERT_MSG(cond, message) \
    if (!(cond)) { \
        fprintf(stderr, "[starneig][assert] %s:%d: %s\n", \
            __FILE__, __LINE__, message); \
        abort(); \
    }

///
/// @brief Sets the library messaging mode.
///
/// @param[in] messages
///         If non-zero, some messages are printed to stdout.
///
/// @param[in] verbose
///         If non-zero, verbose messages are printed to stdout.
///
void starneig_set_message_mode(int messages, int verbose);

#ifdef STARNEIG_ENABLE_VERBOSE

///
/// @brief Begins a verbose message.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_verbose_begin(char const *msg, ...);

///
/// @brief Continues a verbose message.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_verbose_cont(char const *msg, ...);

///
/// @brief Prints a verbose message.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_verbose(char const *msg, ...);

#else

static inline void starneig_verbose_begin(char const *msg, ...) {}
static inline void starneig_verbose_cont(char const *msg, ...) {}
static inline void starneig_verbose(char const *msg, ...) {}

#endif // STARNEIG_ENABLE_VERBOSE

#ifdef STARNEIG_ENABLE_MESSAGES

///
/// @brief Prints a message.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_message(char const *msg, ...);

#else

static inline void starneig_message(char const *msg, ...) {}

#endif // STARNEIG_ENABLE_MESSAGES

///
/// @brief Prints a warning message.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_warning(char const *msg, ...);

///
/// @brief Prints an error message.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_error(char const *msg, ...);

///
/// @brief Prints an error message and aborts the program.
///
/// @param[in] msg
///         C string that contains the text to be written to stdout.
///
/// @param[in] ...
///         Additional arguments.
///
void starneig_fatal_error(char const *msg, ...);

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
/// @prief Return the ceil of a/b.
///
/// @param[in] a
///         Denominator.
///
/// @param[in] b
///         Numerator.
///
/// @return Ceil of a/b.
///
static inline int divceil(int a, int b)
{
    if ((a < 0 && b < 0) || (0 <= a && 0 <= b))
        return (abs(a)+abs(b)-1)/abs(b);
    else
        return a/b;
}

///
/// @prief Return the floor of a/b.
///
/// @param[in] a
///         Denominator.
///
/// @param[in] b
///         Numerator.
///
/// @return Floor of a/b.
///
static inline int divfloor(int a, int b)
{
    if ((a < 0 && b < 0) || (0 <= a && 0 <= b))
        return a/b;
    else
        return -(abs(a)+abs(b)-1)/abs(b);
}

///
/// @brief Allocates a matrix.
///
/// @param[in] m
///         The number of rows in the matrix.
///
/// @param[in] n
///         The number of columns in the matrix.
///
/// @param[in] elemsize
///         The matrix element size.
///
/// @param[out] ld
///         Returns the leading dimension of the matrix.
///
/// @return Pointer to the allocated matrix.
///
void * starneig_alloc_matrix(
    int m, int n, size_t elemsize, size_t *ld);
///
/// @brief Frees a matrix.
///
/// @param[in] A
///         The matrix.
///
void starneig_free_matrix(void *A);

///
/// @brief Allocates a matrix using pinned memory.
///
/// @param[in] m
///         The number of rows in the matrix.
///
/// @param[in] n
///         The number of columns in the matrix.
///
/// @param[in] elemsize
///         The matrix element size.
///
/// @param[out] ld
///         Returns the leading dimension of the matrix.
///
/// @return Pointer to the allocated matrix.
///
void * starneig_alloc_pinned_matrix(
    int m, int n, size_t elemsize, size_t *ld);
///
/// @brief Frees a pinned matrix.
///
/// @param[in] A
///         The matrix.
///
void starneig_free_pinned_matrix(void *A);

///
/// @brief Copies a matrix.
///
/// @param[in] m
///         The number of rows in the matrix.
///
/// @param[in] n
///         The number of columns in the matrix.
///
/// @param[in] ldA
///         The leading dimension of the source matrix.
///
/// @param[in] ldB
///         The leading dimension of the destination matrix.
///
/// @param[in] elemsize
///         The matrix element size.
///
/// @param[in] A
///         The source matrix.
///
/// @param[out] B
///         The destination matrix.
///
void starneig_copy_matrix(
    int m, int n, size_t ldA, size_t ldB, size_t elemsize,
    void const *A, void *B);

#ifdef STARNEIG_ENABLE_MPI

///
/// @brief MPI info struct.
///
struct mpi_info {
    unsigned tag_offset;    ///< Data handle tag offset.
};

///
/// @brief MPI info.
///
typedef struct mpi_info * mpi_info_t;

#else

typedef void * mpi_info_t;

#endif // STARNEIG_ENABLE_MPI

///
/// @brief Returns the rank of the calling process.
///
/// @return The rank of the calling process.
///
int starneig_mpi_get_comm_rank();

///
/// @brief Returns the total number of processies.
///
/// @return The total number of processies.
///
int starneig_mpi_get_comm_size();

#endif
