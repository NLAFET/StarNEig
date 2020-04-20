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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include "sanity.h"
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

static int messages = 0;
static int verbose = 0;

#ifdef STARNEIG_ENABLE_CUDA

static int pinning = 1;

__attribute__ ((visibility ("default")))
void starneig_node_enable_pinning()
{
    pinning = 1;
}

__attribute__ ((visibility ("default")))
void starneig_node_disable_pinning()
{
    pinning = 0;
}

#endif

void starneig_set_message_mode(int _messages, int _verbose)
{
    messages = _messages;
    verbose = _verbose;
}

#ifdef STARNEIG_ENABLE_VERBOSE

void starneig_verbose_begin(char const *msg, ...)
{
    if (!verbose) return;
    fprintf(stdout, "[starneig][verbose] ");
    va_list args;
    va_start(args, msg);
    vfprintf(stdout, msg, args);
    va_end(args);
    fflush(stdout);
}

void starneig_verbose_cont(char const *msg, ...)
{
    if (!verbose) return;
    va_list args;
    va_start(args, msg);
    vfprintf(stdout, msg, args);
    va_end(args);
    fflush(stdout);
}

void starneig_verbose(char const *msg, ...)
{
    if (!verbose) return;
    fprintf(stdout, "[starneig][verbose] ");
    va_list args;
    va_start(args, msg);
    vfprintf(stdout, msg, args);
    va_end(args);
    fprintf(stdout, "\n");
}

#endif // STARNEIG_ENABLE_VERBOSE

#ifdef STARNEIG_ENABLE_MESSAGES

void starneig_message(char const *msg, ...)
{
    if (!messages) return;
    fprintf(stdout, "[starneig][message] ");
    va_list args;
    va_start(args, msg);
    vfprintf(stdout, msg, args);
    va_end(args);
    fprintf(stdout, "\n");
}

#endif // STARNEIG_ENABLE_MESSAGES

void starneig_warning(char const *msg, ...)
{
    fprintf(stderr, "[starneig][warning] ");
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void starneig_error(char const *msg, ...)
{
    fprintf(stderr, "[starneig][error] ");
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void starneig_fatal_error(char const *msg, ...)
{
    fprintf(stderr, "[starneig][fatal error] ");
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

void * starneig_alloc_matrix(int m, int n, size_t elemsize, size_t *ld)
{
    STARNEIG_ASSERT_MSG(0 < m && 0 < n && 0 < elemsize, "Invalid dimensions.");
    STARNEIG_ASSERT_MSG(ld != NULL, "NULL pointer.");

    *ld = divceil(m, 64/elemsize)*(64/elemsize);
#ifdef ALIGNED_ALLOC_FOUND
    void *ptr = aligned_alloc(64, n*(*ld)*elemsize);
#else
    void *ptr = malloc(n*(*ld)*elemsize);
#endif

    if (ptr == NULL)
        starneig_fatal_error("starneig_alloc_matrix failed.");

    return ptr;
}

void starneig_free_matrix(void *matrix)
{
    free(matrix);
}

void * starneig_alloc_pinned_matrix(int m, int n, size_t elemsize, size_t *ld)
{
#ifdef STARNEIG_ENABLE_CUDA
    if (pinning) {
        STARNEIG_ASSERT_MSG(
            0 < m && 0 < n && 0 < elemsize, "Invalid dimensions.");
        STARNEIG_ASSERT_MSG(ld != NULL, "NULL pointer.");

        *ld = divceil(m, 64/elemsize)*(64/elemsize);
        void *ptr;
        cudaError_t ret =
            cudaHostAlloc(&ptr, n*(*ld)*elemsize, cudaHostRegisterPortable);
        if (ret != cudaSuccess || ptr == NULL)
            starneig_fatal_error("cudaHostAlloc failed.");

        return ptr;
    }
#endif
    return starneig_alloc_matrix(m, n, elemsize, ld);
}

void starneig_free_pinned_matrix(void *matrix)
{
#ifdef STARNEIG_ENABLE_CUDA
    if (pinning) {
        cudaFree(matrix);
        return;
    }
#endif
    starneig_free_matrix(matrix);
}

void starneig_copy_matrix(
    int m, int n, size_t ldA, size_t ldB, size_t elemsize,
    void const *A, void *B)
{
    STARNEIG_ASSERT_MSG(0 < m && 0 < n && 0 < elemsize, "Invalid dimensions.");
    STARNEIG_ASSERT_MSG(m <= ldA && m <= ldB, "Invalid leading dimensions.");
    STARNEIG_SANITY_CHECK_INF(0, m, 0, n, ldA, A, "A");

    for (int i = 0; i < n; i++)
        memcpy(B+i*ldB*elemsize, A+i*ldA*elemsize, m*elemsize);
}

int starneig_mpi_get_comm_rank()
{
#ifdef STARNEIG_ENABLE_MPI
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);

    int my_rank = 0;
    if (mpi_initialized)
        starpu_mpi_comm_rank(starneig_mpi_get_comm(), &my_rank);

    return my_rank;
#else
    return 0;
#endif
}

int starneig_mpi_get_comm_size()
{
#ifdef STARNEIG_ENABLE_MPI
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);

    int comm_size = 1;
    if (mpi_initialized)
        starpu_mpi_comm_size(starneig_mpi_get_comm(), &comm_size);

    return comm_size;
#else
    return 1;
#endif
}
