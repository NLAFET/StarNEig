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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef STARNEIG_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

static unsigned long seed = 2019;
static int pinning = 1;

void init_prand(unsigned int _seed)
{
    seed = _seed;
}

int prand()
{
    return (seed = ((seed * 1103515245) + 12345) & 0x7fffffff);
}

void print_matrix(int m, int n, int ld, double const *mat)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if(mat[(size_t)j*ld+i] != 0.0)
                printf("%10f ", mat[(size_t)j*ld+i]);
            else
                printf("  -------- ");
        }
        printf("\n");
    }
    printf("\n");
}

void compare_print_matrix(
    double eps, int m, int n, int ld, int ldr,
    double const *mat, double const *ref)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if(eps < fabs(mat[(size_t)j*ld+i]-ref[(size_t)j*ldr+i]))
                printf("%10f ", mat[(size_t)j*ld+i]);
            else
                printf("  -------- ");
        }
        printf("\n");
    }
    printf("\n");
}

void set_pinning(int value)
{
    pinning = value;
}

void * alloc_matrix(
    int m, int n, size_t elemsize, size_t *ld)
{
    *ld = divceil(m, 64/elemsize)*(64/elemsize);
    void *ptr;
#ifdef STARNEIG_ENABLE_CUDA
    if (pinning)
        cudaHostAlloc(&ptr, n*(*ld)*elemsize, cudaHostRegisterPortable);
    else
        ptr = aligned_alloc(64, n*(*ld)*elemsize);
#else
    ptr = aligned_alloc(64, n*(*ld)*elemsize);
#endif
    return ptr;
}

void free_matrix(void *matrix)
{
#ifdef STARNEIG_ENABLE_CUDA
    if (pinning)
        cudaFree(matrix);
    else
        free(matrix);
#else
    free(matrix);
#endif
}

void copy_matrix(
    int m, int n, size_t lds, size_t ldd, size_t elemsize,
    void const * restrict source,  void * restrict dest)
{
    for (int i = 0; i < n; i++)
        memcpy(dest+i*ldd*elemsize, source+i*lds*elemsize, m*elemsize);
}
