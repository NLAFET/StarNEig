///
/// @file This file contains code that places a set of 1-by-1 and 2-by-2 blocks
/// to the diagonal of a matrix A or a matrix pencil (A,B).
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
#include "block_placer.h"
#include "crawler.h"

struct block_placer_arg {
    double *real;
    double *imag;
    double *beta;
};

///
/// @brief A crawler function that places the 1-by-1 and 2-by-2 blocks to the
/// diagonal.
///
static int crawler(
    int offset, int size, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    double *real = ((struct block_placer_arg *)arg)->real;
    double *imag = ((struct block_placer_arg *)arg)->imag;
    double *beta = ((struct block_placer_arg *)arg)->beta;

    double *A = ptrs[0];
    size_t ldA = lds[0];

    double *B = NULL;
    size_t ldB = 0;
    if (1 < count) {
        B = ptrs[1];
        ldB = lds[1];
    }

    int i = 0;
    int _size = offset+size < n ? size-1 : size;
    while (i < _size) {
        if (imag[offset+i] != 0.0) {
            A[    i * ldA + i] = real[offset+i];
            A[(i+1) * ldA + i+1] = real[offset+i+1];
            A[(i+1) * ldA + i] = imag[offset+i];
            A[    i * ldA + i+1] = imag[offset+i+1];

            if (B != NULL) {
                B[    i * ldB + i] = beta[offset+i];
                B[(i+1) * ldB + i+1] = beta[offset+i];
                B[(i+1) * ldB + i] = 0.0;
                B[    i * ldB + i+1] = 0.0;
            }

            i += 2;
        }
        else {
            A[i*ldA+i] = real[offset+i];
            if (B != NULL)
                B[i*ldB+i] = beta[offset+i];
            i++;
        }
    }

    return i;
}

void block_placer(
    double *real, double *imag, double *beta, matrix_t A, matrix_t B)
{
    struct block_placer_arg arg = {
        .real = real,
        .imag = imag,
        .beta = beta
    };

    crawl_matrices(
        CRAWLER_RW, CRAWLER_DIAG_WINDOW, &crawler, &arg, 0, A, B, NULL);
}
