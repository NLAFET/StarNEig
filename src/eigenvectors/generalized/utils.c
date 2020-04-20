///
/// @file
///
/// @brief Auxiliary subroutines for initialising and printing matrices
///
/// @author Carl Christian K. Mikkelsen (spock@cs.umu.se), Umeå University
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
#include "utils.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// This macro ensures that addresses are computed as size_t
#define _A(i, j) a[(size_t)(j)*lda+(i)]

void starneig_eigvec_gen_ddm(
    int m, int n, double *a, size_t lda, char *format)
{
    for (int i = 0; i < MIN(m, maxrow); i++) {
        for (int j = 0; j < MIN(n, maxcol); j++) {
            printf(format, _A(i,j));
        }
        printf("\n");
    }
    printf("\n");
}

void starneig_eigvec_gen_ddmi(
    int m, int n, int *a, size_t lda, char *format)
{
    for (int i = 0; i < MIN(m, maxrow); i++) {
        for (int j = 0; j < MIN(n, maxcol); j++) {
            printf(format, _A(i,j));
        }
        printf("\n");
    }
    printf("\n");
}

void starneig_eigvec_gen_zeros(int m, int n, double *a, size_t lda)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            _A(i, j) = 0.0;
}

void starneig_eigvec_gen_ones(int m, int n, double *a, size_t lda)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            _A(i, j) = 1.0;
}

#undef _A
