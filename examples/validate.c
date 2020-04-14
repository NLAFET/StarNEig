///
/// @file
///
/// @brief This file contains code that is used to validate the output of the
/// example codes.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

static inline double squ(double x)
{
    return x*x;
}

void check_orthogonality(int n, size_t ldQ, double const *Q)
{
    size_t ldT = ((n/8)+1)*8;
    double *T = malloc(n*ldT*sizeof(double));

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0,
        Q, ldQ, Q, ldQ, 0.0, T, ldT);

    double dot = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dot += squ(T[i*ldT+j] - (i == j ? 1.0 : 0.0));

    double norm = ((long long)1<<52) * sqrt(dot)/sqrt(n);

    if (norm < 1000) {
        printf("The matrix is orthogonal.\n");
    }
    else {
        fprintf(stderr, "Matrix is not orthogonal.\n");
        exit(EXIT_FAILURE);
    }

    free(T);
}

void check_residual(
    int n, size_t ldQ, size_t ldA, size_t ldZ, size_t ldC,
    double const *Q, double const *A, double const *Z, double const *C)
{
    size_t ldT = ((n/8)+1)*8;
    double *T = malloc(n*ldT*sizeof(double));

    size_t ldY = ((n/8)+1)*8;
    double *Y = malloc(n*ldT*sizeof(double));

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
        Q, ldQ, A, ldA, 0.0, T, ldT);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0,
        T, ldT, Z, ldZ, 0.0, Y, ldY);

    double dot = 0.0, a_dot = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dot += squ(Y[i*ldY+j] - C[i*ldC+j]);
            a_dot += squ(C[i*ldC+j]);
        }
    }

    double norm = ((long long)1<<52) * sqrt(dot)/sqrt(a_dot);

    if (norm < 1000) {
        printf("The residual is small enough.\n");
    }
    else {
        fprintf(stderr, "The residual is too large.\n");
        exit(EXIT_FAILURE);
    }

    free(T);
    free(Y);
}
