///
/// @file
///
/// @brief This example demonstrates how to use the shared memory interface
/// functions with standard eigenvalue problems.
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

#include "validate.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <starneig/starneig.h>

// a predicate function that selects all eigenvalues that have positive a real
// part
static int predicate(double real, double imag, void *arg)
{
    if (0.0 < real)
        return 1;
    return 0;
}

int main()
{
    const int n = 3000; // matrix dimension

    srand((unsigned) time(NULL));

    // generate a full random matrix A and a copy C

    int ldA = ((n/8)+1)*8, ldC = ((n/8)+1)*8;
    double *A = malloc(n*ldA*sizeof(double));
    double *C = malloc(n*ldC*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A[j*ldA+i] = C[j*ldC+i] = 2.0*rand()/RAND_MAX - 1.0;

    // generate an identity matrix Q

    int ldQ = ((n/8)+1)*8;
    double *Q = malloc(n*ldA*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            Q[j*ldQ+i] = i == j ? 1.0 : 0.0;

    // allocate space for the eigenvalues and the eigenvalue selection vector

    double *real = malloc(n*sizeof(double));
    double *imag = malloc(n*sizeof(double));
    int *select = malloc(n*sizeof(int));

    // Initialize the StarNEig library using all available CPU cores and
    // GPUs. The STARNEIG_HINT_SM flag indicates that the library should
    // initialize itself for shared memory computations.

    starneig_node_init(STARNEIG_USE_ALL, STARNEIG_USE_ALL, STARNEIG_HINT_SM);

    // reduce the full matrix matrix A to upper Hessenberg form

    printf("Hessenberg reduction...\n");
    starneig_SEP_SM_Hessenberg(n, A, ldA, Q, ldQ);

    // reduce the upper Hessenberg matrix A to Schur form

    printf("Schur reduction...\n");
    starneig_SEP_SM_Schur(n, A, ldA, Q, ldQ, real, imag);

    // select eigenvalues that have positive a real part

    int num_selected;
    starneig_SEP_SM_Select(n, A, ldA, &predicate, NULL, select, &num_selected);
    printf("Selected %d eigenvalues out of %d.\n", num_selected, n);

    // reorder the selected eigenvalues to the upper left corner of the matrix A

    printf("Reordering...\n");
    starneig_SEP_SM_ReorderSchur(n, select, A, ldA, Q, ldQ, real, imag);

    // de-initialize the StarNEig library

    starneig_node_finalize();

    // check residual || Q A Q^T - C ||_F / || C ||_F

    check_residual(n, ldQ, ldA, ldQ, ldC, Q, A, Q, C);

    // check residual || Q Q^T - I ||_F / || I ||_F

    check_orthogonality(n, ldQ, Q);

    // cleanup

    free(A);
    free(C);
    free(Q);

    free(real);
    free(imag);
    free(select);

    return 0;
}
