///
/// @file
///
/// @brief This example demonstrates how to use the shared memory interface
/// functions with generalized eigenvalue problems.
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

// a predicate function that selects all finite eigenvalues that have positive
// a real part
static int predicate(double real, double imag, double beta, void *arg)
{
    if (0.0 < real && beta != 0.0)
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

    // generate a full random matrix B and a copy D

    int ldB = ((n/8)+1)*8, ldD = ((n/8)+1)*8;
    double *B = malloc(n*ldB*sizeof(double));
    double *D = malloc(n*ldD*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            B[j*ldB+i] = D[j*ldD+i] = 2.0*rand()/RAND_MAX - 1.0;

    // generate an identity matrix Q

    int ldQ = ((n/8)+1)*8;
    double *Q = malloc(n*ldA*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            Q[j*ldQ+i] = i == j ? 1.0 : 0.0;

    // generate an identity matrix Z

    int ldZ = ((n/8)+1)*8;
    double *Z = malloc(n*ldZ*sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            Z[j*ldZ+i] = i == j ? 1.0 : 0.0;

    double *X = NULL; int ldX = 0;

    // allocate space for the eigenvalues and the eigenvalue selection vector

    double *real = malloc(n*sizeof(double));
    double *imag = malloc(n*sizeof(double));
    double *beta = malloc(n*sizeof(double));
    int *select = malloc(n*sizeof(int));

    // Initialize the StarNEig library using all available CPU cores and
    // GPUs. The STARNEIG_HINT_SM flag indicates that the library should
    // initialize itself for shared memory computations.

    starneig_node_init(STARNEIG_USE_ALL, STARNEIG_USE_ALL, STARNEIG_HINT_SM);

    // reduce the dense-dense matrix pair (A,B) to generalized Schur form
    // (skip reordering)

    printf("Reduce...\n");
    starneig_GEP_SM_Reduce(
        n, A, ldA, B, ldB, Q, ldQ, Z, ldZ, real, imag, beta,
        NULL, NULL, NULL, NULL);

    // select eigenvalues that have positive a real part and allocate space for
    // the eigenvectors

    int num_selected;
    starneig_GEP_SM_Select(
        n, A, ldA, B, ldB, &predicate, NULL, select, &num_selected);
    printf("Selected %d eigenvalues out of %d.\n", num_selected, n);

    ldX = ((n/8)+1)*8;
    X = malloc(num_selected*ldX*sizeof(double));

    // compute a selected set of eigenvectors

    printf("Eigenvectors...\n");
    starneig_GEP_SM_Eigenvectors(n, select, A, ldA, B, ldB, Q, ldQ, X, ldX);

    // de-initialize the StarNEig library

    starneig_node_finalize();

    // check residual || Q A Z^T - C ||_F / || C ||_F

    check_residual(n, ldQ, ldA, ldZ, ldC, Q, A, Z, C);

    // check residual || Q B Z^T - D ||_F / || D ||_F

    check_residual(n, ldQ, ldB, ldZ, ldD, Q, B, Z, D);

    // check residual || Q Q^T - I ||_F / || I ||_F

    check_orthogonality(n, ldQ, Q);

    // check residual || Z Z^T - I ||_F / || I ||_F

    check_orthogonality(n, ldZ, Z);

    // cleanup

    free(A);
    free(C);
    free(B);
    free(D);
    free(Q);
    free(Z);
    free(X);

    free(real);
    free(imag);
    free(beta);
    free(select);

    return 0;
}
