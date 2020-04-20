///
/// @file
///
/// @brief This file contains the combined shared memory interface functions.
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
#include <starneig/node.h>
#include <starneig/sep_sm.h>
#include <starneig/gep_sm.h>
#include <stdlib.h>

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_SM_Reduce(
    int n,
    double A[], int ldA,
    double Q[], int ldQ,
    double real[], double imag[],
    int (*predicate)(double real, double imag, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (n < 1)          return -1;
    if (A == NULL)      return -2;
    if (ldA < n)        return -3;
    if (Q == NULL)      return -4;
    if (ldQ < n)        return -5;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_error_t ret = STARNEIG_SUCCESS;
    int *_selected = NULL;

    ret = starneig_SEP_SM_Hessenberg(n, A, ldA, Q, ldQ);
    if (ret != STARNEIG_SUCCESS)
        goto cleanup;

    ret = starneig_SEP_SM_Schur(n, A, ldA, Q, ldQ, real, imag);
    if (ret != STARNEIG_SUCCESS)
        goto cleanup;

    if (predicate) {
        if (selected == NULL)
            selected = _selected = malloc(n*sizeof(int));

        ret = starneig_SEP_SM_Select(
            n, A, ldA, predicate, arg, selected, num_selected);
        if (ret != STARNEIG_SUCCESS)
            goto cleanup;

        ret = starneig_SEP_SM_ReorderSchur(
            n, selected, A, ldA, Q, ldQ, real, imag);
        if (ret != STARNEIG_SUCCESS)
            goto cleanup;
    }

cleanup:
    free(_selected);
    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_Reduce(
    int n,
    double A[], int ldA,
    double B[], int ldB,
    double Q[], int ldQ,
    double Z[], int ldZ,
    double real[], double imag[], double beta[],
    int (*predicate)(double real, double imag, double beta, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (n < 1)          return -1;
    if (A == NULL)      return -2;
    if (ldA < n)        return -3;
    if (B == NULL)      return -4;
    if (ldB < n)        return -5;
    if (Q == NULL)      return -6;
    if (ldQ < n)        return -7;
    if (Z == NULL)      return -8;
    if (ldZ < n)        return -9;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int ret = STARNEIG_SUCCESS;
    int *_selected = NULL;

    ret = starneig_GEP_SM_HessenbergTriangular(
        n, A, ldA, B, ldB, Q, ldQ, Z, ldZ);
    if (ret != STARNEIG_SUCCESS)
        goto cleanup;

    ret = starneig_GEP_SM_Schur(
        n, A, ldA, B, ldB, Q, ldQ, Z, ldZ, real, imag, beta);
    if (ret != STARNEIG_SUCCESS)
        goto cleanup;

    if (predicate) {
        if (selected == NULL)
            selected = _selected = malloc(n*sizeof(int));

        ret = starneig_GEP_SM_Select(
            n, A, ldA, B, ldB, predicate, arg, selected, num_selected);
        if (ret != STARNEIG_SUCCESS)
            goto cleanup;

        ret = starneig_GEP_SM_ReorderSchur(
            n, selected, A, ldA, B, ldB, Q, ldQ, Z, ldZ, real, imag, beta);
        if (ret != STARNEIG_SUCCESS)
            goto cleanup;
    }

cleanup:
    free(_selected);
    return ret;
}
