///
/// @brief This file contains the shared memory helper interface functions.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
/// @author Lars Karlsson (larsk@cs.umu.se), Umeå University
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
#include <starneig/sep_sm.h>
#include "node_internal.h"
#include "math.h"
#include <stddef.h>
#include <math.h>

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_SM_Select(
    int n,
    double S[], int ldS,
    int (*predicate)(double real, double imag, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (n < 1)              return -1;
    if (S == NULL)          return -2;
    if (ldS < n)            return -3;
    if (predicate == NULL)  return -4;
    if (selected == NULL)   return -6;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int _num_selected = 0;

    for (int i = 0; i < n; i++) {
        if (i+1 < n && S[(size_t)i*ldS+i+1] != 0.0) {
            double real1, imag1, real2, imag2;
            starneig_compute_complex_eigenvalue(
                ldS, 0, &S[i*ldS+i], NULL, &real1, &imag1, &real2, &imag2,
                NULL, NULL);

            if (predicate(real1, imag1, arg)) {
                selected[i] = 1;
                selected[i+1] = 1;
                _num_selected += 2;
            }
            else {
                selected[i] = 0;
                selected[i+1] = 0;
            }

            i++;
        }
        else {
            if (predicate(S[(size_t)i*ldS+i], 0.0, arg)) {
                selected[i] = 1;
                _num_selected++;
            }
            else {
                selected[i] = 0;
            }
        }
    }

    if (num_selected != NULL)
        *num_selected = _num_selected;

    return STARNEIG_SUCCESS;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_Select(
    int n,
    const double S[], int ldS,
    const double T[], int ldT,
    int (*predicate)(double real, double imag, double beta, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (n < 1)              return -1;
    if (S == NULL)          return -2;
    if (ldS < n)            return -3;
    if (T == NULL)          return -4;
    if (ldT < n)            return -5;
    if (predicate == NULL)  return -6;
    if (selected == NULL)   return -8;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int _num_selected = 0;

    for (int i = 0; i < n; i++) {
        if (i+1 < n && S[(size_t)i*ldS+i+1] != 0.0) {
            double real1, imag1, real2, imag2, beta1, beta2;
            starneig_compute_complex_eigenvalue(
                ldS, ldT, &S[i*ldS+i], &T[i*ldT+i],
                &real1, &imag1, &real2, &imag2, &beta1, &beta2);

            if (predicate(real1, imag1, beta1, arg)) {
                selected[i] = 1;
                selected[i+1] = 1;
                _num_selected += 2;
            }
            else {
                selected[i] = 0;
                selected[i+1] = 0;
            }

            i++;
        }
        else {
            if (predicate(S[(size_t)i*ldS+i], 0.0, T[(size_t)i*ldT+i], arg)) {
                selected[i] = 1;
                _num_selected++;
            }
            else {
                selected[i] = 0;
            }
        }
    }

    if (num_selected != NULL)
        *num_selected = _num_selected;

    return STARNEIG_SUCCESS;
}
