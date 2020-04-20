///
/// @file
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
#include <starneig/sep_dm.h>
#include "../common/node_internal.h"

__attribute__ ((visibility ("default")))
starneig_error_t starneig_SEP_DM_Reduce(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t Q,
    double real[], double imag[],
    int (*predicate)(double real, double imag, void *arg),
    void *arg,
    int selected[],
    int *num_selected)
{
    if (A == NULL)  return -1;
    if (Q == NULL)  return -2;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    int ret = STARNEIG_SUCCESS;
    int *_selected = NULL;

    ret = starneig_SEP_DM_Hessenberg(A, Q);
    if (ret)
        goto cleanup;

    ret = starneig_SEP_DM_Schur(A, Q, real, imag);
    if (ret)
        goto cleanup;

    if (predicate) {
        if (selected == NULL)
            selected = _selected =
                malloc(starneig_distr_matrix_get_rows(A)*sizeof(int));

        ret = starneig_SEP_DM_Select(A, predicate, arg, selected, num_selected);
        if (ret)
            goto cleanup;

        ret = starneig_SEP_DM_ReorderSchur(selected, A, Q, real, imag);
        if (ret)
            goto cleanup;
    }

cleanup:
    free(_selected);
    return ret;
}
