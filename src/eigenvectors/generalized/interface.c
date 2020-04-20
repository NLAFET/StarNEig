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
#include "sirobust-geig.h"
#include "robust.h"
#include "../common/common.h"
#include "../common/node_internal.h"
#include <starneig/gep_sm.h>
#include <cblas.h>
#include <stdlib.h>
#include <starpu.h>

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_Eigenvectors_expert(
    struct starneig_eigenvectors_conf *_conf,
    int n,
    int selected[],
    double S[], int ldS,
    double T[], int ldT,
    double Z[], int ldZ,
    double X[], int ldX)
{
    if (n < 1)              return -2;
    if (selected == NULL)   return -3;
    if (S == NULL)          return -4;
    if (ldS < n)            return -5;
    if (T == NULL)          return -6;
    if (ldT < n)            return -7;
    if (Z == NULL)          return -8;
    if (ldZ < n)            return -9;
    if (X == NULL)          return -10;
    if (ldX < n)            return -11;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    starneig_error_t ret = STARNEIG_SUCCESS;
    double *_X = NULL; int ld_X;

    // use default configuration if necessary
    struct starneig_eigenvectors_conf *conf;
    struct starneig_eigenvectors_conf local_conf;
    if (_conf == NULL)
        starneig_eigenvectors_init_conf(&local_conf);
    else
        local_conf = *_conf;
    conf = &local_conf;

    if (conf->tile_size == STARNEIG_EIGENVECTORS_DEFAULT_TILE_SIZE) {
        conf->tile_size = MAX(64, divceil(0.016*n, 8)*8);
        starneig_message("Setting tile size to %d.", conf->tile_size);
    }
    else if (conf->tile_size < 8) {
        starneig_error("Invalid tile size.");
        ret = STARNEIG_INVALID_CONFIGURATION;
        goto cleanup;
    }

    int selected_count = 0;
    for (int i = 0; i < n; i++)
        if (selected[i]) selected_count++;

    {
        size_t ld;
        _X = starneig_alloc_matrix(n, selected_count, sizeof(double), &ld);
        ld_X = ld;
    }

    //
    // solve
    //

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_SM);
    starneig_node_resume_starpu();

    starneig_eigvec_gen_initialize_omega(100);
    int _ret = starneig_eigvec_gen_sinew(n, S, ldS, T, ldT, selected, _X, ld_X,
        conf->tile_size, conf->tile_size);

    starpu_task_wait_for_all();
    starneig_node_pause_starpu();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    if (_ret != 0) {
        ret = STARNEIG_GENERIC_ERROR;
        goto cleanup;
    }

    //
    // back transformation
    //

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_PARALLEL);
    starneig_node_pause_awake_starpu();

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        n, selected_count, n, 1.0, Z, ldZ, _X, ld_X, 0.0, X, ldX);

    starneig_node_resume_awake_starpu();
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

cleanup:

    starneig_free_matrix(_X);

    return ret;
}

__attribute__ ((visibility ("default")))
starneig_error_t starneig_GEP_SM_Eigenvectors(
    int n,
    int selected[],
    double S[], int ldS,
    double T[], int ldT,
    double Z[], int ldZ,
    double X[], int ldX)
{
    if (n < 1)              return -1;
    if (selected == NULL)   return -2;
    if (S == NULL)          return -3;
    if (ldS < n)            return -4;
    if (T == NULL)          return -5;
    if (ldT < n)            return -6;
    if (Z == NULL)          return -7;
    if (ldZ < n)            return -8;
    if (X == NULL)          return -9;
    if (ldX < n)            return -10;

    if (!starneig_node_initialized())
        return STARNEIG_NOT_INITIALIZED;

    return starneig_GEP_SM_Eigenvectors_expert(
        NULL, n, selected, S, ldS, T, ldT, Z, ldZ, X, ldX);
}
