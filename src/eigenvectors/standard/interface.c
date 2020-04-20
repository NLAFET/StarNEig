///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
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
#include <starneig/error.h>
#include "core.h"
#include "typedefs.h"
#include "robust.h"
#include "cpu.h"
#include "partition.h"
#include "../../common/common.h"
#include "../../common/node_internal.h"
#include "../../common/matrix.h"
#include <starneig/sep_sm.h>
#include <cblas.h>
#include <stdlib.h>
#include <starpu.h>
#include <math.h>
#include <float.h>


static int search_multiplicities(int n, double *lambda, int *lambda_type)
{
    int multiplicity = 0;
    for (int i = 0; i < n; i++) {
        if (lambda_type[i] == 0) { // REAL
            double real = lambda[i];
            for (int j = i+1; j < n; j++) {
                if (lambda_type[j] == 0 && lambda[j] == real) {
                    multiplicity = 1;
                    break;
                }
            }
        }
        else { // CMPLX
            double real = lambda[i];
            double imag = lambda[i+1];
            for (int j = i+2; j < n; j++) {
                if (lambda_type[j] == 1 && lambda_type[j+1]) {
                    if (lambda[j] == real && lambda[j+1] == imag) {
                        multiplicity = 1;
                        break;
                    }

                    j++;
                }
            }
            i++;
        }
    }

    return multiplicity;
}


static starneig_error_t eigenvectors(
    struct starneig_eigenvectors_conf const *_conf,
    int n, int *selected,
    double *S, int ldS,
    double *Q, int ldQ,
    double *Y, int ldY)
{
#define S(i,j) S[(i) + (j) * (size_t)ldS]
#define Q(i,j) Q[(i) + (j) * (size_t)ldQ]
#define X(i,j) X[(i) + (j) * (size_t)ldX]
#define Y(i,j) Y[(i) + (j) * (size_t)ldY]

    // use default configuration if necessary
    struct starneig_eigenvectors_conf *conf;
    struct starneig_eigenvectors_conf local_conf;
    if (_conf == NULL)
        starneig_eigenvectors_init_conf(&local_conf);
    else
        local_conf = *_conf;
    conf = &local_conf;


    //
    // check mandatory arguments
    //

    if (selected == NULL) {
        starneig_error("Eigenvalue selection bitmap is NULL. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    int num_selected = starneig_eigvec_std_count_selected(n, selected);
    if (num_selected == 0) {
        starneig_error("Eigenvalue selection bitmap does not have any "
                       "selected eigenvalues. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (S == NULL) {
        starneig_error("Matrix S is NULL. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (Q == NULL) {
        starneig_error("Matrix Q is NULL. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }

    if (Y == NULL) {
        starneig_error("Eigenvector matrix is NULL. Exiting...");
        return STARNEIG_INVALID_ARGUMENTS;
    }


    //
    // check configuration
    //

    if (conf->tile_size == STARNEIG_EIGENVECTORS_DEFAULT_TILE_SIZE) {
        double select_ratio = (double) num_selected/n;
        conf->tile_size = MIN(MAX(240, sqrt(n)/sqrt(select_ratio)), 936);
        starneig_message("Setting tile size to %d.", conf->tile_size);
    }

    if (conf->tile_size <= 0) {
        starneig_error("Tile size is %d. Exiting...", conf->tile_size);
        return STARNEIG_INVALID_CONFIGURATION;
    }

    //
    // preprocess
    //

    // extract the eigenvalues and their type (1-by-1 or 2-by-2 block)
    double *lambda = (double *) malloc((size_t)n * sizeof(double));
    int *lambda_type = (int *) malloc((size_t)n * sizeof(int));
    lambda[n-1] = S(n-1,n-1);
    lambda_type[n-1] = 0;
    for (int i = 0; i < n - 1; i++) {
        if (S(i+1,i) != 0.0) {
            // A 2-by-2 block in canonical Schur form has the shape
            // [ S(i,i)      S(i,i+1)   ] = [ a b ]  or [ a -b ]
            // [ S(i+1,i)    S(i+1,i+1) ]   [-c a ]     [ c  a ].
            lambda_type[i] = 1;
            lambda_type[i+1] = 1;
            double real = S(i+1,i+1);
            double imag = sqrt(fabs(S(i+1,i)))*sqrt(fabs(S(i,i+1)));
            lambda[i] = real;
            lambda[i+1] = imag;
            i++;
        }
        else {
            // 1-by-1 block
            lambda_type[i] = 0;
            lambda[i] = S(i,i);
        }
    }


    //
    // sanity check
    //

    starneig_error_t ret = STARNEIG_SUCCESS;
    int illposed = search_multiplicities(n, lambda, lambda_type);
    if (illposed) {
        starneig_warning("Multiple eigenvalues detected.\n");
        ret = STARNEIG_CLOSE_EIGENVALUES;
    }



    //
    // overflow control
    //

    const double eps = DBL_EPSILON/2;
    const double smlnum = MAX(2*DBL_MIN, DBL_MIN*((double)n/eps));


    //
    // partition
    //

    int num_tiles = (n+conf->tile_size-1)/conf->tile_size;
    int *first_row = (int *) malloc((num_tiles+1)*sizeof(int));
    int *first_col = (int *) malloc((num_tiles+1)*sizeof(int));
    starneig_eigvec_std_partition(n, lambda_type, conf->tile_size, first_row);
    starneig_eigvec_std_partition_selected(n, first_row, selected, num_tiles, first_col);


    //
    // workspace
    //

    int ldX = ldY;
    double *X = (double *) malloc((size_t)ldX*num_selected*sizeof(double));

    size_t num_segments = (size_t) num_tiles*num_selected;

    double *Xnorms = (double *) malloc(num_segments*sizeof(double));
#define Xnorms(col, tilerow) Xnorms[(col) + (tilerow) * (size_t)num_selected]

    scaling_t *scales = (scaling_t*) malloc(num_segments*sizeof(scaling_t));
#define scales(col, tilerow) scales[(col) + (tilerow) * (size_t)num_selected]
    starneig_eigvec_std_init_scaling_factor(num_tiles*num_selected, scales);

    double *Snorms =
        (double *) malloc((size_t)num_tiles*num_tiles*sizeof(double));
#define Snorms(i,j) Snorms[(i) + (j) * (size_t)num_tiles]

    int *info = (int *) malloc((size_t)num_selected*sizeof(int));

    // Copy all selected eigenvalue types to a compact memory representation.
    int *selected_lambda_type = (int *) malloc((size_t)num_selected*sizeof(int));
    int idx = 0;
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            selected_lambda_type[idx] = lambda_type[i];
            idx++;
        }
    }


    //
    // register
    //

    starpu_data_handle_t **S_tiles;
    starpu_data_handle_t **Q_tiles;
    starpu_data_handle_t **X_tiles;
    starpu_data_handle_t **Y_tiles;
    starpu_data_handle_t *selected_tiles;
    starpu_data_handle_t *lambda_tiles;
    starpu_data_handle_t *lambda_type_tiles;
    starpu_data_handle_t *selected_lambda_type_tiles;
    starpu_data_handle_t **Xnorms_tiles;
    starpu_data_handle_t **scales_tiles;
    starpu_data_handle_t **S_tiles_norms;
    starpu_data_handle_t *info_tiles;
    S_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    Q_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    X_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    Y_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    selected_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t));
    lambda_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t));
    lambda_type_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t));
    selected_lambda_type_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t));
    Xnorms_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    scales_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    S_tiles_norms = malloc(num_tiles*sizeof(starpu_data_handle_t *));
    info_tiles = malloc(num_tiles*sizeof(starpu_data_handle_t));
    for (int i = 0; i < num_tiles; i++) {
        S_tiles[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));
        Q_tiles[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));
        X_tiles[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));
        Y_tiles[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));
        Xnorms_tiles[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));
        scales_tiles[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));
        S_tiles_norms[i] = malloc(num_tiles*sizeof(starpu_data_handle_t));

        starpu_vector_data_register(
            &selected_tiles[i],
            STARPU_MAIN_RAM,
            (uintptr_t)(&selected[first_row[i]]),
            first_row[i+1]-first_row[i],
            sizeof(int));

        starpu_vector_data_register(
            &lambda_tiles[i],
            STARPU_MAIN_RAM,
            (uintptr_t)(&lambda[first_row[i]]),
            first_row[i+1]-first_row[i],
            sizeof(double));

        starpu_vector_data_register(
            &lambda_type_tiles[i],
            STARPU_MAIN_RAM,
            (uintptr_t)(&lambda_type[first_row[i]]),
            first_row[i+1]-first_row[i],
            sizeof(int));

        starpu_vector_data_register(
            &selected_lambda_type_tiles[i],
            STARPU_MAIN_RAM,
            (uintptr_t)(&selected_lambda_type[first_col[i]]),
            first_col[i+1]-first_col[i],
            sizeof(int));

        starpu_vector_data_register(
            &info_tiles[i],
            STARPU_MAIN_RAM,
            (uintptr_t)(&info[first_col[i]]),
            first_col[i+1]-first_col[i],
            sizeof(int));

        for (int j = 0; j < num_tiles; j++) {
            if (i <= j) {
                starpu_matrix_data_register(
                    &S_tiles[i][j],
                    STARPU_MAIN_RAM,
                    (uintptr_t)(&S(first_row[i], first_row[j])),
                    ldS,
                    first_row[i+1]-first_row[i],
                    first_row[j+1]-first_row[j],
                    sizeof(double));

                starpu_variable_data_register(
                    &S_tiles_norms[i][j],
                    STARPU_MAIN_RAM,
                    (uintptr_t)(&Snorms(i,j)),
                    sizeof(double));

                starpu_matrix_data_register(
                    &X_tiles[i][j],
                    STARPU_MAIN_RAM,
                    (uintptr_t)(&X(first_row[i], first_col[j])),
                    ldX,
                    first_row[i+1]-first_row[i],
                    first_col[j+1]-first_col[j],
                    sizeof(double));

                starpu_vector_data_register(
                    &Xnorms_tiles[i][j],
                    STARPU_MAIN_RAM,
                    (uintptr_t)(&Xnorms(first_col[j],i)),
                    first_col[j+1]-first_col[j],
                    sizeof(double));

                starpu_vector_data_register(
                    &scales_tiles[i][j],
                    STARPU_MAIN_RAM,
                    (uintptr_t)(&scales(first_col[j],i)),
                    first_col[j+1]-first_col[j],
                    sizeof(scaling_t));
            }

            starpu_matrix_data_register(
                &Q_tiles[i][j],
                STARPU_MAIN_RAM,
                (uintptr_t)(&Q(first_row[i], first_row[j])),
                ldQ,
                first_row[i+1]-first_row[i],
                first_row[j+1]-first_row[j],
                sizeof(double));

            starpu_matrix_data_register(
                &Y_tiles[i][j],
                STARPU_MAIN_RAM,
                (uintptr_t)(&Y(first_row[i], first_col[j])),
                ldY,
                first_row[i+1]-first_row[i],
                first_col[j+1]-first_col[j],
                sizeof(double));
        }
    }


    //
    // insert tasks
    //

    starneig_eigvec_std_insert_backsolve_tasks(num_tiles,
        S_tiles, S_tiles_norms, lambda_tiles, lambda_type_tiles,
        X_tiles, scales_tiles, Xnorms_tiles, selected_tiles,
        selected_lambda_type_tiles, info_tiles, smlnum,
        STARPU_MAX_PRIO, STARPU_DEFAULT_PRIO);

    starpu_task_wait_for_all();

    starneig_eigvec_std_unify_scaling(num_tiles, first_row, first_col, scales, X, ldX,
        lambda_type, selected);

    starneig_eigvec_std_insert_backtransform_tasks(first_row, num_tiles,
        Q_tiles, X_tiles, Y_tiles);


    //
    // evaluate reliability
    //

    for (int i = 0; i < num_selected; i++) {
        if (info[i] != STARNEIG_SUCCESS) {
            starneig_warning("Eigenvector column X(:,%d) was perturbed and "
                             "cannot be trusted.", i);
            ret = STARNEIG_CLOSE_EIGENVALUES;
        }
    }

    starpu_task_wait_for_all();


    //
    // clean up
    //

    for (int i = 0; i < num_tiles; i++) {
        starpu_data_unregister(selected_tiles[i]);
        starpu_data_unregister(lambda_tiles[i]);
        starpu_data_unregister(lambda_type_tiles[i]);
        starpu_data_unregister(selected_lambda_type_tiles[i]);
        starpu_data_unregister(info_tiles[i]);
        for (int j = 0; j < num_tiles; j++) {
            if (i <= j) {
                starpu_data_unregister(S_tiles[i][j]);
                starpu_data_unregister(S_tiles_norms[i][j]);
                starpu_data_unregister(X_tiles[i][j]);
                starpu_data_unregister(Xnorms_tiles[i][j]);
                starpu_data_unregister(scales_tiles[i][j]);
            }
            starpu_data_unregister(Q_tiles[i][j]);
            starpu_data_unregister(Y_tiles[i][j]);
        }
        free(S_tiles[i]);
        free(Q_tiles[i]);
        free(X_tiles[i]);
        free(Y_tiles[i]);
        free(Xnorms_tiles[i]);
        free(scales_tiles[i]);
        free(S_tiles_norms[i]);
    }
    free(S_tiles);
    free(Q_tiles);
    free(X_tiles);
    free(Y_tiles);
    free(Xnorms_tiles);
    free(scales_tiles);
    free(selected_tiles);
    free(lambda_tiles);
    free(lambda_type_tiles);
    free(selected_lambda_type_tiles);
    free(info_tiles);
    free(X);
    free(Xnorms);
    free(Snorms);
    free(info);
    free(scales);
    free(lambda_type);
    free(selected_lambda_type);
    free(lambda);
    free(first_row);
    free(first_col);


#undef Xnorms
#undef Snorms
#undef scales
#undef S
#undef Q
#undef X
#undef Y

    return ret;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__attribute__ ((visibility ("default")))
void starneig_eigenvectors_init_conf(struct starneig_eigenvectors_conf *conf) {
    conf->tile_size = STARNEIG_EIGENVECTORS_DEFAULT_TILE_SIZE;
}


__attribute__ ((visibility ("default")))
int starneig_SEP_SM_Eigenvectors_expert(
    struct starneig_eigenvectors_conf *conf,
    int n,
    int selected[],
    double S[], int ldS,
    double Q[], int ldQ,
    double X[], int ldX)
{
    CHECK_INIT();

    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_SEQUENTIAL);
    starneig_node_set_mode(STARNEIG_MODE_SM);
    starneig_node_resume_starpu();

    starneig_error_t ret = eigenvectors(
        conf, n, selected, S, ldS, Q, ldQ, X, ldX);

    starpu_task_wait_for_all();
    starneig_node_pause_starpu();
    starneig_node_set_mode(STARNEIG_MODE_OFF);
    starneig_node_set_blas_mode(STARNEIG_BLAS_MODE_ORIGINAL);

    return ret;
}



__attribute__ ((visibility ("default")))
int starneig_SEP_SM_Eigenvectors(
    int n,
    int selected[],
    double S[], int ldS,
    double Q[], int ldQ,
    double X[], int ldX)
{
    CHECK_INIT();
    return starneig_SEP_SM_Eigenvectors_expert(
        NULL, n, selected, S, ldS, Q, ldQ, X, ldX);
}
