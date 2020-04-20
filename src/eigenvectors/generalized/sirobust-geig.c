///
/// @file
///
/// @brief Generalised eigenvectors from real Schur forms: StarPU, robust,
/// power of 2 scaling factors
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
#include "sirobust-geig.h"
#include "common.h"
#include "utils.h"
#include "blocking.h"
#include "tiling.h"
#include "geig.h"
#include "robust.h"
#include "robust-geig.h"
#include "irobust.h"
#include "irobust-geig.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <starpu.h>

// This macro ensures that addresses are computed as size_t
#define _a(i,j) a[(size_t)(j)*lda+(i)]

// --------------------------------------------------------------------------
//  Kernel implementations
// --------------------------------------------------------------------------

///
/// @brief StarPU auxiliary routine; Fills a tile with zeros
///
static void sZeros(void **buffers , void *args)
{
    struct starpu_matrix_interface_t *a_i=
        (struct starpu_matrix_interface_t *)buffers[0];

    int m=STARPU_MATRIX_GET_NX(a_i);
    int n=STARPU_MATRIX_GET_NY(a_i);
    double *a=(double *)STARPU_MATRIX_GET_PTR(a_i);
    size_t lda=STARPU_MATRIX_GET_LD(a_i);

    // Fill the matrix with the given value
    starneig_eigvec_gen_zeros(m, n, a, lda);
}

static struct starpu_codelet sZeros_cl = {
    .name = "sZeros",
    .cpu_funcs = { sZeros },
    .nbuffers = 1,
    .modes = {STARPU_W}
};

///
/// @brief StarPU kernel for computing selected eigenvalues from a tile
///
static void ComputeEigenvalues(void **buffers, void *args)
{
    // Connect buffers to interface variables
    struct starpu_matrix_interface *s_i=
        (struct starpu_matrix_interface *)buffers[0];
    struct starpu_matrix_interface *t_i=
        (struct starpu_matrix_interface *)buffers[1];
    struct starpu_vector_interface *select_i=
        (struct starpu_vector_interface *)buffers[2];
    struct starpu_vector_interface *alphar_i=
        (struct starpu_vector_interface *)buffers[3];
    struct starpu_vector_interface *alphai_i=
        (struct starpu_vector_interface *)buffers[4];
    struct starpu_vector_interface *beta_i=
        (struct starpu_vector_interface *)buffers[5];

    // Extract information through the interface
    int m=STARPU_MATRIX_GET_NX(s_i);

    double *s=(double *)STARPU_MATRIX_GET_PTR(s_i);
    size_t lds=STARPU_MATRIX_GET_LD(s_i);
    double *t=(double *)STARPU_MATRIX_GET_PTR(t_i);
    size_t ldt=STARPU_MATRIX_GET_LD(t_i);

    int *select=(int *)STARPU_VECTOR_GET_PTR(select_i);
    double *alphar=(double *)STARPU_VECTOR_GET_PTR(alphar_i);
    double *alphai=(double *)STARPU_VECTOR_GET_PTR(alphai_i);
    double *beta=(double *)STARPU_VECTOR_GET_PTR(beta_i);

    // Compute the selected eigenvalues
    starneig_eigvec_gen_generalised_eigenvalues(
        m, s, lds, t, ldt, select, alphar, alphai, beta);
}

static struct starpu_codelet ComputeEigenvalues_cl = {
    .name = "ComputeEigenvalues",
    .cpu_funcs = { ComputeEigenvalues },
    .nbuffers = 6,
    .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W}
};


static void sIntConsistentScaling(void **buffers, void *args)
{
    // Interface
    struct starpu_matrix_interface *a_i=
        (struct starpu_matrix_interface *)buffers[0];
    struct starpu_matrix_interface *scal_i=
        (struct starpu_matrix_interface *)buffers[1];

    // Dimensions
    int m=STARPU_MATRIX_GET_NX(a_i);
    int n=STARPU_MATRIX_GET_NY(a_i);
    int k=STARPU_MATRIX_GET_NY(scal_i);

    // Matrix of
    double *a=(double *)STARPU_MATRIX_GET_PTR(a_i);
    size_t lda=STARPU_MATRIX_GET_LD(a_i);
    int *scal=(int *)STARPU_MATRIX_GET_PTR(scal_i);
    size_t lds=STARPU_MATRIX_GET_LD(scal_i);

    // Unpack the arguments
    int idx; starpu_codelet_unpack_args(args, &idx);

    // Enforce consistent scaling on the tile
    starneig_eigvec_gen_int_consistent_scaling(m, n, k, a, lda, scal, lds, idx);

}

static struct starpu_codelet sIntConsistentScaling_cl = {
    .name = "sIntConsistentScaling",
    .cpu_funcs = { sIntConsistentScaling },
    .nbuffers = 2,
    .modes = {STARPU_RW, STARPU_R}
};



///
/// @brief StarPU kernel for scaling a single tile
///
static void scale(void *buffers[], void *args)
{
    struct starpu_matrix_interface *a_i=
        (struct starpu_matrix_interface *)buffers[0];

    // Extract information
    int m=STARPU_MATRIX_GET_NX(a_i);
    int n=STARPU_MATRIX_GET_NY(a_i);
    size_t lda=STARPU_MATRIX_GET_LD(a_i);
    double *a=(double *)STARPU_MATRIX_GET_PTR(a_i);

    // Extract the argument(s)
    double alpha;
    starpu_codelet_unpack_args(args, &alpha);

    // Scale the columns of A by alpha
    for (int j=0; j<n; j++)
        dscal_(&m, &alpha, &_a(0,j), &int_one);
}

// Scaling of matrix
static struct starpu_codelet scale_cl = {
    .name = "scale",
    .cpu_funcs = { scale },
    .nbuffers = 1,
    .modes = {STARPU_RW}
};

///
/// @brief StarPU codelet for infinity norm computation
///
static void infnorm(void *buffers[], void *args)
{
    /*
        variable    buffer id      buffer type
        a           0              matrix
        anorm       1              variable
        work        2              scratch
    */
    struct starpu_matrix_interface *a_i=
        (struct starpu_matrix_interface *)buffers[0];
    struct starpu_variable_interface *anorm_i=
        (struct starpu_variable_interface *)buffers[1];
    struct starpu_vector_interface *work_i=
        (struct starpu_vector_interface *)buffers[2];

    // Extract information about the matrix
    int m=STARPU_MATRIX_GET_NX(a_i);
    int n=STARPU_MATRIX_GET_NY(a_i);
    size_t lda=STARPU_MATRIX_GET_LD(a_i);
    double *a=(double *)STARPU_MATRIX_GET_PTR(a_i);

    // Get a pointer to the norm
    double *anorm=(double *)STARPU_VARIABLE_GET_PTR(anorm_i);

    // Get a pointer to the work space
    double *work=(double *)STARPU_VECTOR_GET_PTR(work_i);

    // Compute the norm
    *anorm=starneig_eigvec_gen_dlange("I", m, n, a, lda, work);
}


// Norm computation
static struct starpu_codelet infnorm_cl = {
    .name = "InfNorm",
    .cpu_funcs = { infnorm },
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_W, STARPU_SCRATCH}
};


///
/// @brief StarPU codelet for robust multishift solves and power of sf.
///
static void solve(void *buffers[], void *args)
{
    // A wrapper for irobust_solve_task from irobust-geig
    /* Buffer map
          variable       buffer id       buffer type         comment
          s              0                matrix
          cs             1                vector
          t              2                matrix
          ct             3                vector

          blocks         4                vector
          numBlocks                       passed as argument
          alphar         5                vector
          alphai         6                vector
          beta           7                vector
          map            8                vector
          ----------------------------------------------------------
          ap0                             passed as arguments
          ap1
          bp0
          bp1
          cp0
          cp1
          ----------------------------------------------------------
          y              9               matrix             get m, n from this
          yscal         10               vector
          ynorm         11               vector
          work          12               vector

    */

    // Interface with buffers
    struct starpu_matrix_interface *s_i=
        (struct starpu_matrix_interface *)buffers[0];
    struct starpu_vector_interface *cs_i=
        (struct starpu_vector_interface *)buffers[1];
    struct starpu_matrix_interface *t_i=
        (struct starpu_matrix_interface *)buffers[2];
    struct starpu_vector_interface *ct_i=
        (struct starpu_vector_interface *)buffers[3];
    struct starpu_vector_interface *blocks_i=
        (struct starpu_vector_interface *)buffers[4];

    struct starpu_vector_interface *alphar_i=
        (struct starpu_vector_interface *)buffers[5];
    struct starpu_vector_interface *alphai_i=
        (struct starpu_vector_interface *)buffers[6];
    struct starpu_vector_interface *beta_i=
        (struct starpu_vector_interface *)buffers[7];
    struct starpu_vector_interface *map_i=
        (struct starpu_vector_interface *)buffers[8];

    struct starpu_matrix_interface *y_i=
        (struct starpu_matrix_interface *)buffers[9];
    struct starpu_vector_interface *yscal_i=
        (struct starpu_vector_interface *)buffers[10];
    struct starpu_vector_interface *ynorm_i=
        (struct starpu_vector_interface *)buffers[11];

    struct starpu_vector_interface *work_i=
        (struct starpu_vector_interface *)buffers[12];

    // Extract information through the interfaces
    int m=STARPU_MATRIX_GET_NX(y_i);
    int n=STARPU_MATRIX_GET_NY(y_i);

    double *s=(double *)STARPU_MATRIX_GET_PTR(s_i);
    size_t lds=STARPU_MATRIX_GET_LD(s_i);
    double *cs=(double *)STARPU_VECTOR_GET_PTR(cs_i);
    double *t=(double *)STARPU_MATRIX_GET_PTR(t_i);
    size_t ldt=STARPU_MATRIX_GET_LD(t_i);
    double *ct=(double *)STARPU_VECTOR_GET_PTR(ct_i);
    int *blocks=(int *)STARPU_VECTOR_GET_PTR(blocks_i);

    double *alphar=(double *)STARPU_VECTOR_GET_PTR(alphar_i);
    double *alphai=(double *)STARPU_VECTOR_GET_PTR(alphai_i);
    double *beta=(double *)STARPU_VECTOR_GET_PTR(beta_i);
    int *map=(int *)STARPU_VECTOR_GET_PTR(map_i);

    double *y=(double *)STARPU_MATRIX_GET_PTR(y_i);
    size_t ldy=STARPU_MATRIX_GET_LD(y_i);
    int *yscal=(int *)STARPU_VECTOR_GET_PTR(yscal_i);
    double *ynorm=(double *)STARPU_VECTOR_GET_PTR(ynorm_i);

    double *work=(double *)STARPU_VECTOR_GET_PTR(work_i);

    // Extract the arguments
    int numBlocks; int ap0; int ap1; int bp0; int bp1; int cp0; int cp1;
    starpu_codelet_unpack_args(args, &numBlocks,
			     			     &ap0, &ap1, &bp0, &bp1, &cp0, &cp1);

    // Do the actual solve
    starneig_eigvec_gen_irobust_solve_task(m, n,
		     		     s, lds, cs,
		     		     t, ldt, ct,
		     		     blocks, numBlocks,
		     		     alphar, alphai, beta, map,
		     		     ap0, ap1, bp0, bp1, cp0, cp1,
		     		     y, ldy, yscal, ynorm,
		     		     work);
}

// Solve task
static struct starpu_codelet solve_cl = {
    .name = " solve ",
    .cpu_funcs = { solve },
    .nbuffers = 13,
    .dyn_modes = (enum starpu_data_access_mode[])
    { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R,
        STARPU_R, STARPU_R, STARPU_R, STARPU_R,
        STARPU_RW, STARPU_RW, STARPU_RW, STARPU_SCRATCH}
};


///
/// @brief StarPU codelet for multi-shift updates with power of 2 sf.
///
static void update2(void *buffers[], void *args)
{
    // A wrapper for irobust_update_task from irobust-geig

    // Interface with the buffers
    struct starpu_matrix_interface *s_i =
        (struct starpu_matrix_interface *)buffers[0];
    struct starpu_matrix_interface *t_i =
        (struct starpu_matrix_interface *)buffers[1];

    struct starpu_vector_interface *alphar_i =
        (struct starpu_vector_interface *)buffers[2];
    struct starpu_vector_interface *alphai_i =
        (struct starpu_vector_interface *)buffers[3];
    struct starpu_vector_interface *beta_i =
        (struct starpu_vector_interface *)buffers[4];

    struct starpu_matrix_interface *x_i =
        (struct starpu_matrix_interface *)buffers[5];
    struct starpu_vector_interface *xscal_i =
        (struct starpu_vector_interface *)buffers[6];
    struct starpu_vector_interface *xnorm_i =
        (struct starpu_vector_interface *)buffers[7];

    struct starpu_matrix_interface *y_i =
        (struct starpu_matrix_interface *)buffers[8];
    struct starpu_vector_interface *yscal_i =
        (struct starpu_vector_interface *)buffers[9];
    struct starpu_vector_interface *ynorm_i =
        (struct starpu_vector_interface *)buffers[10];

    // Extract information through the interface
    // Dimensions
    int m=STARPU_MATRIX_GET_NX(y_i);
    int n=STARPU_MATRIX_GET_NY(y_i);
    int k=STARPU_MATRIX_GET_NX(x_i);

    // Matrices
    double *s=(double *)STARPU_MATRIX_GET_PTR(s_i);
    size_t lds=STARPU_MATRIX_GET_LD(s_i);
    double *t=(double *)STARPU_MATRIX_GET_PTR(t_i);
    size_t ldt=STARPU_MATRIX_GET_LD(t_i);

    // Shifts
    double *alphar=(double *)STARPU_VECTOR_GET_PTR(alphar_i);
    double *alphai=(double *)STARPU_VECTOR_GET_PTR(alphai_i);
    double *beta=(double *)STARPU_VECTOR_GET_PTR(beta_i);

    // Unpack arguments: snorm, tnorm, as well as
    //    first:last+1 for induced and practical partitioning
    double snorm; double tnorm; int bp0; int bp1; int cp0; int cp1;
    starpu_codelet_unpack_args(args, &snorm, &tnorm, &bp0, &bp1, &cp0, &cp1);

    // Matrix X with its scaling factors and norms
    double *x=(double *)STARPU_MATRIX_GET_PTR(x_i);
    size_t ldx=STARPU_MATRIX_GET_LD(x_i);
    int *xscal=(int *)STARPU_VECTOR_GET_PTR(xscal_i);
    double *xnorm=(double *)STARPU_VECTOR_GET_PTR(xnorm_i);

    // Matrix Y with its scaling factors and norms
    double *y=(double *)STARPU_MATRIX_GET_PTR(y_i);
    size_t ldy=STARPU_MATRIX_GET_LD(y_i);
    int *yscal=(int *)STARPU_VECTOR_GET_PTR(yscal_i);
    double *ynorm=(double *)STARPU_VECTOR_GET_PTR(ynorm_i);

    // Do the actual update
    starneig_eigvec_gen_irobust_update_task(m, n, k,
		      		      s, lds, snorm,
		      		      t, ldt, tnorm,
		      		      alphar, alphai, beta,
		      		      bp0, bp1, cp0, cp1,
		      		      x, ldx, xscal, xnorm,
		      		      y, ldy, yscal, ynorm);
}

// Codelet for update tasks
static struct starpu_codelet update2_cl = {
    .name = "update2",
    .cpu_funcs = { update2 },
    .nbuffers = 11,
    .dyn_modes = (enum starpu_data_access_mode[])
    { STARPU_R, STARPU_R, STARPU_R, STARPU_R,
        STARPU_R, STARPU_R, STARPU_R, STARPU_R,
        STARPU_RW, STARPU_RW, STARPU_RW}
};


///
///
/// @brief StarPU codelet for preprocessing diagonal tiles
///
static void ProcessDiagonalTile(void *buffers[], void *args)

{
    // Establish interface
    struct starpu_matrix_data_interface *s_i=
        (struct starpu_matrix_data_interface *)buffers[0];
    struct starpu_matrix_data_interface *t_i=
        (struct starpu_matrix_data_interface *)buffers[1];

    struct starpu_vector_data_interface *blocks_i=
        (struct starpu_vector_data_interface *)buffers[2];
    struct starpu_variable_data_interface *numBlocks_i=
        (struct starpu_variable_data_interface *)buffers[3];
    struct starpu_vector_data_interface *cs_i=
        (struct starpu_vector_data_interface *)buffers[4];
    struct starpu_vector_data_interface *ct_i=
        (struct starpu_vector_data_interface *)buffers[5];

    // Variables
    int m=STARPU_MATRIX_GET_NX(s_i);
    double *s=(double *)STARPU_MATRIX_GET_PTR(s_i);
    size_t lds=STARPU_MATRIX_GET_LD(s_i);
    double *t=(double *)STARPU_MATRIX_GET_PTR(t_i);
    size_t ldt=STARPU_MATRIX_GET_LD(t_i);
    int *blocks=(int *)STARPU_VECTOR_GET_PTR(blocks_i);
    int *numBlocks=(int *)STARPU_VARIABLE_GET_PTR(numBlocks_i);
    double *cs=(double *)STARPU_VECTOR_GET_PTR(cs_i);
    double *ct=(double *)STARPU_VECTOR_GET_PTR(ct_i);

    // Count actual number of mini-blocks
    *numBlocks=starneig_eigvec_gen_count_blocks(m, s, lds);

    // Find the location of the mini-blocks
    *numBlocks=starneig_eigvec_gen_find_blocks(m, s, lds, blocks, *numBlocks);

    // Compute generalised column majorants based on the mini-block structure
    starneig_eigvec_gen_generalised_column_majorants(
        m, s, lds, blocks, *numBlocks, cs);
    starneig_eigvec_gen_generalised_column_majorants(
        m, t, ldt, blocks, *numBlocks, ct);

}

// Mini-block map and generalised column majorants
static struct starpu_codelet ProcessDiagonalTile_cl = {
    .name = "ProcessDiagonalTile",
    .cpu_funcs = { ProcessDiagonalTile },
    .nbuffers = 6,
    .modes = {STARPU_R, STARPU_R,
	    	    STARPU_W, STARPU_W,
	    	    STARPU_W, STARPU_W}
};



// ************************************************************************
//   Auxililiary routines
// ************************************************************************

// Allocate and register handles into a submatrix
void starneig_AR_MatrixHandles(void *a, size_t lda, int size,
		      		      int *p, int m, int *q, int n,
		      		      starpu_data_handle_t ***ptr)
{
    // Allocate handles
    starpu_data_handle_t **a_h=
        (starpu_data_handle_t **)malloc(m*sizeof(starpu_data_handle_t));
    for (int i=0; i<m; i++) {
        a_h[i]=
            (starpu_data_handle_t *)malloc(n*sizeof(starpu_data_handle_t));
        int lm=p[i+1]-p[i];
        for (int j=0; j<n; j++) {
            int ln=q[j+1]-q[j];
            starpu_matrix_data_register(&a_h[i][j],
				  				  STARPU_MAIN_RAM,
				  				  (uintptr_t)(a+(lda*q[j]+p[i])*size),
				  				  lda, lm, ln,
				  				  size);
        }
    }
    // Set return variables
    *ptr=a_h;
}

// Unregister and free 2D array of handles
void starneig_UF_MatrixHandles(starpu_data_handle_t **a_h, int m, int n) {

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++)
            starpu_data_unregister(a_h[i][j]);
        free(a_h[i]);
    }
    free(a_h);
}


// Allocate and register handles into an array
void starneig_AR_ArrayHandles(void *array, int size, int *p, int n,
		     		     starpu_data_handle_t **ptr)
{
    // Allocate handles
    starpu_data_handle_t *array_h=
        (starpu_data_handle_t *)malloc(n*sizeof(starpu_data_handle_t));

    // Register data handles
    for (int j=0; j<n; j++) {
        // Size of jth segment
        int ln=p[j+1]-p[j];
        // Register handle to current segment
        starpu_vector_data_register(&array_h[j],
								STARPU_MAIN_RAM,
								(uintptr_t)(array+p[j]*size),
								ln,
								size);
    }
    // Set return values
    *ptr=array_h;
}

// Unregister and free an array of handles
void starneig_UF_ArrayHandles(starpu_data_handle_t *array_h, int n)
{
    // Unregister all handles
    for (int j=0; j<n; j++) {
        starpu_data_unregister(array_h[j]);
    }
    // Free array handles
    free(array_h);
}



// Allocate and Register tile handes (i<=j)
void starneig_AR_TileHandles(double *a, size_t lda, int *ap, int m,
		    		    starpu_data_handle_t ***ptr)
{

    starpu_data_handle_t **a_h=
        (starpu_data_handle_t **)malloc(m*m*sizeof(starpu_data_handle_t *));

    // Register data handles to all tiles A(i,j), i<=j, partitioning ap, ap
    for (int i=0; i<m; i++) {
        int lm=ap[i+1]-ap[i];
        a_h[i]=(starpu_data_handle_t *)malloc(m*sizeof(starpu_data_handle_t));
        for (int j=i; j<m; j++) {
            int ln=ap[j+1]-ap[j];
            starpu_matrix_data_register(&a_h[i][j],
  				    				  STARPU_MAIN_RAM,
  				    				  (uintptr_t)(a+lda*ap[j]+ap[i]),
  				    				  lda, lm, ln, sizeof(double));
        }
    }
    // Set return values
    *ptr=a_h;
}

// Unregister and Free tile handles (i<=j)
void starneig_UF_TileHandles(starpu_data_handle_t **a_h, int M) {

    // Unregister all tiles A(i,j) i<=j
    for (int i=0; i<M; i++) {
        for (int j=i; j<M; j++)
            starpu_data_unregister(a_h[i][j]);
        free(a_h[i]);
    }
    free(a_h);
}

// Specialized routine for setting up handles into the matrix Y, yscal, ynorm
void starneig_AR_TilesY(int *ap, int *cp, int numRows, int numCols, int n,
	       	       double *y, size_t ldy, int *yscal, double *ynorm,
	       	       starpu_data_handle_t ***ptr1,
	       	       starpu_data_handle_t ***ptr2,
	       	       starpu_data_handle_t ***ptr3)
{


    // Register data handles to all tiles Y(i,j), partitioning ap, cp
    starpu_data_handle_t **y_h =
        (starpu_data_handle_t **)malloc(numRows*sizeof(starpu_data_handle_t *));
    for (int i=0; i<numRows; i++) {
        y_h[i]=
            (starpu_data_handle_t *)malloc(numCols*sizeof(starpu_data_handle_t));
        int lm=ap[i+1]-ap[i];
        for (int j=0; j<numCols; j++) {
            int ln=cp[j+1]-cp[j];
            starpu_matrix_data_register(&y_h[i][j],
				  				  STARPU_MAIN_RAM,
				  				  (uintptr_t)&y[(size_t)ldy*cp[j]+ap[i]],
				  				  ldy, lm, ln,
				  				  sizeof(double));
        }
    }

    // Register data handles to power of 2 scaling factors for matrix Y
    starpu_data_handle_t **yscal_h=
        (starpu_data_handle_t **)malloc(numRows*sizeof(starpu_data_handle_t *));
    for (int i=0; i<numRows; i++) {
        yscal_h[i]=
            (starpu_data_handle_t *)malloc(numCols*sizeof(starpu_data_handle_t));
        for (int j=0; j<numCols; j++) {
            int ln=cp[j+1]-cp[j];
            starpu_vector_data_register(&yscal_h[i][j],
				  				  STARPU_MAIN_RAM,
				  				  (uintptr_t)&yscal[(size_t)n*i+cp[j]],
				  				  ln,
				  				  sizeof(int));
        }
    }

    // Register data handles to power mini-block column norms for matrix Y
    starpu_data_handle_t **ynorm_h=
        (starpu_data_handle_t **)malloc(numRows*sizeof(starpu_data_handle_t *));
    for (int i=0; i<numRows; i++) {
        ynorm_h[i]=
            (starpu_data_handle_t *)malloc(numCols*sizeof(starpu_data_handle_t));
        for (int j=0; j<numCols; j++) {
            int ln=cp[j+1]-cp[j];
            starpu_vector_data_register(&ynorm_h[i][j],
				  				  STARPU_MAIN_RAM,
				  				  (uintptr_t)&ynorm[(size_t)n*i+cp[j]],
				  				  ln,
				  				  sizeof(double));
        }
    }
    // Set return variables
    *ptr1=y_h; *ptr2=yscal_h; *ptr3=ynorm_h;
}

// Unregister and free tiles of Y
void starneig_UF_TilesY(starpu_data_handle_t **y_h,
	       	       starpu_data_handle_t **yscal_h,
	       	       starpu_data_handle_t **ynorm_h, int numRows, int numCols)
{
    for (int i=0; i<numRows; i++) {
        for (int j=0; j<numCols; j++) {
            starpu_data_unregister(y_h[i][j]);
            starpu_data_unregister(yscal_h[i][j]);
            starpu_data_unregister(ynorm_h[i][j]);
        }
        free(y_h[i]); free(yscal_h[i]); free(ynorm_h[i]);
    }
    free(y_h); free(yscal_h); free(ynorm_h);
}


// Compute infinity norm of all tiles (i<=j)
void starneig_ComputeTileNorms(starpu_data_handle_t **a_h,
		      		      starpu_data_handle_t anorm_h, int M,
		      		      starpu_data_handle_t work)
{

    struct starpu_data_filter cols =
        {
            // Partition along Y dimension
            .filter_func = starpu_matrix_filter_vertical_block,
            .nchildren = M
        };

    struct starpu_data_filter rows =
        {
            // Partition along X
            .filter_func = starpu_matrix_filter_block,
            .nchildren = M
        };

    // Apply filters, hopefully cutting anorm_h into many subhandles
    starpu_data_map_filters(anorm_h, 2, &rows, &cols);

    // Insert tasks computing norms of blocks (i<=j)
    for (int i=0; i<M; i++) {
        for (int j=i; j<M; j++) {
            starpu_data_handle_t aux_h =
  	  	starpu_data_get_sub_data(anorm_h, 2, i, j);
            starpu_task_insert(&infnorm_cl,
  			   			 STARPU_PRIORITY, STARPU_MAX_PRIO,
  			   			 STARPU_R, a_h[i][j],
  			   			 STARPU_W, aux_h,
  			   			 STARPU_SCRATCH, work,
  			   			 0);
        }
    }

    // Reassemble anorm_h from all the subhandles.
    starpu_data_unpartition(anorm_h, STARPU_MAIN_RAM);
}

// Scale all tiles (i<=j) by a constant alpha
void starneig_eigvec_std_scaleTiles(double alpha, starpu_data_handle_t **a_h, int M)
{
    // Insert tasks which scales each tile by alpha
    for (int i=0; i<M; i++)
        for (int j=i; j<M; j++)
            starpu_task_insert(&scale_cl,
			 			 STARPU_PRIORITY, STARPU_MAX_PRIO,
			 			 STARPU_VALUE, &alpha, sizeof(double),
			 			 STARPU_RW, a_h[i][j],
			 			 0);
}



/* // Main routines */

int starneig_eigvec_gen_sinew(int m,
	  	  double *s, size_t lds,
	  	  double *t, size_t ldt,
	  	  int *select, double *y, size_t ldy,
	  	  int mb, int nb)

{

    // *************************************************************************
    // Computing tilings
    // *************************************************************************

    // Left looking array and map of all selected eigenvalues
    int *l; int *map;

    // Practical row, induced column and practical column tiling
    int *ap; int *bp; int *cp;

    // Number of tile rows and tile columns for matrix Y
    int numRows; int numCols;

    // Find all relevant information
    starneig_eigvec_gen_find_tilings(m, mb, nb, s, lds, select,
	      	      &l, &map, &ap, &bp, &cp, &numRows, &numCols);

    // Isolate the number of selected eigenvectors
    int n=cp[numCols];

    // ***********************************************************************
    //   Scratch space for each worker
    // ***********************************************************************

    starpu_data_handle_t work;
    starpu_vector_data_register(&work, -1, (uintptr_t)0,
  			        			      6*(mb+1), sizeof(double));

    // ***********************************************************************
    //   Allocate and register tiles of matrices S, T
    // ***********************************************************************

    // Handles to tiles of matrices S and T
    starpu_data_handle_t **s_h;
    starpu_data_handle_t **t_h;

    // Allocate (malloc) and register StarPU-handles to tiles of S, T
    starneig_AR_TileHandles(s, lds, ap, numRows, &s_h);
    starneig_AR_TileHandles(t, ldt, ap, numRows, &t_h);


    // ***********************************************************************
    //    Compute norms of all tiles (i<=j) and scale if necessary
    // ***********************************************************************

    // We store the tile norms in square matrices of dimension numRows
    double *snorm=(double *)malloc(numRows*numRows*sizeof(double));
    double *tnorm=(double *)malloc(numRows*numRows*sizeof(double));

    // Nullification must be done here. See the calls to dlange below.
    starneig_eigvec_gen_zeros(numRows, numRows, snorm, numRows);
    starneig_eigvec_gen_zeros(numRows, numRows, tnorm, numRows);

    // Data handles to matrices of tile norms
    starpu_data_handle_t snorm_h;
    starpu_data_handle_t tnorm_h;

    // Register handle to matrix of norms of tiles of S
    starpu_matrix_data_register(&snorm_h, STARPU_MAIN_RAM,
  			        			      (uintptr_t)snorm,
  			        			      numRows, numRows, numRows, sizeof(double));

    // Register handle to matrix of norms of tiles of T
    starpu_matrix_data_register(&tnorm_h, STARPU_MAIN_RAM,
  			        			      (uintptr_t)tnorm,
  			        			      numRows, numRows, numRows, sizeof(double));


    // Computer norms of all tiles (i<=j)
    starneig_ComputeTileNorms(s_h, snorm_h, numRows, work);
    starneig_ComputeTileNorms(t_h, tnorm_h, numRows, work);

    // Unregister snorm_h, tnorm_h forcing data to be written to memory
    // All tile norms are passed as parameters when inserting update task
    starpu_data_unregister(snorm_h);
    starpu_data_unregister(tnorm_h);


    // Determine the maximum element of snorm, tnorm
    // The nullification of subdiagonal simplifies this process
    double aux1=dlange_("M", &numRows, &numRows, snorm, &numRows, NULL);
    double aux2=dlange_("M", &numRows, &numRows, tnorm, &numRows, NULL);

    // Determine the largest norm of *any* tile in play (i<=j)
    double aux=MAX(aux1,aux2);

    // Check for overflow
    if (aux>Omega) {
        // Scaling *is* necessary
        aux=Omega/aux;
        // Apply scaling to all tiles of S, T  (i<=j)
        starneig_eigvec_std_scaleTiles(aux, s_h, numRows);
        starneig_eigvec_std_scaleTiles(aux, t_h, numRows);
    }


    // ***********************************************************************
    //  Mini-block structure and generalized column majorants
    // ***********************************************************************

    // Allocate space for mini-block maps and counts
    int **blocks=(int **)malloc(numRows*sizeof(int *));

    // The number of mini-blocks for the ith tile is numBlocks[i]
    int *numBlocks=(int *)malloc(numRows*sizeof(int));

    // Allocate space for generalized column majorants
    double **cs=(double **)malloc(numRows*sizeof(double *));
    double **ct=(double **)malloc(numRows*sizeof(double *));

    /* Below we allocate enough memory to handle the worst case.
          It is possible to determine exactly how much memory is necessary, but
          this requires communication between the workers who do the computation
          and the master who does the allocation.

          The ith diagonal tile has dimension lm=ap[i+1]-ap[i].
          Such a tile can have at most lm mini-blocks which are all 1-by-1.
          Therefore we need at most lm+1 words of data to store the mini-block
          maps. Moreover, at most lm words are needed for set of column majorants.

    */

    // Loop over the diagonal blocks
    for (int i=0; i<numRows; i++) {
        // Dimension of current block is upper bound for number of mini-blocks
        int lm=ap[i+1]-ap[i];
        // Space for the mini-block map of tile S(i,i)
        blocks[i]=(int *)malloc((lm+1)*sizeof(int));
        // Column majorants for S and T
        cs[i]=(double *)malloc(lm*sizeof(double));
        ct[i]=(double *)malloc(lm*sizeof(double));
    }

    // Handles to mini-block maps and column majorants
    starpu_data_handle_t *blocks_h=
        (starpu_data_handle_t *)malloc(numRows*sizeof(starpu_data_handle_t));
    starpu_data_handle_t *numBlocks_h=
        (starpu_data_handle_t *)malloc(numRows*sizeof(starpu_data_handle_t));

    starpu_data_handle_t *cs_h=
        (starpu_data_handle_t *)malloc(numRows*sizeof(starpu_data_handle_t));
    starpu_data_handle_t *ct_h=
        (starpu_data_handle_t *)malloc(numRows*sizeof(starpu_data_handle_t));

    // Registration of handles
    for (int i=0; i<numRows; i++) {
        // Dimension of current block
        int lm=ap[i+1]-ap[i];
        // Mini-block map for S(i,i)
        starpu_vector_data_register(&blocks_h[i],
  				  				STARPU_MAIN_RAM,
  				  				(uintptr_t)blocks[i], lm+1,
  				  				sizeof(int));
        // Number of mini-blocks for S(i,i)
        starpu_variable_data_register(&numBlocks_h[i],
  				    				  STARPU_MAIN_RAM,
  				    				  (uintptr_t)&numBlocks[i],
  				    				  sizeof(int));
        // Column majorants for S(i,i)
        starpu_vector_data_register(&cs_h[i],
  				  				STARPU_MAIN_RAM,
  				  				(uintptr_t)cs[i], lm,
  				  				sizeof(double));
        // Column majorants for T(i,i)
        starpu_vector_data_register(&ct_h[i],
  				  				STARPU_MAIN_RAM,
  				  				(uintptr_t)ct[i], lm,
  				  				sizeof(double));
    }
    // Insert tasks which process diagonal tiles
    for (int i=0; i<numRows; i++)
        starpu_task_insert(&ProcessDiagonalTile_cl,
  		         		       STARPU_PRIORITY, STARPU_MAX_PRIO,
  		         		       STARPU_R, s_h[i][i],
  		         		       STARPU_R, t_h[i][i],
  		         		       STARPU_W, blocks_h[i],
  		         		       STARPU_W, numBlocks_h[i],
  		         		       STARPU_W, cs_h[i],
  		         		       STARPU_W, ct_h[i],
  		         		       0);

    // Unregister handles forcing write back to main memory.
    // numBlocks will be passed to solve tasks as a parameter
    starneig_UF_ArrayHandles(numBlocks_h, numRows);

    // ***********************************************************************
    //    Eigenvalues
    // ***********************************************************************

    // Allocate space for eigenvalues
    double *alphar=(double *)malloc(n*sizeof(double));
    double *alphai=(double *)malloc(n*sizeof(double));
    for (int i=0; i<n; i++) alphai[i]=-7;
    double *beta=(double *)malloc(n*sizeof(double));

    // Data handles for eigenvalues
    starpu_data_handle_t *alphar_h;
    starpu_data_handle_t *alphai_h;
    starpu_data_handle_t *beta_h;

    // Data handles into selection array
    starpu_data_handle_t *select_h;

    // Allocate and register handles into select using ap
    starneig_AR_ArrayHandles((void *)select, sizeof(int), ap, numRows, &select_h);

    // Allocate and register handles to eigenv using bp
    starneig_AR_ArrayHandles((void *)alphar, sizeof(double), bp, numRows, &alphar_h);
    starneig_AR_ArrayHandles((void *)alphai, sizeof(double), bp, numRows, &alphai_h);
    starneig_AR_ArrayHandles((void *)beta, sizeof(double), bp, numRows, &beta_h);

    // Insert tasks to compute select eigenvalue of diagonal tasks
    for (int i=0; i<numRows; i++) {
        // Compute number of eigenvalues
        int ln=bp[i+1]-bp[i];
        // Only insert non-trivial tasks
        if (ln>0)
            starpu_task_insert(&ComputeEigenvalues_cl,
			 			 STARPU_PRIORITY, STARPU_MAX_PRIO,
			 			 STARPU_R, s_h[i][i],
			 			 STARPU_R, t_h[i][i],
			 			 STARPU_R, select_h[i],
			 			 STARPU_W, alphar_h[i],
			 			 STARPU_W, alphai_h[i],
			 			 STARPU_W, beta_h[i],
			 			 0);
    }

    // Unregister and free array handles for eigenvalues forcing write back
    starneig_UF_ArrayHandles(alphar_h, numRows);
    starneig_UF_ArrayHandles(alphai_h, numRows);
    starneig_UF_ArrayHandles(beta_h, numRows);

    // Allocate and register *new* handles for e.v. using cp (practical part.)
    starneig_AR_ArrayHandles((void *)alphar, sizeof(double), cp, numCols, &alphar_h);
    starneig_AR_ArrayHandles((void *)alphai, sizeof(double), cp, numCols, &alphai_h);
    starneig_AR_ArrayHandles((void *)beta, sizeof(double), cp, numCols, &beta_h);

    // Allocate and register handles into the map of e.v. using cp
    starpu_data_handle_t *map_h;
    starneig_AR_ArrayHandles((void *)map, sizeof(int), cp, numCols, &map_h);

    // ************************************************************************
    //    Initialize eigenvectors
    // ************************************************************************

    //  Allocate space for scaling factors
    int *yscal=(int *)malloc(numRows*n*sizeof(int));

    // Initialize the power of 2 scaling factors of the columns of Y
    for (int k=0; k<numRows*n; k++) yscal[k]=0;

    // Allocat space for norms
    double *ynorm=(double *)malloc(numRows*n*sizeof(double));

    // Allocate and register all handles associate with matrix Y
    starpu_data_handle_t **y_h;
    starpu_data_handle_t **yscal_h;
    starpu_data_handle_t **ynorm_h;
    starneig_AR_TilesY(ap, cp, numRows, numCols, n, y, ldy, yscal, ynorm,
	    	    &y_h, &yscal_h, &ynorm_h);

    // Nullify Y in using task
    for (int i=0; i<numRows; i++)
        for (int j=0; j<numCols; j++)
            starpu_task_insert(&sZeros_cl,
			 			 STARPU_PRIORITY, STARPU_MAX_PRIO,
			 			 STARPU_W, y_h[i][j],
			 			 0);


    // TODO: This as tasks
    // Initialize the norms of Y
    starneig_eigvec_gen_zeros(1,numRows*n,ynorm,1);


    // ************************************************************************
    //   Main loop follows below
    // ************************************************************************


    // Loop over the *practical* tiling of Y
    for (int j=0; j<numCols; j++) {

        for (int i=numRows-1; i>=0; i--) {

            // Does the work region begin before the current column ends?
            if (bp[i]<cp[j+1]) {

		// Insert solve task
		starpu_task_insert(&solve_cl,
			   			   STARPU_PRIORITY, STARPU_MAX_PRIO,
			   			   STARPU_R, s_h[i][i], STARPU_R, cs_h[i],
			   			   STARPU_R, t_h[i][i], STARPU_R, ct_h[i],
			   			   STARPU_R, blocks_h[i],
			   			   STARPU_VALUE, &numBlocks[i], sizeof(int),
			   			   STARPU_R, alphar_h[j],
			   			   STARPU_R, alphai_h[j],
			   			   STARPU_R, beta_h[j],
			   			   STARPU_R, map_h[j],
			   			   STARPU_VALUE, &ap[i+0], sizeof(int),
			   			   STARPU_VALUE, &ap[i+1], sizeof(int),
			   			   STARPU_VALUE, &bp[i+0], sizeof(int),
			   			   STARPU_VALUE, &bp[i+1], sizeof(int),
			   			   STARPU_VALUE, &cp[j+0], sizeof(int),
			   			   STARPU_VALUE, &cp[j+1], sizeof(int),
			   			   STARPU_RW, y_h[i][j],
			   			   STARPU_RW, yscal_h[i][j],
			   			   STARPU_RW, ynorm_h[i][j],
			   			   STARPU_SCRATCH, work,
			   			   0);



     	     	// Update all data above the *active* region of Y(i,j)
		for (int k=0; k<i; k++) {
	  	  // ****************************************************************
	  	  //  Y(k,j) = Y(k,j) - (S(k,i)*Y(i,j)*D(j,j) - T(k,i)*Y(i,j)*B(j,j)
	  	  // ****************************************************************

	  	  // Insert update task
	  	  starpu_task_insert(&update2_cl,
			     			     STARPU_PRIORITY,
			     			     MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO+k-i),
			     			     STARPU_R, s_h[k][i],
			     			     STARPU_VALUE, &snorm[numRows*i+k], sizeof(double),
			     			     STARPU_R, t_h[k][i],
			     			     STARPU_VALUE, &tnorm[numRows*i+k], sizeof(double),
			     			     STARPU_R, alphar_h[j],
			     			     STARPU_R, alphai_h[j],
			     			     STARPU_R, beta_h[j],
			     			     STARPU_VALUE, &bp[i+0], sizeof(int),
			     			     STARPU_VALUE, &bp[i+1], sizeof(int),
			     			     STARPU_VALUE, &cp[j+0], sizeof(int),
			     			     STARPU_VALUE, &cp[j+1], sizeof(int),
			     			     STARPU_R, y_h[i][j],
			     			     STARPU_R, yscal_h[i][j],
			     			     STARPU_R, ynorm_h[i][j],
			     			     STARPU_RW, y_h[k][j],
			     			     STARPU_RW, yscal_h[k][j],
			     			     STARPU_RW, ynorm_h[k][j],
			     			     0);
		}
            }
        }
    }



    // **********************************************************************
    //   Unregistration follows below.
    // **********************************************************************

    // Handles into select
    starneig_UF_ArrayHandles(select_h, numRows);

    // Handles into eigenvalues
    starneig_UF_ArrayHandles(alphar_h, numCols);
    starneig_UF_ArrayHandles(alphai_h, numCols);
    starneig_UF_ArrayHandles(beta_h, numCols);

    // Handles into maps
    starneig_UF_ArrayHandles(map_h, numCols);

    // Unregister and free tiles handles
    starneig_UF_TileHandles(s_h, numRows);
    starneig_UF_TileHandles(t_h, numRows);

    // Unregister datahandles to scratch space
    starpu_data_unregister(work);

    // Handles for mini-block structure
    starneig_UF_ArrayHandles(blocks_h, numRows);

    // Handles for column majorants
    starneig_UF_ArrayHandles(cs_h, numRows);
    starneig_UF_ArrayHandles(ct_h, numRows);

    // **********************************************************************
    //   Final scaling
    // **********************************************************************

    /*   Map of scaling factors

              <---------------------- n words of memory -------------------->
              |<- scal. y11 ->|<- scal. y12 ->|<- scal. y13 ->|<- scal. y14 ->|
              |<- scal. y21 ->|<- scal. y22 ->|<- scal. y23 ->|<- scal. y24 ->|
              |<- scal. y31 ->|<- scal. y32 ->|<- scal. y33 ->|<- scal. y34 ->|

              We want to isolate all scaling factors associated with the jth tile column
              of Y, i.e. Y(:,j). It is critical to recognize that these scaling factors
              are not continuous in memory, but are spread in strips of length

              ln = cp[j+1]-cp[j]

              These strips are n words apart in memory.
    */

    // Unregister yscal_h forcing write back of data
    for (int i=0; i<numRows; i++) {
        for (int j=0; j<numCols; j++)
            starpu_data_unregister(yscal_h[i][j]);
        free(yscal_h[i]);
    }
    free(yscal_h);

    // Create handles to the scaling factors related to tile column Y(:,j)
    // It is vital to recall the image above ...

    starpu_data_handle_t *zscal_h=
        (starpu_data_handle_t *)malloc(numCols*sizeof(starpu_data_handle_t));
    for (int j=0; j<numCols; j++) {
        int ln=cp[j+1]-cp[j];
        starpu_matrix_data_register(&zscal_h[j],
								STARPU_MAIN_RAM,
								(uintptr_t)(yscal+cp[j]),
								n, ln, numRows, sizeof(double));
    }

    // Insert tasks which enforce consistent scaling upon Y
    for (int i=0; i<numRows; i++) {
        for (int j=0; j<numCols; j++) {
            starpu_task_insert(&sIntConsistentScaling_cl,
			 			 STARPU_PRIORITY, STARPU_MAX_PRIO,
			 			 STARPU_RW, y_h[i][j],
			 			 STARPU_R, zscal_h[j],
			 			 STARPU_VALUE, &i, sizeof(int),
			 			 0);
        }
    }

    // Unregister and free handles zscal_h
    starneig_UF_ArrayHandles(zscal_h, numCols);

    // Unregister handles into ynorm
    starneig_UF_MatrixHandles(ynorm_h, numRows, numCols);

    // Unregister handles into Y
    starneig_UF_MatrixHandles(y_h, numRows, numCols);

    // *************************************************************************
    // Deallocation of memory follows here
    // *************************************************************************

    // From the construction of tilings
    free(l); free(map); free(ap); free(bp); free(cp);

    // From computation of eigenvalues
    free(alphar); free(alphai); free(beta);

    // From mini-block-structure
    for (int i=0; i<numRows; i++)
        free(blocks[i]);
    free(blocks); free(numBlocks);

    // Needed for robusts
    free(snorm); free(tnorm);
    for (int i=0; i<numRows; i++) {
        free(cs[i]);
        free(ct[i]);
    }
    free(cs); free(ct);

    // Also needed for robustness
    free(ynorm); free(yscal);

    // Dummy return code
    return 0;
}

#undef _a
