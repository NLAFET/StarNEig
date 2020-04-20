///
/// @file
///
/// @brief Generalised eigenvectors from real Schur forms: tiled
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
#include "geig.h"
#include "common.h"
#include "tiling.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

// This macro ensures that addresses are computed as size_t
#define _s(i,j) s[(size_t)(j)*lds+(i)]
#define _t(i,j) t[(size_t)(j)*ldt+(i)]
#define _x(i,j) x[(size_t)(j)*ldx+(i)]
#define _y(i,j) y[(size_t)(j)*ldy+(i)]
#define _z(i,j) z[(size_t)(j)*ldz+(i)]

void starneig_eigvec_gen_find_tilings(
    int m, int mb, int nb, double *s, size_t lds, int *select, int **ptr1,
    int **ptr2, int **ptr3, int **ptr4, int **ptr5, int *num1, int *num2)
{

    // MEM's left looking array
    int *l=(int *)malloc(m*sizeof(int));
    starneig_eigvec_gen_find_left(m, s, lds, l);

    // Global index of all selected eigenvalues
    int n=starneig_eigvec_gen_count_selected(m, l, select);
    int *map=(int *)malloc(n*sizeof(int));
    n=starneig_eigvec_gen_find_selected(m, l, select, map);

    // Practical tiling of S, T
    int M=divceil(m,mb);
    int *ap=(int *)malloc((M+1)*sizeof(int));
    int numRows=starneig_eigvec_gen_practical_row_tiling(m, mb, l, ap);

    // Induced column tiling of X, Y
    int *bp=(int *)malloc((numRows+1)*sizeof(int));
    starneig_eigvec_gen_induced_column_tiling(m, select, l, numRows, ap, bp);

    // Practical column tiling of X, Y
    int N=divceil(n,nb);
    int *cp=(int *)malloc((N+1)*sizeof(int *));
    int numCols=starneig_eigvec_gen_practical_column_tiling(n, nb, map, l, cp);

    // Set return variables
    *ptr1=l; *ptr2=map;
    *ptr3=ap; *ptr4=bp; *ptr5=cp;
    *num1=numRows; *num2=numCols;
}

void starneig_eigvec_gen_mini_block_column_norms(
    int m, int n, double *alphai, double *x, size_t ldx, double *xnorm)
{

    // Work space for LAPACK's infinity norm calculation.
    double *work=(double *)malloc(m*sizeof(double));

    // Column index
    int j=0;

    // Loop over the columns of X recognizing mini-block columns
    while (j<n) {
        if (alphai[j]==0) {
            // Real shift, i.e., single column processing
            xnorm[j] = starneig_eigvec_gen_dlange(
                "I", m, int_one, &_x(0,j), ldx, work);
            // Move forward a single column
            j++;
        } else {
            // Complex shift, i.e., compute infinity norm of two adjacent columns
            xnorm[j] = starneig_eigvec_gen_dlange(
                "I", m, int_two, &_x(0,j), ldx, work);
            // Dublicate the result
            xnorm[j+1]=xnorm[j];
            // Move forward two columns,
            j=j+2;
        }
    }
    // Release work space.
    free(work);
}

int starneig_eigvec_gen_generalised_eigenvalues(
    int m, double *s, size_t lds, double *t, size_t ldt, int *select,
	double *alphar, double *alphai, double *beta)
{

    // Pointers to mini-blocks
    double *sjj;
    double *tjj;

    // Variables needed by DLAG2
    double scale1;
    double scale2;
    double wr1;
    double wr2;
    double wi;

    // Column index
    int j=0;

    // Initialize number of selected eigenvalues/vectors
    int k=0;

    // Loop over columns of S, T
    while (j<m) {
        if (j<m-1) {
            // Check for 2-by-2 block
            if (_s(j+1,j)!=0) {
		// Current block is 2-by-2
		if ((select[j]==1) || (select[j+1]==1)) {
	  	  // Find eigenvalues with DLAG2
	  	  sjj=&_s(j,j); tjj=&_t(j,j);
	  	  starneig_eigvec_gen_dlag2(sjj, lds, tjj, ldt,
				smin, &scale1, &scale2, &wr1, &wr2, &wi);

	  	  // Copy values into output arrays
	  	  alphar[k+0]=wr1; alphai[k+0]= wi; beta[k+0]=scale1;
	  	  alphar[k+1]=wr2; alphai[k+1]=-wi; beta[k+1]=scale2;
	  	  // Increase eigenvalue count: +2
	  	  k=k+2;
		}
		// Always advance to the next mini-block: +2
		j=j+2;
            } else {
		// Current block is 1-by-1
		if (select[j]==1) {
	  	  // Copy values into output arrays
	  	  alphar[k]=_s(j,j); alphai[k]=0; beta[k]=_t(j,j);
	  	  // Increase eigenvalue count: +1
	  	  k++;
		}
		// Always advance to the next mini-block: +1
		j++;
            }
        } else { // Last column: j=m-1
            // Current block is 1-by-1
            if (select[j]==1) {
		// Copy values into output arrays
		alphar[k]=_s(j,j); alphai[k]=0; beta[k]=_t(j,j);
		// Increase eigenvalue count : +1
		k++;
            }
            // Always advance to the next mini-block +1
            j++;
        }
    }
    // Return the number of selected eigenvalues
    return k;
}

int starneig_eigvec_gen_multi_shift_update(
    int m, int n, int k, double *s, size_t lds, double *t, size_t ldt,
	double *alphar, double *alphai, double *beta, double *x, size_t ldx,
	double *y, size_t ldy)
{
    /* Performs the linear update

          Y = Y - (S*X*D - T*X*B)

          where D is a diagonal matrix and B is a block diagonal matrix.

          INPUT:
          m, n, k   the dimension of the problem
          Y is m by n
          S is m by k
          X is k by n
          D is n by n (diagonal matrix)
          T is m by k
          B is n by n (mini-block diagonal matrix)

          If alphai(j)!=0, then

          D(j,j) = D(j+1,j+1) = beta(j)

          |B( j , j ) B( j ,j+1)|   | alphar(j)  alphai(j)|
          |B(j+1, j ) B(j+1,j+1)|   |-alphai(j)  alphar(j)|

          If  alphai(j)=0, then

          D(j,j) = beta(j)
          B(j,j) = alphar(j)

          Therefore, some portions of alphar, beta are not accessed.

          ALGORITHM:

          1) Z=X*D      column scalings with beta's
          2) Y=Y-S*Z    DGEMM
          3) Z=X*B      block column scalings with alphar, alphai
          4) Y=Y+T*Z    DGEMM

          REMARK:
          a) The mini-block structure of S, T is not exploited.
          b) It would be possible to treat S, T as block Hessenberg matrices

    */

    // 2-by-2 matrix for representing complex shifts
    size_t ldb=2; double b[4]; int ln=2;

    // Allocate space for matrix Z: k-by-n matrix
    size_t ldz=MAX(k,1); double *z=(double *)malloc(ldz*n*sizeof(double));

    // Copy X into Z
    starneig_eigvec_gen_dlacpy("A", k, n, x, ldx, z, ldz);

    // Start at the first column of Z
    int j=0;

    // Assume that eigenvalues where computed using DLAG2
    for (int j=0; j<n; j++) {
        // Scale the jth column using beta(j)
        dscal_(&k, &beta[j], &_z(0,j), &int_one);
    }

    // Compute Y=Y-S*Z
    starneig_eigvec_gen_dgemm("N", "N", m, n, k, double_minus_one,
		s, lds,
		z, ldz, double_one,
		y, ldy);

    // Copy X into Z
    starneig_eigvec_gen_dlacpy("A", k, n, x, ldx, z, ldz);

    // Compute Z=Z*B:
    // Start at first column of Z
    // Please note that in one branch Z is overwritten, in the other is updated.
    j=0;
    while (j<n) {
        if (alphai[j]!=0) { // Complex shift
            // Construct 2-by-2 matrix representing complex shift
            b[0]= alphar[j]; b[2]=alphai[j];
            b[1]=-alphai[j]; b[3]=alphar[j];
            // Update columns j and j+1 using the complex shift
            starneig_eigvec_gen_dgemm("N", "N", k, ln, ln,
	    	    double_one, &_x(0,j), ldx, b, ldb,
	    	    double_zero, &_z(0,j), ldz);
            // Advance +2 columns
            j=j+2;
        } else { // Real shift
            // Update column j using the real shift
            dscal_(&k, &alphar[j], &_z(0,j), &int_one);
            // Advance +1 column
            j=j+1;
        }
    }

    // Compute Y=Y+T*Z
    starneig_eigvec_gen_dgemm("N", "N", m, n, k,
	 	 double_one,
		t, ldt,
		z, ldz, double_one,
		y, ldy);

    // Free the workspace
    free(z);

    // Dummy return code
    return 0;
}

double starneig_eigvec_gen_relative_residual(
    int m, int n, double *s, size_t lds, double *t, size_t ldt,
	double *alphar, double *alphai, double *beta, double *x, size_t ldx,
	double *f, size_t ldf, double *rres)
{
    // LAPACK workspace
    double *work=(double *)malloc(m*sizeof(double));

    // Compute infinity norms of matrices S, T
    double snorm=starneig_eigvec_gen_dlange("I", m, m, s, lds, work);
    double tnorm=starneig_eigvec_gen_dlange("I", m, m, t, ldt, work);

    // Allocate space for residual R
    size_t ldr=m; double* r=(double *)malloc(ldr*n*sizeof(double));

    // Copy F into R
    starneig_eigvec_gen_dlacpy("A", m, n, f, ldf, r, ldr);

    // Calculate residual
    starneig_eigvec_gen_multi_shift_update(
        m, n, m, s, lds, t, ldt, alphar, alphai, beta, x, ldx, r, ldr);

    // Allocate space for mini-block column norms
    double *rnorm=(double *)malloc(n*sizeof(double));

    // Compute mini-block norms
    starneig_eigvec_gen_mini_block_column_norms(m, n, alphai, r, ldr, rnorm);

    // Allocate space for mini-block column norms of X
    double *xnorm=(double *)malloc(n*sizeof(double));

    // Compute mini-block norms
    starneig_eigvec_gen_mini_block_column_norms(m, n, alphai, x, ldx, xnorm);

    // Allocate space for mini-block column norms of F
    double *fnorm=(double *)malloc(n*sizeof(double));

    // Compute mini-block norms
    starneig_eigvec_gen_mini_block_column_norms(m, n, alphai, f, ldf, fnorm);

    // Loop over all columns calculating the relative residual
    double rc=0;
    for (int j=0; j<n; j++) {
        double aux=fabs(alphar[j])+fabs(alphai[j]);
        rres[j]=rnorm[j]/( (beta[j]*snorm+aux*tnorm)*xnorm[j] + fnorm[j]);
        rc=MAX(rc, rres[j]);
    }

    // free memory
    free(work); free(r); free(rnorm); free(xnorm); free(fnorm);

    // Return the largest relative residual separately
    return rc;
}

#undef _s
#undef _t
#undef _x
#undef _y
#undef _z
