///
/// @file
///
/// @brief Generalised eigenvectors from real Schur forms: tiled, robust,
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

// This macro ensures that addresses are computed as size_t
#define _a(i,j) a[(size_t)(j)*lda+(i)]
#define _x(i,j) x[(size_t)(j)*ldx+(i)]
#define _y(i,j) y[(size_t)(j)*ldy+(i)]
#define _z(i,j) z[(size_t)(j)*ldz+(i)]
#define _r(i,j) r[(size_t)(j)*ldr+(i)]
#define _s(i,j) s[(size_t)(j)*lds+(i)]
#define _t(i,j) t[(size_t)(j)*ldt+(i)]
#define _f(i,j) f[(size_t)(j)*ldf+(i)]

void starneig_eigvec_gen_int_consistent_scaling(
    int m, int n, int k, double *a, size_t lda, int *scal, size_t lds, int idx)
{

    // Loop over the columns of tile A, matrix SCAL
    for (int j=0; j<n; j++) {
        int temp=0;
        // Determine the smallest scaling factor attached to jth column
        for (int i=0; i<k; i++)
            temp=MIN(temp,scal[i*lds+j]);
        // Construct log2 of rescaling factor
        long aux=temp-scal[idx*lds+j];
        // Construct rescaling factor
        double alpha=scalbln(1,aux);
        // Apply rescaling
        dscal_(&m, &alpha, &_a(0,j), &int_one);
    }
}

void starneig_eigvec_gen_irobust_solve_task(
    int m, int n, double *s, size_t lds, double *cs, double *t, size_t ldt,
    double *ct, int *blocks, int numBlocks, double *alphar, double *alphai,
    double *beta, int *map, int ap0, int ap1, int bp0, int bp1, int cp0,
    int cp1, double *y, size_t ldy, int *yscal, double *ynorm, double *work)
{


    // LAPACK variables
    int select[m];
    int info;

    // Determine start of eigenvector computation (global index)
    int p0=MAX(bp0,cp0);
    // Determine end+1 of eigenvector computation (global index)
    int p1=MIN(bp1,cp1);
    // Are there any eigenvectors to initialize
    if (p0<p1) {
        // Clear selection array of garbage
        for (int j=0; j<m; j++)
            select[j]=0;
        // Identify the selected eigenvalues
        for (int j=p0; j<p1; j++)
            select[map[j-cp0]-ap0]=1;

        // Determine start of eigenvector computation relative to tile
        int idx=p0-cp0;

        // Number of columns available to LAPACK
        int n1=n-idx;

        // Number of columns returned by LAPACK
        int l;

        // Determine number of eigenvectors to compute
        int k=p1-p0;

        // Initialize k eigenvectors in columns idx:idx+k-1
        starneig_eigvec_gen_dtgevc("R", "S", select, m, s, lds, t, ldt,
	   	   &_y(0,idx), ldy, &_y(0,idx), ldy,
	    	    n1, &l, work, &info);

        // Apply scalings as needed
        starneig_eigvec_gen_int_mini_block_column_norms_and_scalings(
            m, k, &alphai[idx], &_y(0,idx), ldy, &yscal[idx], &ynorm[idx]);
    }

    // Determine start of multi-shift region (global index)
    int q0=MAX(bp1,cp0);
    // Determine end+1 of multi-shift region (global index)
    int q1=cp1;
    // Are there any multi-shift linear solves to complete
    if (q0<q1) {
        // Yes. Isolate the number of shifts
        int k=q1-q0;
        // Determine start of the multi-shift region relative to tile
        int idx=q0-cp0;
        // Do the multi-shift linear solve
        starneig_eigvec_gen_int_robust_multi_shift_solve(m, k,
			     			     s, lds, cs,
			     			     t, ldt, ct,
			     			     blocks, numBlocks,
			     			     &alphar[idx], &alphai[idx], &beta[idx],
			     			     &_y(0,idx), ldy, &yscal[idx], &ynorm[idx]);
    }
}

void starneig_eigvec_gen_irobust_update_task(
    int m, int n, int k, double *s, size_t lds, double snorm, double *t,
    size_t ldt, double tnorm, double *alphar, double *alphai, double *beta,
	int bp0, int bp1, int cp0, int cp1, double *x, size_t ldx, int *xscal,
    double *xnorm, double *y, size_t ldy, int *yscal, double *ynorm)
{
    // Determine start of update region (global index)
    int p0=MAX(bp0,cp0);
    // Determine end+1 of update region (global index)
    int p1=cp1;
    // Are there any vectors to update?
    if (p0<p1) {
        // Yes. Determine their number
        int num=p1-p0;
        // Determine start of update region relative to start of tile
        int idx=p0-cp0;
        // Do the multi-shift update
        starneig_eigvec_gen_int_robust_multi_shift_update(m, num, k,
			      			      s, lds, snorm,
			      			      t, ldt, tnorm,
			      			      &alphar[idx], &alphai[idx], &beta[idx],
			      			      &_x(0,idx), ldx, &xscal[idx], &xnorm[idx],
			      			      &_y(0,idx), ldy, &yscal[idx], &ynorm[idx]);
    }
}

void starneig_eigvec_gen_int_mini_block_column_norms_and_scalings(
    int m, int n, double *alphai, double *x, size_t ldx, int *xscal,
    double *xnorm)

{
    // Compute mini-block column norms of X on entry
    starneig_eigvec_gen_mini_block_column_norms(m, n, alphai, x, ldx, xnorm);

    // Loop over the columns of X scaling as needed.
    // Recall: norms are replicated if two columns form a mini-block column.
    for (int j=0; j<n; j++) {
        if (xnorm[j]>Omega) {
            // Scaling is needed
            int p=ilogb(Omega/xnorm[j]);
            double aux=scalbn(1,p);
            // Apply scaling
            dscal_(&m, &aux, &_x(0,j), &int_one);
            // Update scaling factor
            xscal[j]=xscal[j]+p;
            // Set new norm
            xnorm[j]=xnorm[j]*aux;
        }
    }
}

int starneig_eigvec_gen_int_robust_single_shift_solve(
    int m, double *s, size_t lds, double *cs, double *t, size_t ldt, double *ct,
	int *blocks, int numBlocks, double alphar, double alphai, double beta,
	double *f, size_t ldf, int *scal, double *norm, double *work)
{

    // Solves equation SXD-TXB=F for X and very special S, T, D, B

    // The original scaling factor of the right-hand side is irrelevant.
    // We compute any additional scalings and do an update at the very end
    int fscal=0;

    // The infinity norm of the top portion of the right-hand side.
    double fnorm;

    // Different scalings
    double gamma0;
    double gamma1;
    double gamma2;
    double gamma3;
    double gamma4;

    // Matrix B is either a 1-by-1 or a 2-by-2 matrix.
    double b[4]; size_t ldb=2; double bnorm;

    // The number of right-hand sides
    int ln;
    if (alphai==0) {
        ln=1;  // Real shift, RHS has 1 column
    } else {
        ln=2;  // Complex shift, RHS has 2 columns
    }

    // Define matrix B. We will only access the first ln rows/colums.
    b[0]= alphar; b[2]=alphai;
    b[1]=-alphai; b[3]=alphar;

    // Computes the infinity norm of the matrix B regardless of the value of ln.
    bnorm=fabs(alphar)+fabs(alphai);

    // Solution of local system
    double x[4]; size_t ldx=2; double xnorm;

    // WARNING: LAPACK's scaling factor is a DP number
    // We will scale further down to the nearest power of 2 scaling factor.
    double xscal;

    // Auxiliary array y used to scale matrix X with matrix D
    double y[4]; size_t ldy=2; double ynorm;

    // Auxuliary array z used to scale matrix X with matrix B
    double z[4]; size_t ldz=2; double znorm;

    // Diagonal values of mini-blocks of T
    double t11;
    double t22;

    // Variable used by LAPACK DLALN2
    int info;

    // Loop backwards over the mini-block structure of S
    for (int k=numBlocks-1; k>=0; k--) {

        // Determine the index of the current column
        int col=blocks[k];



        // Determine the dimension of the current mini-block
        int dim=blocks[k+1]-blocks[k];

        // *********************************************************************
        // Set values for dlaln2
        // *********************************************************************

        // First diagonal element
        t11=_t(col,col); t22=0;
        if (dim==2) {
            // Second diagonal element
            t22=_t(col+1,col+1);
        }

        // Solve for F(k)
        starneig_eigvec_gen_dlaln2(int_zero, dim, ln, smin, beta,
	   	   &_s(col,col), lds, t11, t22,
	   	   &f[col], ldf,
	   	   alphar, alphai,
	   	   x, ldx, &xscal, &xnorm,
	   	   &info);

        // Ensure consistency between F and X at all times!
        // LAPACK scaling factor is a DP number <= 1.
        if (xscal<1) {
            // We need a power of 2 which is smaller than xscal
            int p=ilogb(xscal);
            // Compute the rescaling factor xi<=1
            volatile double xi=scalbn(1,p)/xscal;
            // Rescale x to scaling factor 2^p
            for (int j=0; j<4; j++) x[j]=x[j]*xi;
            // Update xnorm
            xnorm=xnorm*xi;
            // Scale F by 2^p
            double nu=scalbn(1,p);
            for (int j=0; j<ln; j++)
		dscal_(&m, &nu, &_f(0,j), &int_one);
            // Update scaling for F
            fscal=fscal+p;
        }

        // The documentation does not specify the overflow threshold for DLALN2.
        if (xnorm>Omega) {
            // Determine power of 2 scaling which will reduce xnorm below Omega
            int p0=ilogb(Omega/xnorm); gamma0=scalbn(1,p0);
            for (int j=0; j<4; j++)
		x[j]=x[j]*gamma0;
            // Update norm
            xnorm=xnorm*gamma0;
            // Scale F by gamma0 maintain consistency
            for (int j=0; j<ln; j++)
		dscal_(&m, &gamma0, &_f(0,j), &int_one);
            // Update scaling for F
            fscal=fscal+p0;
        }

        // Copy X into F. Note: X has already been scaled above
        starneig_eigvec_gen_dlacpy("A", dim, ln, x, ldx, &f[col], ldf);

        if (k>0) {
            // **************************************************************
            // F(0:k-1) = F(0:k-1) - beta*S(0:k-1,k)*X + T(0:k-1,k)*X*B
            // **************************************************************

            // Determine the number of rows in the F(0:k-1)
            int lm=blocks[k];

            // Determine infinity norm of F(0:k-1)
            fnorm=starneig_eigvec_gen_dlange("I", lm, ln, f, ldf, work);

            // Protect Y=beta*X against overflow
            int p1=starneig_eigvec_gen_int_protect_update(beta, xnorm, 0);
            if (p1<0) {
		// Scale X and xnorm by gamma1=2^p1
		gamma1=scalbn(1,p1);
		for (int j=0; j<4; j++)
	  	  x[j]=x[j]*gamma1;
		// Update norm
		xnorm=xnorm*gamma1;

		// Scale F by gamma1 to maintain OVERALL consistency
		for (int j=0; j<ln; j++)
	  	  dscal_(&m, &gamma1, &_f(0,j), &int_one);
		// Update scaling and norm
		fscal=fscal+p1;
		fnorm=fnorm*gamma1;
            }

            // Now compute Y=beta*X and ynorm = beta*xnorm without fear of overflow
            for (int j=0; j<4; j++)
		y[j]=beta*x[j];
            // Update norm
            ynorm=xnorm*beta;

            // Protect update F(0:k-1) = F(0:k-1) - S(0:k-1,k)*Y
            int p2=starneig_eigvec_gen_int_protect_update(cs[k],ynorm,fnorm);
            if (p2<0) {
		// Scale Y by gamma2 = 2^p2
		gamma2=scalbn(1,p2);
		for (int j=0; j<4; j++)
	  	  y[j]=y[j]*gamma2;
		// Update scaling
		ynorm=ynorm*gamma2;

		// Scale F by gamma2
		for (int j=0; j<ln; j++)
	  	  dscal_(&m, &gamma2, &_f(0,j), &int_one);
		// Update scaling. The norm will be computed shortly
		fscal=fscal+p2;

      	      	// Scale X and xnorm by gamma2 to maintain OVERALL consistency
		for (int j=0; j<4; j++)
	  	  x[j]=x[j]*gamma2;
		// Update scaling
		xnorm=xnorm*gamma2;
            }
            // Do linear update F(1:k-1) = F(1:k-1) - S(1:k-1,k)*Y without fear
            starneig_eigvec_gen_dgemm("N", "N", lm, ln, dim,
	    	    double_minus_one, &_s(0,col), lds,
	     	     y, ldy, double_one, f, ldf);

            // Recompute fnorm
            fnorm=starneig_eigvec_gen_dlange("I", lm, ln, f, ldf, work);

            // Protect Z=X*B
            int p3=starneig_eigvec_gen_int_protect_update(xnorm, bnorm, 0);
            if (p3<0) {
		// Scale X, xnorm by gamma3 = 2^p3
		gamma3=scalbn(1,p3);
		for (int j=0; j<4; j++)
	  	  x[j]=x[j]*gamma3;
		// Update norm
		xnorm=xnorm*gamma3;

		// Scale F  by gamma3 to maintain OVERALL consistency
		for (int j=0; j<ln; j++)
	  	  dscal_(&m, &gamma3, &_f(0,j), &int_one);
		// Update scaling and norm
		fscal=fscal+p3;
		fnorm=fnorm*gamma3;
            }

            // Compute Z=X*B without fear of overflow
            starneig_eigvec_gen_dgemm("N", "N", dim, ln, ln,
	     	     double_one, x, ldx, b, ldb, double_zero, z, ldz);
            // Compute the norm of Z. The scaling of Z is identical to X
            znorm=starneig_eigvec_gen_dlange("I", dim, ln, z, ldz, work);

            // Protect F(0:k-1) = F(0:k-1) + T(0:k-1,k)*Z
            int p4=starneig_eigvec_gen_int_protect_update(ct[k], znorm, fnorm);
            if (p4<0) {
		// Scale Z by gamma4 = 2^p4
		gamma4=scalbn(1,p4);
		for (int j=0; j<4; j++)
	  	  z[j]=z[j]*gamma4;

		// Scale F by gamma4
		for (int j=0; j<ln; j++)
	  	  dscal_(&m, &gamma4, &_f(0,j), &int_one);
		// Update scaling
		fscal=fscal+p4;
            }
            // Do update F(0:k-1) = F(0:k-1) + T(0:k-1,k)*Z without fear of
            // overflow
            starneig_eigvec_gen_dgemm("N", "N", lm, ln, dim, double_one,
	    	    &_t(0,col), ldt, z, ldz, double_one, f, ldf);

        }
    }
    // Compute the norm of F.
    fnorm=starneig_eigvec_gen_dlange("I", m, ln, f, ldf, work);

    // Set return variables, dublicating results iff the shift was complex.
    for (int j=0; j<ln; j++) {
        // Will we run out of scaling factors?
        if (scal[j]>=INT_MIN - fscal) {
            // It is safe to update
            scal[j]=scal[j]+fscal;
        } else {
            // We have run out of integers, this is an extreme event.
            scal[j]=INT_MIN;
            printf("IROBUST-GEIG: IntRobustSingleShiftSolve reports extreme " \
            "event. Power of 2 scaling factors exhausted");
        }
        norm[j]=fnorm;
    }
    // Dummy return code
    return 0;
}

int starneig_eigvec_gen_int_robust_multi_shift_solve(
    int m, int n, double *s, size_t lds, double *cs, double *t, size_t ldt,
    double *ct, int *blocks, int numBlocks, double* alphar, double* alphai,
    double* beta, double *f, size_t ldf, int *scal, double *norm)
{
    // A wrapper for RobustSingleShiftSolve

    // Allocate workspace. This is used for infinity norm calculation.
    int lwork=MAX(m,4); double *work=(double *)malloc(lwork*sizeof(double));

    // Column index;
    int j=0;
    while (j<n) {
        // Solve for 1 or 2 columns of the RHS
        starneig_eigvec_gen_int_robust_single_shift_solve(m,
			      			      s, lds, cs,
			      			      t, ldt, ct,
			      			      blocks, numBlocks,
			      			      alphar[j], alphai[j], beta[j],
			      			      &_f(0,j), ldf,
			      			      &scal[j], &norm[j], work);
        // Did we solve for 1 or 2 columns
        if (alphai[j]==0) {
            // Real shift, single column
            j++;
        } else {
            // Complex shift, two columns
            j=j+2;
        }
    }

    // Free workspace
    free(work);
    // Dummy return
    return 0;
}

int starneig_eigvec_gen_int_robust_multi_shift_update(
    int m, int n, int k, double *s, size_t lds, double snorm, double *t,
    size_t ldt, double tnorm, double *alphar, double *alphai, double *beta,
	 double *x, size_t ldx, int *xscal, double *xnorm, double *y, size_t ldy,
     int *yscal, double *ynorm)
{
    /* In the absence of overflow protection the algoritm is trivial

          1) Z=X*D      column scalings with beta
          2) Y=Y-S*Z    simple DGEMM
          3) Z=X*B      (block) column scalings with alphar (and alphai)
          4) Y=Y+T*Z    simple DGEMM
    */

    // 2-by-2 matrix for representing pairs of complex conjugate shifts
    size_t ldb=2; double b[4]; int ln=2; double bnorm;

    // Allocate space for matrix Z: k-by-n matrix
    size_t ldz=MAX(k,1); double *z=(double *)malloc(ldz*n*sizeof(double));

    // Norms and scalings of Z
    int *zscal=(int *)malloc(n*sizeof(int));
    double *znorm=(double *)malloc(n*sizeof(double));

    // Copy X into Z
    starneig_eigvec_gen_dlacpy("A", k, n, x, ldx, z, ldz);
    for (int j=0; j<n; j++) {
        zscal[j]=xscal[j]; znorm[j]=xnorm[j];
    }

    // Scale jth column of Z by beta[j]; beta[j+1] = beta[j] for c.c. shifts
    for (int j=0; j<n; j++)
        starneig_eigvec_gen_int_robust_scaling(
            beta[j], k, 1, &_z(0,j), ldz, &zscal[j], &znorm[j]);

    // Compute Y:=Y-S*Z robustly
    starneig_eigvec_gen_int_robust_update(m, n, k,
		  		  double_minus_one,
		  		  s, lds, snorm,
		  		  z, ldz, zscal, znorm,
		  		  double_one,
		  		  y, ldy, yscal, ynorm);

    // Compute norms of mini-block columns of Y
    starneig_eigvec_gen_mini_block_column_norms(m, n, alphai, y, ldy, ynorm);

    // ************************************************************************
    //  This is the midpoint of the computation
    // ************************************************************************

    // Obtain a fresh copy of X. Z:=X
    starneig_eigvec_gen_dlacpy("A", k, n, x, ldx, z, ldz);
    for (int j=0; j<n; j++) {
        zscal[j]=xscal[j]; znorm[j]=xnorm[j];
    }

    // Scale to survive Z=X*B
    for (int j=0; j<n; j++) {
        bnorm=fabs(alphar[j])+fabs(alphai[j]);

        // Determine scaling needed to survive multiplication by B
        int p=starneig_eigvec_gen_int_protect_update(znorm[j],bnorm,0);
        double gamma=scalbn(1,p);

        // Apply scaling
        dscal_(&k, &gamma, &_z(0,j), &int_one);
        // Update scalings
        zscal[j]=zscal[j]+p;
    }

    // Create matrix which will equal Z*B
    size_t ldr=k; double *r=(double *)malloc(ldr*n*sizeof(double));

    // At this point it is safe to compute R:=0*R+Z*B
    int col=0;
    while (col<n) {
        // Construct 2-by-2 matrix representing complex shift
        b[0]= alphar[col]; b[2]=alphai[col];
        b[1]=-alphai[col]; b[3]=alphar[col];
        if (alphai[col]==0) {
            // Real Shift
            ln=1;
        } else {
            // Complex shift
            ln=2;
        }
        // Update ln columns
        starneig_eigvec_gen_dgemm("N", "N", k, ln, ln,
	  	  double_one, &_z(0,col), ldz, b, ldb,
	  	  double_zero, &_r(0,col), ldr);
        // Advance ln columns; ln is either 1 (real shift) or 2 (complex shift)
        col=col+ln;
    }
    // Copy R into Z and continue
    starneig_eigvec_gen_dlacpy("A", k, n, r, ldr, z, ldz);

    // Compute norms of mini-block columns of Z
    starneig_eigvec_gen_mini_block_column_norms(k, n, alphai, z, ldz, znorm);

    // Compute Y = Y + T*Z robustly
    starneig_eigvec_gen_int_robust_update(m, n, k, double_one,
		  		  t, ldt, tnorm,
		  		  z, ldz, zscal, znorm,
		  		  double_one,
		  		  y, ldy, yscal, ynorm);

    // Compute norms of mini-block columns of Y
    starneig_eigvec_gen_mini_block_column_norms(m, n, alphai, y, ldy, ynorm);

    // Free the workspace
    free(z); free(zscal); free(znorm); free(r);

    // Dummy return code
    return 0;
}

#undef _a
#undef _x
#undef _y
#undef _z
#undef _r
#undef _s
#undef _t
#undef _f
