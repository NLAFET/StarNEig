///
/// @file
///
/// @brief Generalised eigenvectors from real Schur forms: tiled
///
/// @author Carl Christian K. Mikkelsen (spock@cs.umu.se), Umeå University
///
/// @section LICENSE
///
/// Copyright (c) 2019, Umeå Universitet
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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "common.h"
#include "tiling.h"
#include "geig.h"

// This macro ensures that addresses are computed as size_t
#define _s(i,j) s[(size_t)(j)*lds+(i)]
#define _t(i,j) t[(size_t)(j)*ldt+(i)]
#define _x(i,j) x[(size_t)(j)*ldx+(i)]
#define _y(i,j) y[(size_t)(j)*ldy+(i)]
#define _z(i,j) z[(size_t)(j)*ldz+(i)]

///
/// @brief Auxiliary routines which finds all information related to tilings
///
/// @param[in] m the dimension of the problem
/// @param[in] mb number of rows per block of Y, target value
/// @param[in] nb number of columns per block of Y, target value
/// @param[in] s array containing matrix S
/// @param[in] lds leading dimension of array s
/// @param[out] ptr1 pointer to left looking array
/// @param[out] ptr2 pointer to map of selected eigenvalues
/// @param[out] ptr3 pointer to practical row tiling
/// @param[out] ptr4 pointer to induced column tiling
/// @param[out] ptr5 pointer to practical column tiling
///
void FindTilings(int m, int mb, int nb,
		 double *s, size_t lds, int *select,
		 int **ptr1, int **ptr2,
		 int **ptr3, int **ptr4, int **ptr5, int *num1, int *num2)
{

  // MEM's left looking array
  int *l=(int *)malloc(m*sizeof(int)); FindLeft(m, s, lds, l);

  // Global index of all selected eigenvalues
  int n=CountSelected(m, l, select);
  int *map=(int *)malloc(n*sizeof(int));
  n=FindSelected(m, l, select, map);

  // Practical tiling of S, T
  int M=divceil(m,mb);
  int *ap=(int *)malloc((M+1)*sizeof(int));
  int numRows=PracticalRowTiling(m, mb, l, ap);

  // Induced column tiling of X, Y
  int *bp=(int *)malloc((numRows+1)*sizeof(int));
  InducedColumnTiling(m, select, l, numRows, ap, bp);

  // Practical column tiling of X, Y
  int N=divceil(n,nb);
  int *cp=(int *)malloc((N+1)*sizeof(int *));
  int numCols=PracticalColumnTiling(n, nb, map, l, cp);

  // Set return variables
  *ptr1=l; *ptr2=map;
  *ptr3=ap; *ptr4=bp; *ptr5=cp;
  *num1=numRows; *num2=numCols;
}



///
/// @brief Mini-block column norms of a matrix
///
/// @param[in] m number of rows of matrix X
/// @param[in] n number of columns of matrix Y
/// @param[in] alphai array of real numbers which dictate column structure
/// @param[in] x array containing the matrix X
/// @param[in] ldx leading dimension of the array x
/// @param[out] xnorm array of infinity norms of the mini-block columns of X
///
void MiniBlockColumnNorms(int m, int n, double *alphai,
			  double *x, size_t ldx, double *xnorm)
{

  // Work space for LAPACK's infinity norm calculation.
  double *work=(double *)malloc(m*sizeof(double));

  // Column index
  int j=0;

  // Loop over the columns of X recognizing mini-block columns
  while (j<n) {
    if (alphai[j]==0) {
      // Real shift, i.e., single column processing
      xnorm[j]=dlange("I", m, int_one, &_x(0,j), ldx, work);
      // Move forward a single column
      j++;
    } else {
      // Complex shift, i.e., compute infinity norm of two adjacent columns
      xnorm[j]=dlange("I", m, int_two, &_x(0,j), ldx, work);
      // Dublicate the result
      xnorm[j+1]=xnorm[j];
      // Move forward two columns,
      j=j+2;
    }
  }
  // Release work space.
  free(work);
}


///
/// @brief Computes selected generalised eigenvalues from pencil (S,T) in real
/// Schur form
///
/// The jth eigenvalue is lambda[j] = (alphar[j] + i*alphai[j])/beta[j].
/// Complex conjugate eigenvalues are stored next to each other.
/// If lambda[j] and lambda[j+1] are a pair of complex conjugate eigenvalues,
/// then beta[j+1]=beta[j], alphai[j+1]=-alphai[j] and alphar[j+1]=alphar[j].
/// If alphai[j]=0 then lambda[j] is real. In this case is beta[j]=0 possible.
/// This corresponds to an infinite eigenvalue.
///
/// @param[in] m  dimension of matrices S, T
/// @param[in] s  array containing matrix S
/// @param[in] lds  leading dimension of array s
/// @param[in] t  array containing matrix T
/// @param[in] ldt  leading dimension of array t
/// @param[in] select  LAPACK style selection array of length m
/// @param[out]  alphar array of length m
/// @param[out]  alphai array of length m
/// @param[out]  beta array of length m
///
int GeneralisedEigenvalues(int m,
			   double *s, size_t lds,
			   double *t, size_t ldt,
			   int *select,
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
	  dlag2(sjj, lds, tjj, ldt,
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


///
/// @brief Performs the multishift linear update Y:=Y-(S*X*D-T*X*B)
///
/// @param[in] m  number of rows of S, T, Y.
/// @param[in] n  number of shifts and number of columns of Y.
/// @param[in] k  number of columns of S and T, number of rows of X.
/// @param[in] s  array containing the matrix S.
/// @param[in] lds  leading dimension of s.
/// @param[in] t  array containing the matrix T.
/// @param[in] ldt  leading dimension of t.
/// @param[in] alphar  array of length at least n.
/// @param[in] alphai  array of length at least n.
/// @param[in] beta  array of length at least n.
/// @param[in] x  array containing the matrix X.
/// @param[in] ldx  leading dimension of array x.
/// @param[in,out] y  array containing matrix Y.
///         On entry, the original value of Y.
///         On exit, overwritten by the updated value of Y.
/// @param[in] ldy leading dimension of array y.
///
int MultiShiftUpdate(int m, int n, int k,
		     double *s, size_t lds,
		     double *t, size_t  ldt,
		     double *alphar, double *alphai, double *beta,
		     double *x, size_t ldx,
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
  size_t ldz=max(k,1); double *z=(double *)malloc(ldz*n*sizeof(double));

  // Copy X into Z
  dlacpy("A", k, n, x, ldx, z, ldz);

  // Start at the first column of Z
  int j=0;

  // Assume that eigenvalues where computed using DLAG2
  for (int j=0; j<n; j++) {
    // Scale the jth column using beta(j)
    dscal_(&k, &beta[j], &_z(0,j), &int_one);
  }

  // Compute Y=Y-S*Z
  dgemm("N", "N", m, n, k, double_minus_one,
	s, lds,
	z, ldz, double_one,
	y, ldy);

  // Copy X into Z
  dlacpy("A", k, n, x, ldx, z, ldz);

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
      dgemm("N", "N", k, ln, ln,
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
  dgemm("N", "N", m, n, k,
	 double_one,
	t, ldt,
	z, ldz, double_one,
	y, ldy);

  // Free the workspace
  free(z);

  // Dummy return code
  return 0;
}


double RelativeResidual(int m, int n,
			double *s, size_t lds,
			double *t, size_t ldt,
			double *alphar, double *alphai, double *beta,
			double *x, size_t ldx,
			double *f, size_t ldf,
			double *rres)
{
  // LAPACK workspace
  double *work=(double *)malloc(m*sizeof(double));

  // Compute infinity norms of matrices S, T
  double snorm=dlange("I", m, m, s, lds, work);
  double tnorm=dlange("I", m, m, t, ldt, work);

  // Allocate space for residual R
  size_t ldr=m; double* r=(double *)malloc(ldr*n*sizeof(double));

  // Copy F into R
  dlacpy("A", m, n, f, ldf, r, ldr);

  // Calculate residual
  MultiShiftUpdate(m, n, m, s, lds, t, ldt, alphar, alphai, beta,
		   x, ldx, r, ldr);

  // Allocate space for mini-block column norms
  double *rnorm=(double *)malloc(n*sizeof(double));

  // Compute mini-block norms
  MiniBlockColumnNorms(m, n, alphai, r, ldr, rnorm);

  // Allocate space for mini-block column norms of X
  double *xnorm=(double *)malloc(n*sizeof(double));

  // Compute mini-block norms
  MiniBlockColumnNorms(m, n, alphai, x, ldx, xnorm);

  // Allocate space for mini-block column norms of F
  double *fnorm=(double *)malloc(n*sizeof(double));

  // Compute mini-block norms
  MiniBlockColumnNorms(m, n, alphai, f, ldf, fnorm);

  // Loop over all columns calculating the relative residual
  double rc=0;
  for (int j=0; j<n; j++) {
    double aux=fabs(alphar[j])+fabs(alphai[j]);
    rres[j]=rnorm[j]/( (beta[j]*snorm+aux*tnorm)*xnorm[j] + fnorm[j]);
    rc=maxf(rc, rres[j]);
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
