///
/// @file
///
/// @brief Overflow protection of some BLAS routines; power of 2 scaling
/// scaling factors.
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
#include "utils.h"
#include "common.h"
#include "robust.h"
#include "irobust.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>

// This macro ensures that addresses are computed as size_t
#define _a(i,j) a[(size_t)(j)*lda+(i)]
#define _x(i,j) x[(size_t)(j)*ldx+(i)]
#define _y(i,j) y[(size_t)(j)*ldy+(i)]
#define _z(i,j) z[(size_t)(j)*ldz+(i)]

int starneig_eigvec_gen_int_protect_division(double b, double t)
{
    double alpha = starneig_eigvec_gen_protect_division(b, t);
    return ilogb(alpha);
}

int starneig_eigvec_gen_int_protect_update(double t, double x, double y)
{
    double alpha = starneig_eigvec_gen_protect_update(t, x, y);
    return ilogb(alpha);
}

void starneig_eigvec_gen_int_robust_scaling(
    double alpha, int m, int n, double *x, size_t ldx, int *xscal,
    double *xnorm)

/* TODO: Determine if it is safe to combine the two scalings into one.
      This is problematic because if alpha is large, then you may want to scale
      alpha, and if alpha is small, then you might want to scale xij.
      The problem can be elimated by using integer based scaling factors.
      Simply multiply the significands, adjust the individual exponents by at most
      1 and adjust the overall scaling factor using the exponent of alpha.
*/

{
    for (int j=0; j<n; j++) {
        // Compute scaling to survive scaling of jth column
        int k=starneig_eigvec_gen_int_protect_update(fabs(alpha),xnorm[j],0);
        if (k<0) {
            // Construct the scaling factor gamma = 2^k<1
            double gamma=scalbn(1,k);
            // Scale the jth vector by gamma
            dscal_(&m, &gamma, &_x(0,j), &int_one);
            // Accumulate the scaling factor
            xscal[j]=xscal[j]+k;
            // Scale the norm of the jth vector by gamma
            xnorm[j]=xnorm[j]*gamma;
        }
    }
    // It is now safe to scale X by alpha
    for (int j=0; j<n; j++) {
        dscal_(&m, &alpha, &_x(0,j), &int_one);
        xnorm[j]=xnorm[j]*alpha;
    }
}

void starneig_eigvec_gen_int_robust_update(
    int m, int n, int k, double alpha, double *a, size_t lda, double anorm,
	double *x, size_t ldx, int *xscal, double *xnorm, double beta, double *y,
    size_t ldy, int *yscal, double *ynorm)
{
    /* The algorithm is

          0: Copy Z:=X
          1: Robust computation Y:=beta*Y
          2: Robust computation Z:=alpha*Z;
          3: Robust Y:=Y+A*Z

          Step 0 ensures that X can be read-only in StarPU.
          There is more than one way of computing Y = beta*Y + alpha*A*X.
          Steps 1 and 2 removes the freedom of choice from LAPACK.
          Moreover, ProtectUpdate ensures that norm(Y)+norm(A)*norm(Z) <= Omega.
          This renders the order of the aritmhetic operations irrelevant.

    */

    // ************************************************************************
    // STEP 0: Make copy of Z:=X
    // ************************************************************************

    // Create a copy Z = X: k by n matrix
    size_t ldz=MAX(k,1); double *z=malloc(ldz*n*sizeof(double));
    starneig_eigvec_gen_dlacpy("A", k, n, x, ldx, z, ldz);

    // Copy norms and scalings
    int *zscal=(int *)malloc(n*sizeof(double));
    double *znorm=(double *)malloc(n*sizeof(double));
    for (int j=0; j<n; j++) {
        zscal[j]=xscal[j];
        znorm[j]=xnorm[j];
    }

    // ************************************************************************
    // STEP 1: Robust computation of Y:=beta*Y
    // ************************************************************************
    if (beta!=1)
        starneig_eigvec_gen_int_robust_scaling(beta, m, n, y, ldy, yscal, ynorm);

    // ************************************************************************
    // STEP 2: Robust computation of Z:=alpha*Z
    // ************************************************************************
    if (alpha!=1)
        starneig_eigvec_gen_int_robust_scaling(alpha, k, n, z, ldz, zscal, znorm);

    // ************************************************************************
    // STEP 3 Robust computation of Y:=Y+A*Z
    // ************************************************************************

    // Loop over the columns of Y and Z
    for (int j=0; j<n; j++) {
        // Determine a consistent scaling
        int p=MIN(yscal[j],zscal[j]);
        // Calculate rescaling factor
        long k1=p-yscal[j];
        long k2=p-zscal[j];
        double aux1=scalbln(1,k1);
        double aux2=scalbln(1,k2);
        // Implicitly rescale columns Y(:,j) and Z(:,j) to consistent scaling
        ynorm[j]=ynorm[j]*aux1;
        znorm[j]=znorm[j]*aux2;
        // Determine scaling needed to survive update Y(:,j)=Y(:,j)+A*Z(:,j)
        int q=starneig_eigvec_gen_int_protect_update(anorm, znorm[j], ynorm[j]);
        double delta=scalbn(1,q);
        // Update rescaling factors to include overflow protection
        double aux3=aux1*delta;
        double aux4=aux2*delta;
        // Scale column Y(:,j); rescaling and overflow protection
        dscal_(&m, &aux3, &_y(0,j), &int_one);
        // Record the new scaling factor
        yscal[j]=p+q;
        // Scale column Z(:,j); rescaling and overflow protection
        dscal_(&k, &aux4, &_z(0,j), &int_one);
        // By design Y(:,j) and Z(:,j) have the *same* scaling factor
        zscal[j]=yscal[j];
    }
    // Do the linear update Y:=A*Z+Y
    starneig_eigvec_gen_dgemm("N", "N", m, n, k,
	 	 double_one, a, lda, z, ldz,
	 	 double_one, y, ldy);

    // Free memory
    free(z); free(zscal); free(znorm);

    // The final computation of the norms is omitted.
    // In general, it depends on the structure imposed on Y.
}

#undef _a
#undef _x
#undef _y
#undef _z
