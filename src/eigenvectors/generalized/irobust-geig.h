///
/// @file
///
/// @brief Header file
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

#ifndef IROBUST_GEIG_H_
#define IROBUST_GEIG_H_

#include <stddef.h>

// Obtain a consist scaling of a tile of matrix
void starneig_IntConsistentScaling(int m, int n, int k,
			  double *a, size_t lda,
			  int *scal, size_t lds, int idx);

// Scale matrix to ensure mini-block columns are less than Omega.
void starneig_IntMiniBlockColumnNormsAndScalings(int m, int n, double *alphai,
					double *x, size_t ldx,
					int *xscal, double *xnorm);

// Robust solution of a specialize linear system with a "single" right-hand side
int starneig_IntRobustSingleShiftSolve(int m,
			      double *s, size_t lds, double *cs,
			      double *t, size_t ldt, double *ct,
			      int *blocks, int numBlocks,
			      double alphar, double alphai, double beta,
			      double *f, size_t ldf, int *scal, double *norm,
			      double *work);

// Wrapper for RobustSingleShiftSolve
int starneig_IntRobustMultiShiftSolve(int m, int n,
			     double *s, size_t lds, double *cs,
			     double *t, size_t ldt, double *ct,
			     int *blocks, int numBlocks,
			     double* alphar, double* alphai, double* beta,
			     double *f, size_t ldf, int *scal, double *norm);

// Robust linear update Y:=Y-(S*X*D-T*X*B)
int starneig_IntRobustMultiShiftUpdate(int m, int n, int k,
			      double *s, size_t lds, double snorm,
			      double *t, size_t ldt, double tnorm,
			      double *alphar, double *alphai, double *beta,
			      double *x, size_t ldx, int *xscal, double *xnorm,
			      double *y, size_t ldy, int *yscal, double *ynorm);


// Needed for StarPU
void starneig_irobust_solve_task(int m, int n,
			double *s, size_t lds, double *cs,
			double *t, size_t ldt, double *ct,
			int *blocks, int numBlocks,
			double *alphar, double *alphai, double *beta, int *map,
			int ap0, int ap1, int bp0, int bp1, int cp0, int cp1,
			double *y, size_t ldy, int *yscal, double *ynorm,
			double *work);


// Needed for StarPU
void starneig_irobust_update_task(int m, int n, int k,
			 double *s, size_t lds, double snorm,
			 double *t, size_t ldt, double tnorm,
			 double *alphar, double *alphai, double *beta,
			 int bp0, int bp1, int cp0, int cp1,
			 double *x, size_t ldx, int *xscal, double *xnorm,
			 double *y, size_t ldy, int *yscal, double *ynorm);


#endif
