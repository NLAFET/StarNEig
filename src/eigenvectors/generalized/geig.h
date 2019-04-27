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

#ifndef GEIG_GUARD_H_
#define GEIG_GUARD_H_

#include <stddef.h>

// Find all tilings needed for generalized eigenvector computation
void starneig_FindTilings(int m, int mb, int nb,
		 double *s, size_t lds, int *select,
		 int **ptr1, int **ptr2,
		 int **ptr3, int **ptr4, int **ptr5, int *num1, int *num2);

// Compute infinity norms of the mini-block columns of a matrix.
void starneig_MiniBlockColumnNorms(int m, int n, double *alphai,
			  double *x, size_t ldx, double *xnorm);

// Compute eigenvalues from generalised real Schur form.
int starneig_GeneralisedEigenvalues(int m,
			   double *s, size_t lds,
			   double *t, size_t ldt,
			   int *select,
			   double *alphar, double *alphai, double *beta);

// Multishift linear update needed by Relative Residual
int starneig_MultiShiftUpdate(int m, int n, int k,
		     double *s, size_t lds,
		     double *t, size_t ldt,
		     double *alphar, double *alphai, double *beta,
		     double *x, size_t ldx,
		     double *y, size_t ldy);

// Infinity norm relative residual for each mini-block column
double starneig_RelativeResidual(int m, int n,
			double *s, size_t lds,
			double *t, size_t ldt,
			double *alphar, double *alphai, double *beta,
			double *x, size_t ldx,
			double *f, size_t ldf,
			double *rres);

#endif
