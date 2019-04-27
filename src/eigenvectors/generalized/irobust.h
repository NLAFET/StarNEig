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

#ifndef IROBUST_H_
#define IROBUST_H_

#include <stddef.h>

// Protect a scalar division against overflow
int IntProtectDivision(double b, double t);

// Protect a linear update againt overflow
int IntProtectUpdate(double t, double x, double y);

// Robust scaling of a matrix
void IntRobustScaling(double alpha, int m, int n,
		      double *x, size_t ldx, int *xscal, double *xnorm);

// Robust linear update Y:=alpha*A*X+beta*Y
void IntRobustUpdate(int m, int n, int k,
		     double alpha,
		     double *a, size_t lda, double anorm,
		     double *x, size_t ldx, int *xscal, double *xnorm,
		     double beta,
		     double *y, size_t ldy, int *yscal, double *ynorm);

#endif
