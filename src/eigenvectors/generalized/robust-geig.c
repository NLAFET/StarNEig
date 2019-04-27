///
/// @file
///
/// @brief Generalised eigenvectors from real Schur forms: tiled, robust.
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
#include "robust-geig.h"

// This macro ensures that addresses are computed as size_t
#define _a(i,j) a[(size_t)(j)*lda+(i)]

///
/// @brief Generalised column majorants for quasi-upper triangular matrix A
///
/// Computes the infinity norm of the strictly super-diagonal portion of each
/// mini-block column. A mini-block column is either a single column or a pair
/// of adjacent columns.
///
///        x|xx|x|x|xx|x|x|x|
///        ------------------
///         |xx|x|x|xx|x|x|x|
///         |xx|x|x|xx|x|x|x|
///        ------------------
///         |  |x|x|xx|x|x|x|
///        ------------------
///         |  | |x|xx|x|x|x|
///        ------------------
///         |  | | |xx|x|x|x|
///         |  | | |xx|x|x|x|
///        ------------------
///         |  | | |  |x|x|x|
///        ------------------
///         |  | | |  | |x|x|
///        ------------------
///         |  | | |  | | |x|
///        ------------------
///
///
/// @param[in] m dimension of the matrix
/// @param[in] a array containing the matrix
/// @param[in] lda leading dimension of the array
/// @param[in] blocks offset of all mini-blocks along the main diagonal of A
/// @param[in] numBlocks number of mini-blocks
/// @param[out] ca generalised column majorants for mini-block colmuns of A
///
void starneig_GeneralisedColumnMajorants(int m, double *a, size_t lda,
				int *blocks, int numBlocks, double *ac)
{

  // Work space array needed by LAPACK
  double *work=malloc(m*sizeof(double));

  // The first value is always zero - as nothing is above the first mini-block
  ac[0]=0;

  // Loop over the remaining mini-block columns of matrix A
  for (int k=1; k<numBlocks; k++) {
    // Determine the width of the current mini-block
    int ln=blocks[k+1]-blocks[k];
    // Determine the number of rows above the current mini-blocks
    int lm=blocks[k];
    // Compute infinity norm
    ac[k]=starneig_dlange("I", lm, ln, &_a(0,lm), lda, work);
  }
  // Free workspace
  free(work);
}

#undef _a
