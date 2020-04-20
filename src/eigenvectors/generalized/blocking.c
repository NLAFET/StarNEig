///
/// @file
///
/// @brief Count and locate mini-blocks of quasi-upper triangular matrices.
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
#include "blocking.h"

// This macro ensures that addresses are computed as size_t
#define _a(i,j) a[(size_t)(j)*lda+(i)]

int starneig_eigvec_gen_count_blocks(int m, double *a, size_t lda)
{
    // Column index
    int j=0;
    // Block index
    int k=0;

    // Loop over the columns of A
    while (j<m) {
        if (j<m-1) {
            // Check subdiagonal entry
            if (_a(j+1,j)!=0) {
                // 2 by 2 block detected
                j=j+2;
            } else {
                // 1 by 1 block detected
                j++;
            }
        } else {
            // We have j=m-1, i.e., last column
            j++;
        }
        // Increment the number of blocks detected
        k++;
    }
    // Return the number of blocks detected
    return k;
}

int starneig_eigvec_gen_find_blocks(
    int m, double *a, size_t lda, int *blocks, int numBlocks)
{
    // Column pointer
    int j=0;
    // Block pointer
    int k=0;

    // The first block starts at the first column;
    blocks[0]=0;

    // Find at most numBlocks blocks
    while ((j<m) && (k<numBlocks)) {
        if (j<m-1) {
            // Check subdiagonal entry A(j+1,j)
            if (_a(j+1,j)!=0) {
                // 2-by-2 block found
                j=j+2;
            } else {
                // 1-by-1 block found
                j++;
            }
        } else {
            // j=m-1, last column of the matrix
            j++;
        }
        // Mark start of next block
        blocks[k+1]=j;
        // Move to next block
        k++;
    }
    // Return the number of blocks found
    return k;
}

#undef _a
