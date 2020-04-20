///
/// @file
///
/// @brief Generate tilings for generalised eigenvector problems
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
#include "tiling.h"
#include "common.h"

// This macro ensures that addresses are computed as size_t
#define _a(i,j) a[(size_t)(j)*lda+(i)]

int starneig_eigvec_gen_remove_duplicates(int m, int *l)
{
    int i=0; int j=1;
    while (j<m) {
        if (l[i]<l[j]) {
            // Save the jth entry at the next available position
            l[i+1]=l[j];
            // Move forward
            i++;
        }
        // Move to the next candiate
        j++;
    }
    // Return number of distinct entries
    return i;
}

void starneig_eigvec_gen_find_left(int m, double *a, size_t lda, int *l)
{
    // By definition, the entry to the left of the matrix is zero
    l[0]=0;
    for (int j=1; j<m; j++) {
        // Check entry A(j,j-1)
        if (_a(j,j-1)!=0) {
            // Non-zero entry
            l[j]=1;
        } else {
            // Zero entry
            l[j]=0;
        }
    }
}

int starneig_eigvec_gen_practical_row_tiling(int m, int mb, int *l, int *ap)
{
    // Maximum number of tiles
    int M=divceil(m,mb);

    // Initialize ap
    for (int k=0; k<M; k++)
        ap[k]=k*mb;
    // Last entry points at the first row/column beyond the matrix
    ap[M]=m;

    // Check for splitting of 2-by-2 block
    for (int k=1; k<M; k++) {
        // Index of first column of current tile
        int j=ap[k];
        // Look left at the entry A(j-1,j)
        if (l[j]==1) {
            // We are splitting a block. Push the current tile forward
            ap[k]++;
        }
    }
    /* At this point we have ap[k]=k*mb+r[k], where r[k]=0 or r[k]=1
          Tiles can collapse only if mb=1.
          The last tile may have been pushed out of the matrix.
    */

    // Remove any tiles which have collapsed
    int numTiles=starneig_eigvec_gen_remove_duplicates(M+1,ap);

    return numTiles;
}

int starneig_eigvec_gen_induced_column_tiling(
    int m, int *select, int *l, int M, int *ap, int *bp)
{
    // Used to accumulate the width of the current tile column.
    int aux=0;

    // First tile column starts at the first column
    bp[0]=0;

    // Loop over the tiles of matrices S, T
    for (int k=0; k<M; k++) {
        // Loop over the columns of the kth tile
        int j=ap[k];
        while (j<ap[k+1]-1) {
            // There remains room for a 2-by-2 block
            if (l[j+1]==1) {
		// Current block is 2-by-2
		if ((select[j]==1) || (select[j+1]==1)) {
	  	  // Current block is selected
	  	  aux=aux+2;
		}
		// Advance to next block
		j=j+2;
            } else {
		// Current block is 1-by-1
		if (select[j]==1) {
	  	  // Current block is selected
	  	  aux=aux+1;
		}
		// Advance to next block
		j++;
            }
        }
        // At this point j=ap[k+1]-1 or j=ap[k+1]
        if (j==ap[k+1]-1) {
            // Current block is 1-by-1
            if (select[j]==1) {
		aux=aux+1;
            }
            j++;
        }
        // At this point aux equals the size of the current tile
        // Compute the location of the next tile
        bp[k+1]=bp[k]+aux;
        // Reset aux
        aux=0;
    }
    // Return the number of selected eigenvalues
    return bp[M];
}

int starneig_eigvec_gen_practical_column_tiling(
    int n, int nb, int *map, int *l, int *cp)
{
    // Compute maximum number of columns needed
    int N=divceil(n,nb);

    // Preliminary values
    for (int k=0; k<N; k++)
        cp[k]=k*nb;
    // Last entry points to just beyond the end of the matrix of eigenvectors
    cp[N]=n;

    // Check to see if any complex eigenvectors are split
    for (int k=1; k<N; k++) {
        // Isolate global column index;
        int j=map[cp[k]];
        // Look left at entry A(j-1,j)
        if (l[j]==1) {
            // Complex eigenvector is being split, push current column to the right
            cp[k]++;
        }
    }

    /* At this point we have cp[k]=k*nb+r[k], where r[k]=0 or r[k]=1
          No column can collapse *unless* nb=1.
          The last column can be pushed out of the matrix
    */

    // Remove any tile columns which have collapsed
    N=starneig_eigvec_gen_remove_duplicates(N+1,cp);

    // Return the number of tile columns necessary
    return N;
}

int starneig_eigvec_gen_count_selected(int m, int *l, int *select)
{

    // Index into selection array and left-looking array.
    int j=0;
    // Number of selected eigenvalues
    int k=0;

    // Loop over the columns of the matrix
    while (j<m-1) {
        // There remains room for a 2-by-2 block
        if (l[j+1]==1) {
            // Current block is 2-by-2
            if ((select[j]==1) || (select[j+1]==1)) {
		// Current block is selected
		k=k+2;
            }
            // Advance to next block
            j=j+2;
        } else {
            // Current block is 1-by-1
            if (select[j]==1) {
		// Current block is selected
		k=k+1;
            }
            j=j+1;
        }
    }
    // At this point j = m or j = m-1.
    if (j==m-1) {
        // Last block is necessarily 1-by-1
        if (select[j]==1) {
            // Current block is selected
            k=k+1;
        }
    }
    // Return the number of selected eigenvalues
    return k;
}


int starneig_eigvec_gen_find_selected(int m, int *l, int *select, int *map)
{
    // Index into selection array and left-looking array.
    int j=0;
    // Index into map
    int k=0;

    // Loop over the columns of the matrix
    while (j<m-1) {
        // There remains room for a 2-by-2 block
        if (l[j+1]==1) {
            // Current block is 2-by-2
            if ((select[j]==1) || (select[j+1]==1)) {
		// Current block is selected
		map[k]=j; map[k+1]=j+1; k=k+2;
            }
            // Advance to next block
            j=j+2;
        } else {
            // Current block is 1-by-1
            if (select[j]==1) {
		// Current block is selected
		map[k]=j; k=k+1;
            }
            j=j+1;
        }
    }
    // At this point j = m or j = m-1.
    if (j==m-1) {
        // Last block is necessarily 1-by-1
        if (select[j]==1) {
            // Current block is selected
            map[k]=j; k=k+1;
        }
    }
    // Return the number of selected eigenvalues
    return k;
}

#undef a_
