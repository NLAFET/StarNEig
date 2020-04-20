///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
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
#include <starneig/error.h>
#include "../../common/common.h"
#include "../../common/node_internal.h"
#include "partition.h"


void starneig_eigvec_std_partition(int n, const int *lambda_type, int tile_size, int *p)
{
    int num_tiles = (n+tile_size-1)/tile_size;

    for (int i = 0; i < num_tiles; i++)
        p[i] = i*tile_size;

    // Fill pad so that #rows = p[k + 1] - p[k].
    p[num_tiles] = n;

    // Adjust partitioning to not split 2-by-2 blocks across tiles.
    int num_cmplx = 0;

    // Absolute column indices.
    int first_idx = 0;
    int last_idx = MIN(first_idx+tile_size, n);

    for (int k = 0; k < num_tiles; k++) {
        // Count complex eigenvalues in this tile.
        for (int i = first_idx; i < last_idx; i++)
            if (lambda_type[i] == 1) // CMPLX
                num_cmplx++;

        if ((num_cmplx%2) == 0) {
            // This tile respects pairs of complex eigenvalues. Proceed.
            first_idx = last_idx;
            last_idx = MIN(first_idx + tile_size, n);
        }
        else {
            // 2-by-2 block split across tiles. Adapt.
            if (k == num_tiles-1)
                continue;

            p[k+1]++;

            first_idx = last_idx+1;
            last_idx = MIN(first_idx+tile_size-1, n);
            num_cmplx = 0;
        }
    }
}


int starneig_eigvec_std_count_selected(int n, const int *selected)
{
    int count = 0;
    for (int i = 0; i < n; i++)
        if (selected[i])
            count++;

    return count;
}


void starneig_eigvec_std_partition_selected(
    int n, const int *pr, int *selected, int num_tiles, int *pc)
{
    int num_selected = starneig_eigvec_std_count_selected(n, selected);

    // Fill pad so that #cols = pr[i]-pr[i-1].
    pc[0] = 0;
    pc[num_tiles] = num_selected;

    for (int i = 1; i < num_tiles; i++) {
        // Shrink width of tile column to number of selected.
        int width = pr[i]-pr[i-1];
        int quantity = starneig_eigvec_std_count_selected(width, selected+pr[i-1]);
        pc[i] = pc[i-1]+quantity;
    }
}
