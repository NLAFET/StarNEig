///
/// @file
///
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
#include "distr_matrix_internal.h"
#include "../common/math.h"
#include <starpu_mpi.h>

///
/// @brief Finds a valid tile size that is closest to a given preferred tile
/// size and is a multiple of 8.
///
/// @param[in] preferred_size - preferred tile size
/// @param[in] A - distributed matrix A
/// @param[in] B - distributed matrix B
/// @param[in] Q - distributed matrix Q
/// @param[in] Z - distributed matrix Z
///
/// @return valid tile size if one is found, -1 otherwise
///
static int find_valid_tile_size8(int preferred_size,
    struct starneig_distr_matrix const *A,
    struct starneig_distr_matrix const *B,
    struct starneig_distr_matrix const *Q,
    struct starneig_distr_matrix const *Z)
{
    //
    // find the largest valid tile size that is a multiple of 8
    //

    int mul; // the tile size is going to be 8 * mul

    // check matrix A
    if (A->row_blksz % 8 != 0 || A->col_blksz % 8 != 0)
        return -1;
    mul = starneig_largers_factor(A->row_blksz/8, A->col_blksz/8);

    // check matrix B
    if (B != NULL) {
        if (B->row_blksz % 8 != 0 || B->col_blksz % 8 != 0)
            return -1;
        mul = starneig_largers_factor(
            mul, starneig_largers_factor(B->row_blksz/8, B->col_blksz/8));
    }

    // check matrix Q
    if (Q != NULL) {
        if (Q->row_blksz % 8 != 0 || Q->col_blksz % 8 != 0)
            return -1;
        mul = starneig_largers_factor(
            mul, starneig_largers_factor(Q->row_blksz/8, Q->col_blksz/8));
    }

    // check matrix Z
    if (Z != NULL) {
        if (Z->row_blksz % 8 != 0 || Z->col_blksz % 8 != 0)
            return -1;
        mul = starneig_largers_factor(
            mul, starneig_largers_factor(Z->row_blksz/8, Z->col_blksz/8));
    }

    //
    // find a valid tile size that is close enough to the preferred tile size
    //

    if (0 < preferred_size && 0.1*preferred_size < abs(8*mul-preferred_size)) {
        int closest = mul;
        int closest_dist = abs(8*mul - preferred_size);

        // check numbers mul/2, mul/3, mul/4, ...
        for (int i = 2; i <= mul; i++) {

            // skip if mul/i is not an integer
            if (mul % i != 0)
                continue;

            int val = mul/i;
            int dist = abs(8*val - preferred_size);

            // check if we closer that before
            if (dist < closest_dist) {
                closest = val;
                closest_dist = dist;
            }
        }

        if (0.1*preferred_size < abs(8*closest-preferred_size))
            return 8*closest;
        return -1;
    }

    return 8 * mul;
}

int starneig_mpi_find_valid_tile_size(int preferred_size,
    struct starneig_distr_matrix const *A,
    struct starneig_distr_matrix const *B,
    struct starneig_distr_matrix const *Q,
    struct starneig_distr_matrix const *Z)
{
    int tile_size;

    // try to find a tile size that is a multiple of 8
    tile_size = find_valid_tile_size8(preferred_size, A, B, Q, Z);
    if (0 < tile_size)
        return tile_size;

    // find largest valid tile size
    tile_size = starneig_largers_factor(A->row_blksz, A->col_blksz);
    if (B != NULL)
        tile_size = starneig_largers_factor(tile_size,
            starneig_largers_factor(B->row_blksz, B->col_blksz));
    if (Q != NULL)
        tile_size = starneig_largers_factor(tile_size,
            starneig_largers_factor(Q->row_blksz, Q->col_blksz));
    if (Z != NULL)
        tile_size = starneig_largers_factor(tile_size,
            starneig_largers_factor(Z->row_blksz, Z->col_blksz));

    //
    // find the valid tile size that is closest to the preferred tile size
    //

    if (0 < preferred_size &&
    0.1*preferred_size < abs(tile_size-preferred_size)) {
        int closest = tile_size;
        int closest_dist = abs(tile_size - preferred_size);

        // check numbers tile_size/2, tile_size/3, tile_size/4, ...
        for (int i = 2; i <= tile_size; i++) {

            // skip if tile_size/i is not an integer
            if (tile_size % i != 0)
                continue;

            int val = tile_size/i;
            int dist = abs(val - preferred_size);

            // check if we closer that before
            if (dist < closest_dist) {
                closest = val;
                closest_dist = dist;
            }
        }
        return closest;
    }

    return tile_size;
}
