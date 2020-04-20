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
#include "utils.h"
#include "common.h"
#include "tiles.h"
#include "matrix.h"
#include "vector.h"
#include "tasks.h"
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif
#include <stddef.h>

int starneig_is_valid_matrix(
    int n, int tile_size, const starneig_matrix_descr_t descr)
{
    return descr == NULL || (
        STARNEIG_MATRIX_N(descr) == n &&
        STARNEIG_MATRIX_M(descr) == n &&
        STARNEIG_MATRIX_BN(descr) == tile_size &&
        STARNEIG_MATRIX_BM(descr) == tile_size);
}


int starneig_calc_update_size(
    int n, int bn, int sbn, int world_size, int worker_count)
{
    if (world_size == 1)
        return MAX(bn, MIN(divceil(n/8,bn)*bn, divceil(n/worker_count,bn)*bn));

    int size = MAX(bn, (sbn/2)*bn);

    int div = 2;
    while (bn < size && n/size < world_size * worker_count) {
        if (sbn % div == 0)
            size = (sbn/div)*bn;
        div++;
    }

    return MAX(bn, size);
}

#ifdef STARNEIG_ENABLE_MPI

struct matrix_to_vector_distr_arg {
    const starneig_matrix_descr_t descr;
    int tile_size;
};

static int matrix_to_vector_distr(int i, void const *ptr)
{
    struct matrix_to_vector_distr_arg const *arg = ptr;
    return starneig_get_elem_owner_matrix_descr(
        i*arg->tile_size, i*arg->tile_size, arg->descr);
}

#endif

starneig_vector_descr_t starneig_init_matching_vector_descr(
    const starneig_matrix_descr_t descr, size_t elemsize, void *vec,
    mpi_info_t mpi)
{
    int tile_size = starneig_largers_factor(
        STARNEIG_MATRIX_BM(descr), STARNEIG_MATRIX_BN(descr));
    if (
    tile_size < MIN(STARNEIG_MATRIX_BM(descr)/2, STARNEIG_MATRIX_BN(descr)/2))
        tile_size = STARNEIG_MATRIX_BM(descr);

    int (*distrib)(int, void const *) = NULL;
    void *distarg = NULL;

#ifdef STARNEIG_ENABLE_MPI
    struct matrix_to_vector_distr_arg arg = {
        .descr = descr,
        .tile_size = tile_size
    };
    if (mpi != NULL) {
        distrib = matrix_to_vector_distr;
        distarg = &arg;
    }
#endif

    return starneig_register_vector_descr(
        STARNEIG_MATRIX_M(descr), tile_size, elemsize, distrib, distarg,
        vec, mpi);
}

static void extract_subdiagonals_func(
    int size, int rbegin, int cbegin, int m, int n, int ldA, int ldB,
    void const *arg, void const *_A, void const *_B, void **masks)
{
    double const *A = _A;
    int *mask = masks[0];

    if (cbegin == 0)
        mask[0] = 0;

    for (int i = 0 < cbegin ? 0 : 1; i < size; i++)
        mask[i] = A[(cbegin+i-1)*ldA+rbegin+i] != 0.0;
}

starneig_vector_descr_t starneig_extract_subdiagonals(
    starneig_matrix_descr_t descr, mpi_info_t mpi)
{
    starneig_vector_descr_t ret = starneig_init_matching_vector_descr(
        descr, sizeof(int), NULL, mpi);

    starneig_insert_scan_diagonal(
        0, STARNEIG_MATRIX_M(descr), 0, 0, 0, 1, 0, STARPU_MAX_PRIO,
        extract_subdiagonals_func, NULL, descr, NULL, mpi, ret, NULL);

    return ret;
}

void * starneig_acquire_vector_descr(starneig_vector_descr_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (STARNEIG_VECTOR_DISTRIBUTED(descr)) {
        int world_size = starneig_mpi_get_comm_size();
        for (int i = 0; i < world_size; i++)
            starneig_gather_vector_descr(i, descr);
    }
#endif

    void *ret =
        malloc(STARNEIG_VECTOR_M(descr)*STARNEIG_VECTOR_ELEMSIZE(descr));

    int tiles = divceil(STARNEIG_VECTOR_M(descr), STARNEIG_VECTOR_BM(descr));
    for (int i = 0; i < tiles; i++) {
        starpu_data_handle_t handle =
            starneig_get_tile_from_vector_descr(i, descr);
        starpu_data_acquire(handle, STARPU_R);
        memcpy(
            ret + (size_t)i * STARNEIG_VECTOR_BM(descr) *
                STARNEIG_VECTOR_ELEMSIZE(descr),
            starpu_data_get_local_ptr(handle),
            MIN(
                STARNEIG_VECTOR_BM(descr),
                STARNEIG_VECTOR_M(descr) - i * STARNEIG_VECTOR_BM(descr)) *
                    STARNEIG_VECTOR_ELEMSIZE(descr));

        starpu_data_release(handle);
    }

    return ret;
}
