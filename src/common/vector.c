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
#include "vector.h"
#include "common.h"
#include "tasks.h"

#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

///
/// @brief vector descriptor structure.
///
struct starneig_vector_descr {
    int rbegin;                             ///< first row
    int rend;                               ///< last row + 1
    int bm;                                 ///< tile height (row count)
    int elemsize;                           ///< element size
    int tm_count;                           ///< number of tile rows
    starpu_data_handle_t *tiles;            ///< data tiles
#ifdef STARNEIG_ENABLE_MPI
    int tag_offset;                         ///< tag offset
    int *owners;                            ///< section owners (MPI ranks)
#endif
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int starneig_vector_single_owner_func(int i, void const *ptr)
{
    return *((int const *) ptr);
}

starneig_vector_t starneig_vector_init(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, mpi_info_t mpi)
{
    STARNEIG_ASSERT_MSG(0 < m && 0 < elemsize, "Invalid dimensions.");
    STARNEIG_ASSERT_MSG(0 < bm, "Invalid tile dimensions.");

    starneig_vector_t descr =
        malloc(sizeof(struct starneig_vector_descr));

    descr->rbegin = 0;
    descr->rend = m;
    descr->bm = bm;
    descr->elemsize = elemsize;
    descr->tm_count = divceil(m, bm);

    descr->tiles = malloc(descr->tm_count*sizeof(starpu_data_handle_t));
    for (int i = 0; i < descr->tm_count; i++)
        descr->tiles[i] = NULL;

#ifdef STARNEIG_ENABLE_MPI
    descr->tag_offset = -1;
    descr->owners = NULL;
    if (mpi != NULL) {
        STARNEIG_ASSERT_MSG(distrib != NULL, "Missing distribution function.");
        descr->tag_offset = mpi->tag_offset;
        mpi->tag_offset += descr->tm_count;

        descr->owners = malloc(descr->tm_count*sizeof(int*));
        for (int i = 0; i < descr->tm_count; i++)
            descr->owners[i] = distrib(i, distarg);
    }
#endif

    return descr;
}

starneig_vector_t starneig_vector_register(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, void *vec, mpi_info_t mpi)
{
    int my_rank = starneig_mpi_get_comm_rank();

    starneig_vector_t descr = starneig_vector_init(
        m, bm, elemsize, distrib, distarg, mpi);

    if (vec != NULL) {
        for (int i = 0; i < descr->tm_count; i++) {
            starpu_data_handle_t handle;
            starpu_vector_data_register(&handle, STARPU_MAIN_RAM,
                (uintptr_t)vec+i*descr->bm*elemsize,
                MIN(descr->bm, m - i*descr->bm), elemsize);
            if (starneig_vector_get_tile_owner(i, descr) != my_rank)
                starpu_data_invalidate(handle);
            starneig_vector_set_tile(i, handle, descr);
        }
    }

    return descr;
}

void starneig_vector_unregister(starneig_vector_t descr)
{
    if (descr == NULL)
        return;

    for (int i = 0; i < descr->tm_count; i++) {
        if (descr->tiles[i] != NULL)
            starpu_data_unregister(descr->tiles[i]);
        descr->tiles[i] = NULL;
    }
}

void starneig_vector_free(starneig_vector_t descr)
{
    if (descr == NULL)
        return;

    if (descr->tiles != NULL) {
        for (int i = 0; i < descr->tm_count; i++)
            if (descr->tiles[i] != NULL)
                starpu_data_unregister_submit(descr->tiles[i]);
        free(descr->tiles);
    }

#ifdef STARNEIG_ENABLE_MPI
    free(descr->owners);
#endif

    free(descr);
}

int starneig_vector_get_rbegin(const starneig_vector_t descr)
{
    return descr->rbegin;
}

int starneig_vector_get_rend(const starneig_vector_t descr)
{
    return descr->rend;
}

int starneig_vector_get_rows(const starneig_vector_t descr)
{
    return descr->rend - descr->rbegin;
}

int starneig_vector_get_tile_size(const starneig_vector_t descr)
{
    return descr->bm;
}

size_t starneig_vector_get_elemsize(
    const starneig_vector_t descr)
{
    return descr->elemsize;
}

int starneig_vector_is_distributed(
    const starneig_vector_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return 0 <= descr->tag_offset;
#else
    return 0;
#endif
}

int starneig_vector_get_tile_idx(int row, const starneig_vector_t descr)
{
    return (starneig_vector_get_rbegin(descr) + row) /
        starneig_vector_get_tile_size(descr);
}

int starneig_vector_get_in_tile_idx(
    int row, int tile, const starneig_vector_t descr)
{
    return
        starneig_vector_get_rbegin(descr) + row
            - tile * starneig_vector_get_tile_size(descr);
}

int starneig_vector_get_ext_idx(int tile, int row, starneig_vector_t descr)
{
    return
        tile * starneig_vector_get_tile_size(descr)
            + starneig_vector_get_rbegin(descr) + row;
}

void starneig_vector_set_tile(
    int i, starpu_data_handle_t handle, starneig_vector_t descr)
{
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= i && i < descr->tm_count);
    STARNEIG_ASSERT(descr->tiles[i] == NULL);

#ifdef STARNEIG_ENABLE_MPI
    // set MPI tag and tell StarPU who owns the actual handle
    if (0 <= descr->tag_offset)
        starpu_mpi_data_register_comm(
            handle, descr->tag_offset + i, descr->owners[i],
            starneig_mpi_get_comm());
#endif

    descr->tiles[i] = handle;
}

starpu_data_handle_t starneig_vector_get_tile(int i, starneig_vector_t descr)
{
    STARNEIG_ASSERT(descr != NULL);

    STARNEIG_ASSERT(0 <= i && i < descr->tm_count);

    // register a placeholder if the tile does not exist
    if (descr->tiles[i] == NULL) {
        starpu_vector_data_register(&descr->tiles[i], -1, (uintptr_t)NULL,
            MIN(descr->bm, descr->rend - i*descr->bm),
            descr->elemsize);

#ifdef STARNEIG_ENABLE_MPI
        if (0 <= descr->tag_offset) {
            int my_rank = starneig_mpi_get_comm_rank();
            int owner = starneig_vector_get_tile_owner(i, descr);
            starpu_mpi_data_register_comm(
                descr->tiles[i], descr->tag_offset + i, owner,
                starneig_mpi_get_comm());
            if (my_rank == owner)
                starneig_insert_set_vector_to_zero(
                    STARPU_MAX_PRIO, descr->tiles[i], NULL);
        }
        else {
            starneig_insert_set_vector_to_zero(
                STARPU_MAX_PRIO, descr->tiles[i], NULL);
        }
#else
    starneig_insert_set_vector_to_zero(
        STARPU_MAX_PRIO, descr->tiles[i], NULL);
#endif
    }

    return descr->tiles[i];
}

void starneig_vector_gather(int root, starneig_vector_t descr)
{
    starneig_vector_gather_section(
        root, 0, descr->rend - descr->rbegin, descr);
}

void starneig_vector_scatter(int root, starneig_vector_t descr)
{
    starneig_vector_scatter_section(
        root, 0, descr->rend - descr->rbegin, descr);
}

void starneig_vector_gather_section(
    int root, int begin, int end, starneig_vector_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (descr->tag_offset < 0)
        return;

    int my_rank = starneig_mpi_get_comm_rank();

    int tbegin = (descr->rbegin + begin) / descr->bm;
    int tend = (descr->rbegin + end - 1) / descr->bm + 1;

    for (int i = tbegin; i < tend; i++) {
        if (root == my_rank || descr->owners[i] == my_rank) {
            starpu_data_handle_t handle =
                starneig_vector_get_tile(i, descr);
            starpu_mpi_gather_detached(
                &handle, 1, root, starneig_mpi_get_comm(),
                NULL, NULL, NULL, NULL);
        }
    }
#endif
}

void starneig_vector_scatter_section(
    int root, int begin, int end, starneig_vector_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (descr->tag_offset < 0)
        return;

    int my_rank = starneig_mpi_get_comm_rank();

    int tbegin = (descr->rbegin + begin) / descr->bm;
    int tend = (descr->rbegin + end - 1) / descr->bm + 1;

    for (int i = tbegin; i < tend; i++) {
        if (root == my_rank || descr->owners[i] == my_rank) {
            starpu_data_handle_t handle =
                starneig_vector_get_tile(i, descr);
            starpu_mpi_scatter_detached(
                &handle, 1, root, starneig_mpi_get_comm(),
                NULL, NULL, NULL, NULL);
        }
    }
#endif
}

int starneig_vector_get_tile_owner(int i, starneig_vector_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (0 <= descr->tag_offset)
        return descr->owners[i];
#endif
    return starneig_mpi_get_comm_rank();
}

int starneig_vector_get_elem_owner(int i, starneig_vector_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return starneig_vector_get_tile_owner(
        (descr->rbegin + i)/descr->bm, descr);
#endif
    return starneig_mpi_get_comm_rank();
}

int starneig_vector_involved_with_section(
    int begin, int end, starneig_vector_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (descr->tag_offset < 0)
        return 1;

    int my_rank = starneig_mpi_get_comm_rank();

    int bbegin = (descr->rbegin + begin) / descr->bm;
    int bend = (descr->rbegin + end-1) / descr->bm + 1;

    for (int i = bbegin; i < bend; i++)
        if (starneig_vector_get_tile_owner(i, descr) == my_rank)
            return 1;

    return 0;
#else
    return 1;
#endif
}

int starneig_vector_cut_up(int row, const starneig_vector_t descr)
{
    int rbegin = starneig_vector_get_rbegin(descr);
    int bm = starneig_vector_get_tile_size(descr);

    return MAX(0, ((rbegin + row) / bm) * bm - rbegin);
}

int starneig_vector_cut_down(int row, const starneig_vector_t descr)
{
    int rbegin = starneig_vector_get_rbegin(descr);
    int bm = starneig_vector_get_tile_size(descr);

    return MAX(0, divceil(rbegin + row, bm) * bm - rbegin);
}
