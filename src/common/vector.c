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

#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

static void set_to_zero(void *buffers[], void *cl_args)
{
    void *ptr = (void *) STARPU_VECTOR_GET_PTR(buffers[0]);
    int m = STARPU_VECTOR_GET_NX(buffers[0]);
    size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(buffers[0]);

    memset(ptr, 0, m*elemsize);
}

static void insert_set_to_zero(starpu_data_handle_t tile)
{
    static struct starpu_codelet set_to_zero_cl = {
        .name = "starneig_set_to_zero",
        .cpu_funcs = { set_to_zero },
        .cpu_funcs_name = { "set_to_zero" },
        .nbuffers = 1,
        .modes = { STARPU_W }
    };

    starpu_task_insert(&set_to_zero_cl,
        STARPU_PRIORITY, STARPU_MAX_PRIO, STARPU_W, tile, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

starneig_vector_descr_t starneig_init_vector_descr(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, mpi_info_t mpi)
{
    STARNEIG_ASSERT_MSG(0 < m && 0 < elemsize, "Invalid dimensions.");
    STARNEIG_ASSERT_MSG(0 < bm, "Invalid tile dimensions.");

    starneig_vector_descr_t descr =
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

    descr->parent = NULL;
    descr->mode = STARNEIG_VECTOR_ROOT;

    return descr;
}

starneig_vector_descr_t starneig_register_vector_descr(
    int m, int bm, size_t elemsize, int (*distrib)(int, void const *),
    void *distarg, void *vec, mpi_info_t mpi)
{
    int my_rank = starneig_mpi_get_comm_rank();

    starneig_vector_descr_t descr = starneig_init_vector_descr(
        m, bm, elemsize, distrib, distarg, mpi);

    if (vec != NULL) {
        for (int i = 0; i < descr->tm_count; i++) {
            starpu_data_handle_t handle;
            starpu_vector_data_register(&handle, STARPU_MAIN_RAM,
                (uintptr_t)vec+i*descr->bm*elemsize,
                MIN(descr->bm, m - i*descr->bm), elemsize);
            if (starneig_get_tile_owner_vector_descr(i, descr) != my_rank)
                starpu_data_invalidate(handle);
            starneig_register_tile_with_vector_descr(i, handle, descr);
        }
    }

    return descr;
}

starneig_vector_descr_t starneig_create_sub_vector_descr(
    int begin, int end, starneig_vector_descr_t descr)
{
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= begin && end <= STARNEIG_VECTOR_M(descr));

    starneig_vector_descr_t sub_descr =
        malloc(sizeof(struct starneig_vector_descr));
    memcpy(sub_descr, descr, sizeof(struct starneig_vector_descr));

    sub_descr->rbegin = descr->rbegin + begin;
    sub_descr->rend = descr->rbegin + end;

#ifdef STARNEIG_ENABLE_MPI
    sub_descr->owners = NULL;
#endif

    sub_descr->tiles = NULL;
    sub_descr->parent = descr;
    sub_descr->mode = STARNEIG_VECTOR_SUB_VECTOR;

    return sub_descr;

}

void starneig_unregister_vector_descr(starneig_vector_descr_t descr)
{
    if (descr == NULL || descr->mode == STARNEIG_VECTOR_SUB_VECTOR)
        return;

    for (int i = 0; i < descr->tm_count; i++) {
        if (descr->tiles[i] != NULL)
            starpu_data_unregister(descr->tiles[i]);
        descr->tiles[i] = NULL;
    }
}

void starneig_free_vector_descr(starneig_vector_descr_t descr)
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

void starneig_register_tile_with_vector_descr(
    int i, starpu_data_handle_t handle, starneig_vector_descr_t descr)
{
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(descr->parent == NULL);
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

starpu_data_handle_t starneig_get_tile_from_vector_descr(
    int i, starneig_vector_descr_t descr)
{
    STARNEIG_ASSERT(descr != NULL);

    if (descr->mode == STARNEIG_VECTOR_SUB_VECTOR)
        return starneig_get_tile_from_vector_descr(i, descr->parent);

    STARNEIG_ASSERT(0 <= i && i < descr->tm_count);

    // register a placeholder if the tile does not exist
    if (descr->tiles[i] == NULL) {
        starpu_vector_data_register(&descr->tiles[i], -1, (uintptr_t)NULL,
            MIN(descr->bm, descr->rend - i*descr->bm),
            descr->elemsize);

#ifdef STARNEIG_ENABLE_MPI
        if (0 <= descr->tag_offset) {
            int my_rank = starneig_mpi_get_comm_rank();
            int owner = starneig_get_tile_owner_vector_descr(i, descr);
            starpu_mpi_data_register_comm(
                descr->tiles[i], descr->tag_offset + i, owner,
                starneig_mpi_get_comm());
            if (my_rank == owner)
                insert_set_to_zero(descr->tiles[i]);
        }
        else {
            insert_set_to_zero(descr->tiles[i]);
        }
#else
        insert_set_to_zero(descr->tiles[i]);
#endif
    }

    return descr->tiles[i];
}

void starneig_gather_vector_descr(int root, starneig_vector_descr_t descr)
{
    starneig_gather_segment_vector_descr(
        root, 0, descr->rend - descr->rbegin, descr);
}

void starneig_scatter_vector_descr(int root, starneig_vector_descr_t descr)
{
    starneig_scatter_segment_vector_descr(
        root, 0, descr->rend - descr->rbegin, descr);
}

void starneig_gather_segment_vector_descr(
    int root, int begin, int end, starneig_vector_descr_t descr)
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
                starneig_get_tile_from_vector_descr(i, descr);
            starpu_mpi_gather_detached(
                &handle, 1, root, starneig_mpi_get_comm(),
                NULL, NULL, NULL, NULL);
        }
    }
#endif
}

void starneig_scatter_segment_vector_descr(
    int root, int begin, int end, starneig_vector_descr_t descr)
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
                starneig_get_tile_from_vector_descr(i, descr);
            starpu_mpi_scatter_detached(
                &handle, 1, root, starneig_mpi_get_comm(),
                NULL, NULL, NULL, NULL);
        }
    }
#endif
}

int starneig_get_tile_owner_vector_descr(int i, starneig_vector_descr_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (descr->parent != NULL)
        return starneig_get_tile_owner_vector_descr(i, descr->parent);

    if (0 <= descr->tag_offset)
        return descr->owners[i];
#endif
    return starneig_mpi_get_comm_rank();
}

int starneig_get_elem_owner_vector_descr(int i, starneig_vector_descr_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return starneig_get_tile_owner_vector_descr(
        (descr->rbegin + i)/descr->bm, descr);
#endif
    return starneig_mpi_get_comm_rank();
}

int starneig_involved_with_part_of_vector_descr(
    int begin, int end, starneig_vector_descr_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (descr->tag_offset < 0)
        return 1;

    if (descr->parent != NULL)
        return starneig_involved_with_part_of_vector_descr(
            descr->rbegin+begin, descr->rbegin+end, descr);

    int my_rank = starneig_mpi_get_comm_rank();

    int bbegin = (descr->rbegin + begin) / descr->bm;
    int bend = (descr->rbegin + end-1) / descr->bm + 1;

    for (int i = bbegin; i < bend; i++)
        if (starneig_get_tile_owner_vector_descr(i, descr) == my_rank)
            return 1;

    return 0;
#else
    return 1;
#endif
}
