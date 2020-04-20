///
/// @file
///
/// @brief This file contains the definition of a matrix descriptor that is used
/// throughout the all components of the library.
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
#include "matrix.h"
#include "common.h"
#include "tasks.h"

#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif

struct starneig_matrix_descr {
    int rbegin;                           ///< first row
    int rend;                             ///< last row + 1
    int cbegin;                           ///< first column
    int cend;                             ///< last column + 1
    int bm;                               ///< tile height (row count)
    int bn;                               ///< tile width (column count)
    int sbm;                              ///< section height (tile row count)
    int sbn;                              ///< section width (tile column count)
    int elemsize;                         ///< element size
    int tm_count;                         ///< number of tile rows
    int tn_count;                         ///< number of tile columns
#ifdef STARNEIG_ENABLE_MPI
    int tag_offset;                       ///< tag offset
    int **owners;                         ///< section owners (MPI ranks)
#endif
    starpu_data_handle_t **tiles;         ///< tiles
#ifdef STARNEIG_ENABLE_EVENTS
    char event_label;
    int event_enabled;
    int event_roffset;
    int event_coffset;
#endif
};

static void copy_elem(void *buffers[], void *cl_args)
{
    int i, j;
    starpu_codelet_unpack_args(cl_args, &i, &j);

    void *source = (void *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ld = STARPU_MATRIX_GET_LD(buffers[0]);
    size_t elemsize = STARPU_MATRIX_GET_ELEMSIZE(buffers[0]);
    void *dest = (void *) STARPU_VARIABLE_GET_PTR(buffers[1]);

    memcpy(dest, source+((size_t)j*ld+i)*elemsize, elemsize);
}

static void insert_copy_elem(
    int i, int j, starpu_data_handle_t source, starpu_data_handle_t dest)
{
    static struct starpu_codelet copy_elem_cl = {
        .name = "starneig_copy_elem",
        .cpu_funcs = { copy_elem },
        .cpu_funcs_name = { "copy_elem" },
        .nbuffers = 2,
        .modes = { STARPU_R, STARPU_W }
    };

    starpu_task_insert(
        &copy_elem_cl,
        STARPU_PRIORITY, STARPU_MAX_PRIO,
        STARPU_VALUE, &i, sizeof(i),
        STARPU_VALUE, &j, sizeof(j),
        STARPU_R, source,
        STARPU_W, dest, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

starneig_matrix_t starneig_matrix_init(
    int m, int n, int bm, int bn, int sbm, int sbn, size_t elemsize,
    int (*distrib)(int, int, const void*), void const *distarg, mpi_info_t mpi)
{
    STARNEIG_ASSERT_MSG(0 < m && 0 < n && 0 < elemsize, "Invalid dimensions.");
    STARNEIG_ASSERT_MSG(0 < bm && 0 < bn, "Invalid tile dimensions.");

    if (distrib == NULL || sbm < 1)
        sbm = divceil(m, bm);
    if (distrib == NULL || sbn < 1)
        sbn = divceil(n, bn);

    starneig_matrix_t descr =
        malloc(sizeof(struct starneig_matrix_descr));

    descr->rbegin = 0;
    descr->rend = m;
    descr->cbegin = 0;
    descr->cend = n;
    descr->bm = bm;
    descr->bn = bn;
    descr->sbm = sbm;
    descr->sbn = sbn;
    descr->elemsize = elemsize;
    descr->tm_count = divceil(m, bm);
    descr->tn_count = divceil(n, bn);

    descr->tiles = malloc(descr->tm_count*sizeof(starpu_data_handle_t*));
    for (int i = 0; i < descr->tm_count; i++) {
        descr->tiles[i] = malloc(descr->tn_count*sizeof(starpu_data_handle_t));
        for (int j = 0; j < descr->tn_count; j++)
           descr->tiles[i][j] = NULL;
    }

#ifdef STARNEIG_ENABLE_MPI
    descr->tag_offset = -1;
    descr->owners = NULL;
    if (mpi != NULL) {
        STARNEIG_ASSERT_MSG(distrib != NULL, "Missing distribution function.");
        descr->tag_offset = mpi->tag_offset;
        mpi->tag_offset += descr->tm_count*descr->tn_count;

        int sm_count = divceil(descr->tm_count, descr->sbm);
        int sn_count = divceil(descr->tn_count, descr->sbn);

        // compute section owners
        descr->owners = malloc(sm_count*sizeof(int*));
        for (int i = 0; i < sm_count; i++) {
            descr->owners[i] = malloc(sn_count*sizeof(int));
            for (int j = 0; j < sn_count; j++)
                descr->owners[i][j] = distrib(i, j, distarg);
        }
    }
#endif

#ifdef STARNEIG_ENABLE_EVENTS
    descr->event_enabled = 0;
    descr->event_label = 'X';
    descr->event_roffset = 0;
    descr->event_coffset = 0;
#endif

    return descr;
}

starneig_matrix_t starneig_matrix_register(
    enum starneig_matrix_type type, int m, int n, int bm, int bn, int sbm,
    int sbn, int ld, size_t elemsize, int (*distrib)(int, int, void const *),
    void const *distarg, void *mat, mpi_info_t mpi)
{
    STARNEIG_ASSERT_MSG(mat == NULL || m <= ld, "Invalid leading dimension.");

    starneig_matrix_t descr = starneig_matrix_init(
        m, n, bm, bn, sbm, sbm, elemsize, distrib, distarg, mpi);

    int my_rank = starneig_mpi_get_comm_rank();

    for (int i = 0; i < descr->tm_count; i++) {
        for (int j = 0; j < descr->tn_count; j++) {

            // if the matrix is upper Hessenberg, then skip over those tiles
            // that are below the sub-diagonal
            if (type == MATRIX_TYPE_UPPER_HESSENBERG && (j+1)*bn < i*bm)
                continue;

            // if the matrix is upper triangular, then skip over those tiles
            // that are below the diagonal
            if (type == MATRIX_TYPE_UPPER_TRIANGULAR &&
            (j+1)*bn-1 < i*bm)
                continue;

            starpu_data_handle_t handle;
            starpu_matrix_data_register(&handle, STARPU_MAIN_RAM,
                (uintptr_t)(mat+((size_t)j*bn*ld+i*bm)*elemsize), ld,
                MIN(bm, m-i*bm), MIN(bn, n-j*bn), elemsize);

            if (starneig_matrix_get_tile_owner(i, j, descr) != my_rank)
                starpu_data_invalidate(handle);
            starneig_matrix_set_tile(i, j, handle, descr);
        }
    }

    return descr;
}

void starneig_matrix_acquire(starneig_matrix_t descr)
{
    if (descr == NULL)
        return;

    int my_rank = starneig_mpi_get_comm_rank();

    for (int i = 0; i < descr->tm_count; i++) {
        for (int j = 0; j < descr->tn_count; j++) {
            if (descr->tiles[i][j] != NULL) {
                int owner = starneig_matrix_get_tile_owner(i, j, descr);
                if (owner == my_rank)
                    starpu_data_acquire(descr->tiles[i][j], STARPU_RW);
                else
                    starpu_data_invalidate(descr->tiles[i][j]);
            }
        }
    }
}

void starneig_matrix_release(starneig_matrix_t descr)
{
    if (descr == NULL)
        return;

    int my_rank = starneig_mpi_get_comm_rank();

    for (int i = 0; i < descr->tm_count; i++) {
        for (int j = 0; j < descr->tn_count; j++) {
            if (descr->tiles[i][j] != NULL) {
                int owner = starneig_matrix_get_tile_owner(i, j, descr);
                if (owner == my_rank)
                    starpu_data_release(descr->tiles[i][j]);
            }
        }
    }
}

void starneig_matrix_unregister(starneig_matrix_t descr)
{
    if (descr == NULL)
        return;

    int my_rank = starneig_mpi_get_comm_rank();

    for (int i = 0; i < descr->tm_count; i++) {
        for (int j = 0; j < descr->tn_count; j++) {
            if (descr->tiles[i][j] != NULL) {
                int owner = starneig_matrix_get_tile_owner(i, j, descr);
                if (owner == my_rank)
                    starpu_data_unregister(descr->tiles[i][j]);
                else
                    starpu_data_unregister_submit(descr->tiles[i][j]);
                descr->tiles[i][j] = NULL;
            }
        }
    }
}

void starneig_matrix_free(starneig_matrix_t descr)
{
    if (descr == NULL)
        return;

#ifdef STARNEIG_ENABLE_MPI
    if (descr->owners != NULL) {
        int sm_count = divceil(descr->tm_count, descr->sbm);
        for (int i = 0; i < sm_count; i++)
            free(descr->owners[i]);
        free(descr->owners);
    }
#endif

    if (descr->tiles != NULL) {
        for (int i = 0; i < descr->tm_count; i++) {
            for (int j = 0; j < descr->tn_count; j++)
                if (descr->tiles[i][j] != NULL)
                    starpu_data_unregister_submit(descr->tiles[i][j]);
            free(descr->tiles[i]);
        }
        free(descr->tiles);
    }

    free(descr);
}

void starneig_matrix_set_tile(int i, int j,
    starpu_data_handle_t handle, starneig_matrix_t descr)
{
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= i && i < descr->tm_count);
    STARNEIG_ASSERT(0 <= j && j < descr->tn_count);
    STARNEIG_ASSERT(descr->tiles[i][j] == NULL);

#ifdef STARNEIG_ENABLE_MPI
    // set MPI tag and tell StarPU who owns the actual handle
    if (0 <= descr->tag_offset)
        starpu_mpi_data_register_comm(handle,
            descr->tag_offset + j*descr->tm_count + i,
            starneig_matrix_get_tile_owner(i, j, descr),
            starneig_mpi_get_comm());
#endif

    descr->tiles[i][j] = handle;
}

starpu_data_handle_t starneig_matrix_get_tile(
    int i, int j, starneig_matrix_t descr)
{
    STARNEIG_ASSERT(descr != NULL);

    STARNEIG_ASSERT(0 <= i && i < descr->tm_count);
    STARNEIG_ASSERT(0 <= j && j < descr->tn_count);

    // register a placeholder if the tile does not exist
    if (descr->tiles[i][j] == NULL) {
        starpu_matrix_data_register(&descr->tiles[i][j], -1,
            (uintptr_t)NULL, descr->bm, MIN(descr->bm, descr->rend-i*descr->bm),
            MIN(descr->bn, descr->cend-j*descr->bn), descr->elemsize);

#ifdef STARNEIG_ENABLE_MPI
        if (0 <= descr->tag_offset) {
            int my_rank = starneig_mpi_get_comm_rank();
            int owner = starneig_matrix_get_tile_owner(i, j, descr);
            starpu_mpi_data_register_comm(descr->tiles[i][j],
                descr->tag_offset + j*descr->tm_count + i, owner,
                starneig_mpi_get_comm());
            if (my_rank == owner)
                starneig_insert_set_matrix_to_zero(
                    STARPU_MAX_PRIO, descr->tiles[i][j], NULL);
        }
        else {
            starneig_insert_set_matrix_to_zero(
                STARPU_MAX_PRIO, descr->tiles[i][j], NULL);
        }
#else
        starneig_insert_set_matrix_to_zero(
            STARPU_MAX_PRIO, descr->tiles[i][j], NULL);
#endif
    }

    return descr->tiles[i][j];
}

starpu_data_handle_t starneig_matrix_get_elem(
    int i, int j, starneig_matrix_t descr, mpi_info_t mpi)
{
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= i && i < STARNEIG_MATRIX_M(descr));
    STARNEIG_ASSERT(0 <= j && j < STARNEIG_MATRIX_N(descr));

    starpu_data_handle_t handle;
    starpu_variable_data_register(&handle, -1, (uintptr_t) 0, descr->elemsize);

    int my_rank = starneig_mpi_get_comm_rank();
    int owner = starneig_matrix_get_elem_owner(i, j, descr);

#ifdef STARNEIG_ENABLE_MPI
    if (mpi != NULL)
        starpu_mpi_data_register_comm(
            handle, mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif

    if (owner == my_rank) {
        starpu_data_handle_t tile = starneig_matrix_get_tile(
            (descr->rbegin + i)/descr->bm, (descr->cbegin + j)/descr->bn,
            descr);

        insert_copy_elem(
            (descr->rbegin + i) % descr->bm, (descr->cbegin + j) % descr->bn,
            tile, handle);
    }

    return handle;
}

void starneig_matrix_register_section(
    enum starneig_matrix_type type, int i, int j, int ld, void *mat,
    starneig_matrix_t descr)
{
    if (mat == NULL)
        return;

    int elemsize = descr->elemsize;
    int sbn = descr->sbn;
    int sbm = descr->sbm;
    int bn = descr->bn;
    int bm = descr->bm;
    int m = descr->rend - descr->rbegin;
    int n = descr->cend - descr->cbegin;

    int mtiles = MIN(sbm, divceil(m, bm) - i*sbm);
    int ntiles = MIN(sbn, divceil(n, bn) - j*sbn);

    // if the matrix is upper Hessenberg, then we can ignore everything below
    // the sub-diagonal
    if (type == MATRIX_TYPE_UPPER_HESSENBERG && (j+1)*sbn*bn < i*sbm*bm)
        return;

    // if the matrix is upper triangular, then we can ignore everything below
    // the diagonal
    if (type == MATRIX_TYPE_UPPER_TRIANGULAR &&
    (j+1)*sbn*bn-1 < i*sbm*bm)
        return;

    for (int jj = 0; jj < ntiles; jj++) {
        for (int ii = 0; ii < mtiles; ii++) {

            // if the matrix is upper Hessenberg, then skip over those tiles
            // that are below the sub-diagonal
            if (type == MATRIX_TYPE_UPPER_HESSENBERG &&
            (j*sbn+jj+1)*bn < (i*sbm+ii)*bm)
                continue;

            // if the matrix is upper triangular, then skip over those tiles
            // that are below the diagonal
            if (type == MATRIX_TYPE_UPPER_TRIANGULAR &&
            (j*sbn+jj+1)*bn-1 < (i*sbm+ii)*bm)
                continue;

            starpu_data_handle_t handle;

            starpu_matrix_data_register(&handle, STARPU_MAIN_RAM,
                (uintptr_t)(mat+((size_t)jj*bn*ld+ii*bm)*elemsize), ld,
                MIN(bm, m-(i*sbm+ii)*bm),
                MIN(bn, n-(j*sbn+jj)*bn),
                elemsize);

            starneig_matrix_set_tile(i * sbm + ii, j * sbn + jj,
                handle, descr);
        }
    }
}

int starneig_matrix_get_tile_owner(
    int i, int j, const starneig_matrix_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (0 <= descr->tag_offset)
        return descr->owners[i/descr->sbm][j/descr->sbn];
#endif
    return starneig_mpi_get_comm_rank();
}

int starneig_matrix_get_elem_owner(
    int i, int j, const starneig_matrix_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return starneig_matrix_get_tile_owner(
        (descr->rbegin+i)/descr->bm, (descr->cbegin+j)/descr->bn, descr);
#else
    return starneig_mpi_get_comm_rank();
#endif
}

int starneig_matrix_involved_with_section(
    int rbegin, int rend, int cbegin, int cend,
    const starneig_matrix_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    if (descr->tag_offset < 0)
        return 1;

    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= rbegin && rend <= STARNEIG_MATRIX_M(descr));
    STARNEIG_ASSERT(0 <= cbegin && cend <= STARNEIG_MATRIX_N(descr));

    int my_rank = starneig_mpi_get_comm_rank();

    int srbegin = (descr->rbegin + rbegin) / (descr->sbm * descr->bm);
    int srend = (descr->rbegin + rend-1) / (descr->sbm * descr->bm) + 1;

    int scbegin = (descr->cbegin + cbegin) / (descr->sbn * descr->bn);
    int scend = (descr->cbegin + cend-1) / (descr->sbn * descr->bn) + 1;

    for (int i = srbegin; i < srend; i++)
        for (int j = scbegin; j < scend; j++)
            if (descr->owners[i][j] == my_rank)
                return 1;

    return 0;
#else
    return 1;
#endif
}

void starneig_matrix_flush_section(
    int rbegin, int rend, int cbegin, int cend,
    const starneig_matrix_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= rbegin && rend <= STARNEIG_MATRIX_M(descr));
    STARNEIG_ASSERT(0 <= cbegin && cend <= STARNEIG_MATRIX_N(descr));

    if (descr->tag_offset < 0)
        return;

    int srbegin = (descr->rbegin + rbegin) / descr->bm;
    int srend = (descr->rbegin + rend-1) / descr->bm + 1;

    int scbegin = (descr->cbegin + cbegin) / descr->bn;
    int scend = (descr->cbegin + cend-1) / descr->bn + 1;

    for (int i = srbegin; i < srend; i++)
        for (int j = scbegin; j < scend; j++)
            if (descr->tiles[i][j] != NULL)
                starpu_mpi_cache_flush(
                    starneig_mpi_get_comm(), descr->tiles[i][j]);
#endif
}

void starneig_matrix_prefetch_section(
    int rbegin, int rend, int cbegin, int cend, int node, int async,
    const starneig_matrix_t descr)
{
    STARNEIG_ASSERT(descr != NULL);
    STARNEIG_ASSERT(0 <= rbegin && rend <= STARNEIG_MATRIX_M(descr));
    STARNEIG_ASSERT(0 <= cbegin && cend <= STARNEIG_MATRIX_N(descr));

    int srbegin = (descr->rbegin + rbegin) / descr->bm;
    int srend = (descr->rbegin + rend-1) / descr->bm + 1;

    int scbegin = (descr->cbegin + cbegin) / descr->bn;
    int scend = (descr->cbegin + cend-1) / descr->bn + 1;

    for (int i = srbegin; i < srend; i++)
        for (int j = scbegin; j < scend; j++)
            if (descr->tiles[i][j] != NULL)
                starpu_data_prefetch_on_node(descr->tiles[i][j], node, async);
}

int STARNEIG_MATRIX_RBEGIN(const starneig_matrix_t descr)
{
    return descr->rbegin;
}

int STARNEIG_MATRIX_REND(const starneig_matrix_t descr)
{
    return descr->rend;
}

int STARNEIG_MATRIX_CBEGIN(const starneig_matrix_t descr)
{
    return descr->cbegin;
}

int STARNEIG_MATRIX_CEND(const starneig_matrix_t descr)
{
    return descr->cend;
}

int STARNEIG_MATRIX_M(const starneig_matrix_t descr)
{
    return descr->rend - descr->rbegin;
}

int STARNEIG_MATRIX_N(const starneig_matrix_t descr)
{
    return descr->cend - descr->cbegin;
}

int STARNEIG_MATRIX_BM(const starneig_matrix_t descr)
{
    return descr->bm;
}

int STARNEIG_MATRIX_BN(const starneig_matrix_t descr)
{
    return descr->bn;
}

int STARNEIG_MATRIX_SBM(const starneig_matrix_t descr)
{
    return descr->sbm;
}

int STARNEIG_MATRIX_SBN(const starneig_matrix_t descr)
{
    return descr->sbn;
}

int STARNEIG_MATRIX_SM(const starneig_matrix_t descr)
{
    return descr->sbm*descr->bm;
}

int STARNEIG_MATRIX_SN(const starneig_matrix_t descr)
{
    return descr->sbn*descr->bn;
}

size_t STARNEIG_MATRIX_ELEMSIZE(
    const starneig_matrix_t descr)
{
    return descr->elemsize;
}

int STARNEIG_MATRIX_DISTRIBUTED(
    const starneig_matrix_t descr)
{
#ifdef STARNEIG_ENABLE_MPI
    return 0 <= descr->tag_offset;
#else
    return 0;
#endif
}

int STARNEIG_MATRIX_TILE_IDX(
    int row, const starneig_matrix_t descr)
{
    return (STARNEIG_MATRIX_RBEGIN(descr) + row) / STARNEIG_MATRIX_BM(descr);
}

int STARNEIG_MATRIX_TILE_IDY(
    int column, const starneig_matrix_t descr)
{
    return (STARNEIG_MATRIX_CBEGIN(descr) + column) / STARNEIG_MATRIX_BN(descr);
}

int starneig_matrix_cut_ver_up(
    int row, const starneig_matrix_t descr)
{
    int rbegin = STARNEIG_MATRIX_RBEGIN(descr);
    int bm = STARNEIG_MATRIX_BM(descr);
    int m = STARNEIG_MATRIX_M(descr);

    return MAX(0, MIN(m, ((rbegin + row) / bm) * bm - rbegin));
}

int starneig_matrix_cut_ver_down(
    int row, const starneig_matrix_t descr)
{
    int rbegin = STARNEIG_MATRIX_RBEGIN(descr);
    int bm = STARNEIG_MATRIX_BM(descr);
    int m = STARNEIG_MATRIX_M(descr);

    return MAX(0, MIN(m, divceil(rbegin + row, bm) * bm - rbegin));
}

int starneig_matrix_cut_hor_left(
    int column, const starneig_matrix_t descr)
{
    int cbegin = STARNEIG_MATRIX_CBEGIN(descr);
    int bn = STARNEIG_MATRIX_BN(descr);
    int n = STARNEIG_MATRIX_N(descr);

    return MAX(0, MIN(n, ((cbegin + column) / bn) * bn - cbegin));
}

int starneig_matrix_cut_hor_right(
    int column, const starneig_matrix_t descr)
{
    int cbegin = STARNEIG_MATRIX_CBEGIN(descr);
    int bn = STARNEIG_MATRIX_BN(descr);
    int n = STARNEIG_MATRIX_N(descr);

    return MAX(0, MIN(n, divceil(cbegin + column, bn) * bn - cbegin));
}

int starneig_single_owner_matrix_descr(
    int i, int j, void const *ptr)
{
    return *((int *) ptr);
}
