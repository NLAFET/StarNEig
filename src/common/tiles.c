///
/// @file
///
/// @brief This file contains the definition of a tile packing helper subsystem
/// that is used throughout all components of the library.
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
#include "tiles.h"
#include "common.h"
#include "scratch.h"

///
/// @brief Re-sizes packing helper structure array if necessary.
///
/// @param[in]    k        number of handles to back
/// @param[in,out] helper  packing helper structure
///
static void prep_packing_helper(int k, struct packing_helper *helper)
{
    int left = helper->size - helper->count;
    if (left < k) {
        int new_size = helper->size + k - left + 10;

        struct starpu_data_descr *new_descrs =
            malloc(new_size*sizeof(struct starpu_data_descr));
        memcpy(new_descrs, helper->descrs,
            helper->size*sizeof(struct starpu_data_descr));
        free(helper->descrs);
        helper->descrs = new_descrs;

        packing_mode_flag_t *new_flags =
            malloc(new_size*sizeof(packing_mode_flag_t));
        memcpy(new_flags, helper->flags,
            helper->size*sizeof(packing_mode_flag_t));
        free(helper->flags);
        helper->flags = new_flags;

        helper->size = new_size;
    }
}

///
/// @brief Pre-fills packing info structure.
///
/// @param[in]  rbegin  first row that belongs to the window
/// @param[in]  rend    last row that belongs to the window + 1
/// @param[in]  cbegin  first column that belongs to the window
/// @param[in]  cend    last column that belongs to the window + 1
/// @param[in]  matrix  matrix descriptor
/// @param[out] info    packing information
/// @param[in]  flag    packing mode flag
///
static void prefill_packing_info(
    int rbegin, int rend, int cbegin, int cend,
    const starneig_matrix_descr_t matrix, struct packing_info *info,
    packing_mode_flag_t flag)
{
    STARNEIG_ASSERT(0 <= rbegin && rend <= STARNEIG_MATRIX_M(matrix));
    STARNEIG_ASSERT(0 <= cbegin && cend <= STARNEIG_MATRIX_N(matrix));

    int rbbegin = STARNEIG_MATRIX_TILE_IDX(rbegin, matrix);
    int cbbegin = STARNEIG_MATRIX_TILE_IDY(cbegin, matrix);

    info->flag = flag;
    info->elemsize = STARNEIG_MATRIX_ELEMSIZE(matrix);
    info->bm = STARNEIG_MATRIX_BM(matrix);
    info->bn = STARNEIG_MATRIX_BN(matrix);

    info->rbegin = STARNEIG_MATRIX_RBEGIN(matrix) + rbegin -
        rbbegin*STARNEIG_MATRIX_BM(matrix);
    info->rend = STARNEIG_MATRIX_RBEGIN(matrix) + rend -
        rbbegin*STARNEIG_MATRIX_BM(matrix);
    info->cbegin = STARNEIG_MATRIX_CBEGIN(matrix) + cbegin -
        cbbegin*STARNEIG_MATRIX_BN(matrix);
    info->cend = STARNEIG_MATRIX_CBEGIN(matrix) + cend -
        cbbegin*STARNEIG_MATRIX_BN(matrix);

    info->m = STARNEIG_MATRIX_M(matrix);
    info->n = STARNEIG_MATRIX_N(matrix);
    info->roffset = rbegin;
    info->coffset = cbegin;

    info->handles = 0;
}


static void pack_window_full(
    enum starpu_data_access_mode mode,
    int rbegin, int rend, int cbegin, int cend,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag)
{
    STARNEIG_ASSERT(!(flag & PACKING_MODE_SUBMIT_UNREGISTER));

    int rbbegin = STARNEIG_MATRIX_TILE_IDX(rbegin, matrix);
    int rbend = STARNEIG_MATRIX_TILE_IDX(rend-1, matrix) + 1;

    int cbbegin = STARNEIG_MATRIX_TILE_IDY(cbegin, matrix);
    int cbend = STARNEIG_MATRIX_TILE_IDY(cend-1, matrix) + 1;

    prep_packing_helper((rbend-rbbegin)*(cbend-cbbegin), helper);

    prefill_packing_info(rbegin, rend, cbegin, cend, matrix, info, flag);

    struct starpu_data_descr *descrs = helper->descrs + helper->count;
    packing_mode_flag_t *flags = helper->flags + helper->count;

    int k = 0;
    for (int i = cbbegin; i < cbend; i++) {
        for (int j = rbbegin; j < rbend; j++) {
            descrs[k].handle =
                starneig_get_tile_from_matrix_descr(j, i, matrix);
            descrs[k].mode = mode;
            flags[k] = PACKING_MODE_DEFAULT;
            k++;
        }
    }

    helper->count += k;
    info->handles = k;

#ifdef STARNEIG_ENABLE_EVENTS
    info->event_label = matrix->event_label;
    info->event_enabled = matrix->event_enabled;
    info->event_roffset = matrix->event_roffset;
    info->event_coffset = matrix->event_coffset;
#endif
}

static void pack_window_upper_hess(
    enum starpu_data_access_mode mode, int begin, int end,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag)
{
    STARNEIG_ASSERT(!(flag & PACKING_MODE_SUBMIT_UNREGISTER));
    STARNEIG_ASSERT(STARNEIG_MATRIX_RBEGIN(matrix) == STARNEIG_MATRIX_CBEGIN(matrix));

    int rbbegin = STARNEIG_MATRIX_TILE_IDX(begin, matrix);
    int rbend = STARNEIG_MATRIX_TILE_IDX(end-1, matrix) + 1;

    int cbbegin = STARNEIG_MATRIX_TILE_IDY(begin, matrix);
    int cbend = STARNEIG_MATRIX_TILE_IDY(end-1, matrix) + 1;

    int tiles = 0;
    for (int i = cbbegin; i < cbend; i++)
        for (int j = rbbegin; j < rbend; j++)
            if (
            j*STARNEIG_MATRIX_BM(matrix) <= (i+1)*STARNEIG_MATRIX_BN(matrix))
                tiles++;

    prep_packing_helper(tiles, helper);

    prefill_packing_info(begin, end, begin, end, matrix, info, flag);

    struct starpu_data_descr *descrs = helper->descrs + helper->count;
    packing_mode_flag_t *flags = helper->flags + helper->count;

    int k = 0;
    for (int i = cbbegin; i < cbend; i++) {
        for (int j = rbbegin; j < rbend; j++) {
            if (
            j*STARNEIG_MATRIX_BM(matrix) <= (i+1)*STARNEIG_MATRIX_BN(matrix)) {
                descrs[k].handle =
                    starneig_get_tile_from_matrix_descr(j, i, matrix);
                descrs[k].mode = mode;
                flags[k] = PACKING_MODE_DEFAULT;
                k++;
            }
        }
    }

    helper->count += k;
    info->handles = k;

#ifdef STARNEIG_ENABLE_EVENTS
    info->event_label = matrix->event_label;
    info->event_enabled = matrix->event_enabled;
    info->event_roffset = matrix->event_roffset;
    info->event_coffset = matrix->event_coffset;
#endif
}

static void pack_window_upper_triag(
    enum starpu_data_access_mode mode, int begin, int end,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag)
{
    STARNEIG_ASSERT(!(flag & PACKING_MODE_SUBMIT_UNREGISTER));
    STARNEIG_ASSERT(STARNEIG_MATRIX_RBEGIN(matrix) == STARNEIG_MATRIX_CBEGIN(matrix));

    int rbbegin = STARNEIG_MATRIX_TILE_IDX(begin, matrix);
    int rbend = STARNEIG_MATRIX_TILE_IDX(end-1, matrix) + 1;

    int cbbegin = STARNEIG_MATRIX_TILE_IDY(begin, matrix);
    int cbend = STARNEIG_MATRIX_TILE_IDY(end-1, matrix) + 1;

    int tiles = 0;
    for (int i = cbbegin; i < cbend; i++)
        for (int j = rbbegin; j < rbend; j++)
            if (
            j*STARNEIG_MATRIX_BM(matrix) <= (i+1)*STARNEIG_MATRIX_BN(matrix)-1)
                tiles++;

    prep_packing_helper(tiles, helper);

    prefill_packing_info(begin, end, begin, end, matrix, info, flag);

    struct starpu_data_descr *descrs = helper->descrs + helper->count;
    packing_mode_flag_t *flags = helper->flags + helper->count;

    int k = 0;
    for (int i = cbbegin; i < cbend; i++) {
        for (int j = rbbegin; j < rbend; j++) {
            if (j*STARNEIG_MATRIX_BM(matrix) <=
            (i+1)*STARNEIG_MATRIX_BN(matrix)-1) {
                descrs[k].handle =
                    starneig_get_tile_from_matrix_descr(j, i, matrix);
                descrs[k].mode = mode;
                flags[k] = PACKING_MODE_DEFAULT;
                k++;
            }
        }
    }

    helper->count += k;
    info->handles = k;

#ifdef STARNEIG_ENABLE_EVENTS
    info->event_label = matrix->event_label;
    info->event_enabled = matrix->event_enabled;
    info->event_roffset = matrix->event_roffset;
    info->event_coffset = matrix->event_coffset;
#endif
}

static void join_tiles_full(
    int rbegin, int rend, int cbegin, int cend,
    int bm, int bn, size_t in_ld, size_t out_ld, size_t elemsize,
    struct starpu_matrix_interface **in, void *out, int reverse)
{
    // first tile row and last tile row + 1
    int tr_begin = rbegin / bm;
    int tr_end = (rend - 1) / bm + 1;

    // first tile column and last tile column + 1
    int tc_begin = cbegin / bn;
    int tc_end = (cend - 1) / bn + 1;

    //
    // go through all tiles that make up the window
    //

    for (int i = tc_begin; i < tc_end; i++) {

        // vertical bounds inside the current tile
        int _cbegin = MAX(0, cbegin - i * bn);
        int _cend = MIN(bn, cend - i * bn);

        // vertical offset inside the output buffer
        int column_offset = MAX(0, i * bn - cbegin);

        for (int j = tr_begin; j < tr_end; j++) {

            // horizontal bounds inside the current tile
            int _rbegin = MAX(0, rbegin - j * bm);
            int _rend = MIN(bm, rend - j * bm);

            // horizontal offset inside the output buffer
            int row_offset = MAX(0, j * bm - rbegin);

            //
            // copy
            //

            void *ptr_ = (void *) STARPU_MATRIX_GET_PTR(in[i*in_ld+j]);
            size_t _ld = STARPU_MATRIX_GET_LD(in[i*in_ld+j]);

            if (reverse)
                starneig_copy_matrix(
                    _rend - _rbegin, _cend - _cbegin, out_ld, _ld, elemsize,
                    out + (column_offset*out_ld + row_offset)*elemsize,
                    ptr_ + (_cbegin*_ld + _rbegin)*elemsize);
            else
                starneig_copy_matrix(
                    _rend - _rbegin, _cend - _cbegin, _ld, out_ld, elemsize,
                    ptr_ + (_cbegin*_ld + _rbegin)*elemsize,
                    out + (column_offset*out_ld + row_offset)*elemsize);
        }
    }
}

static void join_tiles_upper_hess(
    int rbegin, int rend, int cbegin, int cend, int bm, int bn,
    size_t out_ld, size_t elemsize, struct starpu_matrix_interface **in,
    void *out, int reverse)
{
    //
    // initialize the lower left corner of the output buffer to zero
    //

    if (!reverse) {
        int n = cend - cbegin;
        for (int i = 0; i < n; i++) {
            int m = (rend-rbegin)-i;
            memset(out+(i*out_ld+i)*elemsize, 0, m*elemsize);
        }
    }

    // first tile row and last tile row + 1
    int tr_begin = rbegin / bm;
    int tr_end = (rend - 1) / bm + 1;

    // first tile column and last tile column + 1
    int tc_begin = cbegin / bn;
    int tc_end = (cend - 1) / bn + 1;

    //
    // go through all tiles that make up the window
    //

    int tid = 0;
    for (int i = tc_begin; i < tc_end; i++) {

        // vertical bounds inside the current tile
        int _cbegin = MAX(0, cbegin - i * bn);
        int _cend = MIN(bn, cend - i * bn);

        // vertical offset inside the output buffer
        int column_offset = MAX(0, i * bn - cbegin);

        for (int j = tr_begin; j < tr_end; j++) {

            if (cbegin+(i+1)*bn < rbegin+j*bm)
                continue;

            // horizontal bounds inside the current tile
            int _rbegin = MAX(0, rbegin - j * bm);
            int _rend = MIN(bm, rend - j * bm);

            // horizontal offset inside the output buffer
            int row_offset = MAX(0, j * bm - rbegin);

            //
            // copy
            //

            void *ptr_ = (void *) STARPU_MATRIX_GET_PTR(in[tid]);
            size_t _ld = STARPU_MATRIX_GET_LD(in[tid]);

            if (reverse)
                starneig_copy_matrix(
                    _rend - _rbegin, _cend - _cbegin, out_ld, _ld, elemsize,
                    out + (column_offset*out_ld + row_offset)*elemsize,
                    ptr_ + (_cbegin*_ld + _rbegin)*elemsize);
            else
                starneig_copy_matrix(
                    _rend - _rbegin, _cend - _cbegin, _ld, out_ld, elemsize,
                    ptr_ + (_cbegin*_ld + _rbegin)*elemsize,
                    out + (column_offset*out_ld + row_offset)*elemsize);

            tid++;
        }
    }
}

static void join_tiles_upper_triag(
    int rbegin, int rend, int cbegin, int cend, int bm, int bn,
    size_t out_ld, size_t elemsize, struct starpu_matrix_interface **in,
    void *out, int reverse)
{

    //
    // initialize the lower left corner of the output buffer to zero
    //

    if (!reverse) {
        int n = cend - cbegin;
        for (int i = 0; i < n; i++) {
            int m = (rend-rbegin)-i;
            memset(out+(i*out_ld+i)*elemsize, 0, m*elemsize);
        }
    }

    // first tile row and last tile row + 1
    int tr_begin = rbegin / bm;
    int tr_end = (rend - 1) / bm + 1;

    // first tile column and last tile column + 1
    int tc_begin = cbegin / bn;
    int tc_end = (cend - 1) / bn + 1;

    //
    // go through all tiles that make up the window
    //

    int tid = 0;
    for (int i = tc_begin; i < tc_end; i++) {

        // vertical bounds inside the current tile
        int _cbegin = MAX(0, cbegin - i * bn);
        int _cend = MIN(bn, cend - i * bn);

        // vertical offset inside the output buffer
        int column_offset = MAX(0, i * bn - cbegin);

        for (int j = tr_begin; j < tr_end; j++) {

            if (cbegin+(i+1)*bn <= rbegin+j*bm)
                continue;

            // horizontal bounds inside the current tile
            int _rbegin = MAX(0, rbegin - j * bm);
            int _rend = MIN(bm, rend - j * bm);

            // horizontal offset inside the output buffer
            int row_offset = MAX(0, j * bm - rbegin);

            //
            // copy
            //

            void *ptr_ = (void *) STARPU_MATRIX_GET_PTR(in[tid]);
            size_t _ld = STARPU_MATRIX_GET_LD(in[tid]);

            if (reverse)
                starneig_copy_matrix(
                    _rend - _rbegin, _cend - _cbegin, out_ld, _ld, elemsize,
                    out + (column_offset*out_ld + row_offset)*elemsize,
                    ptr_ + (_cbegin*_ld + _rbegin)*elemsize);
            else
                starneig_copy_matrix(
                    _rend - _rbegin, _cend - _cbegin, _ld, out_ld, elemsize,
                    ptr_ + (_cbegin*_ld + _rbegin)*elemsize,
                    out + (column_offset*out_ld + row_offset)*elemsize);

            tid++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_init_empty_packing_info(struct packing_info *info)
{
    memset(info, 0, sizeof(struct packing_info));
}

void starneig_init_empty_range_packing_info(struct range_packing_info *info)
{
    memset(info, 0, sizeof(struct range_packing_info));
}

struct packing_helper * starneig_init_packing_helper()
{
    struct packing_helper *helper = malloc(sizeof(struct packing_helper));
    memset(helper, 0, sizeof(struct packing_helper));
    prep_packing_helper(10, helper);
    return helper;
}

void starneig_free_packing_helper(struct packing_helper *helper)
{
    if (helper == NULL)
        return;

    for (int i = 0; i < helper->count; i++)
        if (helper->flags[i] & PACKING_MODE_SUBMIT_UNREGISTER)
            starpu_data_unregister_submit(helper->descrs[i].handle);

    starneig_scratch_flush();

    free(helper->descrs);
    free(helper->flags);
    free(helper);
}

void starneig_pack_handle(
    enum starpu_data_access_mode mode, starpu_data_handle_t handle,
    struct packing_helper *helper, packing_mode_flag_t flag)
{
    prep_packing_helper(1, helper);
    helper->descrs[helper->count].handle = handle;
    helper->descrs[helper->count].mode = mode;
    helper->flags[helper->count] = flag;
    helper->count++;
}

void starneig_pack_scratch_matrix(
    int m, int n, size_t elemsize, struct packing_helper *helper)
{
    starpu_data_handle_t handle;
    starpu_matrix_data_register(&handle, -1, 0, m, m, n, elemsize);
    starneig_pack_handle(
        STARPU_SCRATCH, handle, helper, PACKING_MODE_SUBMIT_UNREGISTER);
}

void starneig_pack_cached_scratch_matrix(
    int m, int n, size_t elemsize, struct packing_helper *helper)
{
    starpu_data_handle_t handle = starneig_scratch_get_matrix(m, n, elemsize);
    starneig_pack_handle(
        STARPU_SCRATCH, handle, helper, PACKING_MODE_DEFAULT);
}

void starneig_pack_range(
    enum starpu_data_access_mode mode, int begin, int end,
    starneig_vector_descr_t vector, struct packing_helper *helper,
    struct range_packing_info *info, packing_mode_flag_t flag)
{
    if (vector == NULL) {
        starneig_init_empty_range_packing_info(info);
        return;
    }

    STARNEIG_ASSERT(!(flag & PACKING_MODE_SUBMIT_UNREGISTER));

    int bbegin = begin / STARNEIG_VECTOR_BM(vector);
    int bend = (end-1) / STARNEIG_VECTOR_BM(vector) + 1;

    prep_packing_helper(bend - bbegin , helper);

    struct starpu_data_descr *descrs = helper->descrs + helper->count;
    packing_mode_flag_t *flags = helper->flags + helper->count;

    int k = 0;
    for (int i = bbegin; i < bend; i++) {
        descrs[k].handle = starneig_get_tile_from_vector_descr(i, vector);
        descrs[k].mode = mode;
        flags[k] = flag;
        k++;
    }

    helper->count += k;

    info->flag = flag;
    info->elemsize = STARNEIG_VECTOR_ELEMSIZE(vector);
    info->bm = STARNEIG_VECTOR_BM(vector);
    info->begin = begin - bbegin*STARNEIG_VECTOR_BM(vector);
    info->end = end - bbegin*STARNEIG_VECTOR_BM(vector);
    info->m = STARNEIG_VECTOR_M(vector);
    info->offset = begin;
    info->handles = k;
}

void starneig_join_range(
    struct range_packing_info const *packing_info,
    struct starpu_vector_interface **in, void *out, int reverse)
{
    if (packing_info->handles == 0)
        return;

    // first tile row and last tile row + 1
    int t_begin = packing_info->begin / packing_info->bm;
    int t_end = (packing_info->end - 1) / packing_info->bm + 1;

    //
    // go through all tiles that make up the window
    //

    for (int i = t_begin; i < t_end; i++) {

        // horizontal bounds inside the current tile
        int _begin = MAX(0, packing_info->begin - i * packing_info->bm);
        int _end =
            MIN(packing_info->bm, packing_info->end - i * packing_info->bm);

        // horizontal offset inside the output buffer
        int row_offset = MAX(0, i * packing_info->bm - packing_info->begin);

        // copy

        void *ptr = (void *) STARPU_VECTOR_GET_PTR(in[i]);

        if (reverse)
            memcpy(ptr+_begin*packing_info->elemsize,
                out+row_offset*packing_info->elemsize,
                (_end-_begin)*packing_info->elemsize);
        else
            memcpy(out+row_offset*packing_info->elemsize,
                ptr+_begin*packing_info->elemsize,
                (_end-_begin)*packing_info->elemsize);
    }
}

void starneig_pack_window(
    enum starpu_data_access_mode mode,
    int rbegin, int rend, int cbegin, int cend,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag)
{
    if (matrix == NULL) {
        starneig_init_empty_packing_info(info);
        return;
    }

    pack_window_full(
        mode, rbegin, rend, cbegin, cend, matrix, helper, info, flag);
}

void starneig_join_window(
    struct packing_info const *packing_info, size_t ld,
    struct starpu_matrix_interface **in, void *out, int reverse)
{
    if (packing_info->handles == 0)
        return;

    join_tiles_full(
        packing_info->rbegin, packing_info->rend,
        packing_info->cbegin, packing_info->cend,
        packing_info->bm, packing_info->bn,
        divceil(packing_info->rend, packing_info->bm),
        ld, packing_info->elemsize, in, out, reverse);
}

void starneig_join_sub_window(
    int rbegin, int rend, int cbegin, int cend,
    struct packing_info const *packing_info, size_t ld,
    struct starpu_matrix_interface **in, void *out, int reverse)
{
    if (packing_info->handles == 0)
        return;

    join_tiles_full(
        packing_info->rbegin+rbegin, packing_info->rbegin+rend,
        packing_info->cbegin+cbegin, packing_info->cbegin+cend,
        packing_info->bm, packing_info->bn,
        divceil(packing_info->rend, packing_info->bm),
        ld, packing_info->elemsize, in, out, reverse);
}

void starneig_pack_diag_window(
    enum starpu_data_access_mode mode, int begin, int end,
    starneig_matrix_descr_t matrix, struct packing_helper *helper,
    struct packing_info *info, packing_mode_flag_t flag)
{
    if (matrix == NULL) {
        starneig_init_empty_packing_info(info);
        return;
    }

    if (flag & PACKING_MODE_UPPER_HESSENBERG)
        pack_window_upper_hess(mode, begin, end, matrix, helper, info, flag);
    else if (flag & PACKING_MODE_UPPER_TRIANGULAR)
        pack_window_upper_triag(mode, begin, end, matrix, helper, info, flag);
    else
        pack_window_full(
            mode, begin, end, begin, end, matrix, helper, info, flag);
}

void starneig_join_diag_window(
    struct packing_info const *packing_info, size_t ld,
    struct starpu_matrix_interface **in, void *out, int reverse)
{
    if (packing_info->handles == 0)
        return;

    if (packing_info->flag & PACKING_MODE_UPPER_HESSENBERG)
        join_tiles_upper_hess(
            packing_info->rbegin, packing_info->rend,
            packing_info->cbegin, packing_info->cend,
            packing_info->bm, packing_info->bn,
            ld, packing_info->elemsize, in, out, reverse);
    else if (packing_info->flag & PACKING_MODE_UPPER_TRIANGULAR)
        join_tiles_upper_triag(
            packing_info->rbegin, packing_info->rend,
            packing_info->cbegin, packing_info->cend,
            packing_info->bm, packing_info->bn,
            ld, packing_info->elemsize, in, out, reverse);
    else
        join_tiles_full(
            packing_info->rbegin, packing_info->rend,
            packing_info->cbegin, packing_info->cend,
            packing_info->bm, packing_info->bn,
            divceil(packing_info->rend, packing_info->bm),
            ld, packing_info->elemsize, in, out, reverse);
}
