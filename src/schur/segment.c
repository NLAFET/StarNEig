///
/// @file
///
/// @brief This file contains code which is related segments.
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
#include "segment.h"
#include <math.h>

struct segment * starneig_create_segment(
    enum segment_status status, int begin, int end)
{
    struct segment *segment = malloc(sizeof(struct segment));
    memset(segment, 0, sizeof(struct segment));

    segment->status = status;
    segment->begin = begin;
    segment->end = end;

    segment->peak_submitted = 0;
    segment->peak_time = 0;
    segment->slope = NAN;

    return segment;
}

struct segment * starneig_move_segment(struct segment *segment)
{
    if (segment == NULL)
        return NULL;

    struct segment *new = malloc(sizeof(struct segment));

    // move everything
    memcpy(new, segment, sizeof(struct segment));
    memset(segment, 0, sizeof(struct segment));

    // handle special cases
    segment->status = SEGMENT_EMPTY;
    segment->iter = new->iter;
    segment->up = new->up;
    segment->down = new->down;
    new->up = new->down = NULL;

    return new;
}

void starneig_free_segment_list(struct segment_list *list);

void starneig_free_segment(struct segment *segment)
{
    if (segment == NULL)
        return;

    if (segment->small_status_h != NULL)
        starpu_data_unregister_submit(segment->small_status_h);

    if (segment->aed_small_lZ_h != NULL &&
    segment->aed_small_lZ_h != segment->aed_small_lQ_h)
        starpu_data_unregister_submit(segment->aed_small_lZ_h);
    if (segment->aed_small_lQ_h != NULL)
        starpu_data_unregister_submit(segment->aed_small_lQ_h);

    if (segment->aed_status_h != NULL)
        starpu_data_unregister_submit(segment->aed_status_h);

    starneig_free_matrix_descr(segment->aed_args.matrix_a);
    starneig_free_matrix_descr(segment->aed_args.matrix_b);
    starneig_free_matrix_descr(segment->aed_args.matrix_q);
    starneig_free_matrix_descr(segment->aed_args.matrix_z);

    if (segment->aed_deflate_status_h != NULL)
        starpu_data_unregister_submit(segment->aed_deflate_status_h);
    if (segment->aed_deflate_inducer_h != NULL)
        starpu_data_unregister_submit(segment->aed_deflate_inducer_h);
    starneig_free_vector_descr(segment->aed_deflate_base);

    starneig_free_vector_descr(segment->shifts_real);
    starneig_free_vector_descr(segment->shifts_imag);

    starneig_free_vector_descr(segment->bulges_aftermath);

    starneig_free_segment_list(segment->children);

    free(segment);
}

struct segment_list * starneig_create_segment_list() {
    struct segment_list *list = malloc(sizeof(struct segment_list));
    list->top = list->bottom = NULL;
    return list;
}

void starneig_add_segment_to_list_top(
    struct segment *segment, struct segment_list *list)
{
    assert (list != NULL);
    if (segment == NULL)
        return;

    if (list->top == NULL) {
        list->top = list->bottom = segment;
        return;
    }

    list->top->up = segment;
    segment->down = list->top;
    list->top = segment;
}

void starneig_add_segment_to_list_bottom(
    struct segment *segment, struct segment_list *list)
{
    assert (list != NULL);
    if (segment == NULL)
        return;

    if (list->bottom == NULL) {
        list->top = list->bottom = segment;
        return;
    }

    list->bottom->down = segment;
    segment->up = list->bottom;
    list->bottom = segment;
}

void starneig_remove_segment_from_list(
    struct segment *segment, struct segment_list *list)
{
    if (segment == NULL || list == NULL)
        return;

    // fix segment pointers
    if (segment->down != NULL)
        segment->down->up = segment->up;
    if (segment->up != NULL)
        segment->up->down = segment->down;

    // fix segment list pointers
    if (list->top == segment)
        list->top = segment->down;
    if (list->bottom == segment)
        list->bottom = segment->up;

    segment->up = segment->down = NULL;
}

void starneig_replace_segment_with_list(struct segment *segment,
    struct segment_list *sublist, struct segment_list *list)
{
    assert (segment != NULL && list != NULL);

    if (sublist == NULL || sublist->top == NULL) {
        starneig_remove_segment_from_list(segment, list);
        starneig_free_segment_list(sublist);
        starneig_free_segment(segment);
        return;
    }

    // free_segment frees the children!
    if (segment->children == sublist)
        segment->children = NULL;

    // fix segment pointers
    if (segment->down != NULL) {
        segment->down->up = sublist->bottom;
        sublist->bottom->down = segment->down;
    }
    if (segment->up != NULL) {
        segment->up->down = sublist->top;
        sublist->top->up = segment->up;
    }

    // fix segment list pointers
    if (list->top == segment)
        list->top = sublist->top;
    if (list->bottom == segment)
        list->bottom = sublist->bottom;

    starneig_free_segment(segment);
    free(sublist);
}

void starneig_free_segment_list(struct segment_list *list)
{
    if (list == NULL)
        return;

    struct segment *iter = list->top;
    while (iter != NULL) {
        struct segment *next = iter->down;
        starneig_free_segment(iter);
        iter = next;
    }

    free(list);
}
