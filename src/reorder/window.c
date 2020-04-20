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
#include "window.h"
#include "../common/common.h"

struct window* starneig_create_window(
    int idx, int gidx, int begin, int end, int swaps)
{
    struct window *window = malloc(sizeof(struct window));
    window->idx = idx;
    window->gidx = gidx;
    window->begin = begin;
    window->end = end;
    window->swaps = swaps;
    window->lq_h = NULL;
    window->lz_h = NULL;
    window->up = NULL;
    window->down = NULL;

    return window;
}

void starneig_unregister_window(struct window *window)
{
    if (window == NULL)
        return;

    if (window->lq_h != NULL)
        starpu_data_unregister_submit(window->lq_h);

    if (window->lz_h != NULL)
        starpu_data_unregister_submit(window->lz_h);

    window->lq_h = NULL;
    window->lz_h = NULL;
}

void starneig_free_window(struct window *window)
{
    if (window != NULL) {
        starneig_unregister_window(window);
        free(window);
    }
}

struct window_chain* starneig_create_chain(int begin, int end)
{
    struct window_chain *chain = malloc(sizeof(struct window_chain));
    chain->begin = begin;
    chain->end = end;
    chain->length = 0;
    chain->effective_length = 0;
    chain->top = NULL;
    chain->bottom = NULL;
    chain->up = NULL;
    chain->down = NULL;

    return chain;
}

void starneig_add_window_to_chain_top(
    struct window *window, struct window_chain *chain)
{
    // if the chain is empty, put the new window to the bottom of the chain
    if (chain->bottom == NULL)
        chain->bottom = window;

    // link and put the new window to the top of the chain
    window->down = chain->top;
    if (chain->top != NULL)
        chain->top->up = window;
    chain->top = window;

    chain->length++;
    chain->effective_length++;
}

struct window* starneig_pop_window_from_chain_bottom(struct window_chain *chain)
{
    if (chain->bottom == NULL)
        return NULL;

    struct window *bottom = chain->bottom;

    if (bottom->up != NULL) {
        bottom->up->down = NULL;
        chain->bottom = bottom->up;
    }
    else {
        chain->top = NULL;
        chain->bottom = NULL;
    }

    bottom->up = NULL;
    bottom->down = NULL;

    chain->length--;


    return bottom;
}

void starneig_unregister_chain(struct window_chain *chain)
{
    if (chain == NULL)
        return;

    struct window *it = chain->top;
    while (it != NULL) {
        starneig_unregister_window(it);
        it = it->down;
    }
}

void starneig_free_chain(struct window_chain *chain)
{
    if (chain == NULL)
        return;

    struct window *it = chain->bottom;
    while (it != NULL) {
        struct window *next = it->up;
        starneig_free_window(it);
        it = next;
    }

    free(chain);
}

struct chain_list* starneig_create_chain_list()
{
    struct chain_list *list = malloc(sizeof(struct chain_list));
    list->top = NULL;
    list->bottom = NULL;
    list->next = NULL;
    list->prev = NULL;
    list->longest_eff_length = 0;
    list->total_length = 0;
    return list;
}

void starneig_add_chain_to_list_bottom(
    struct window_chain *chain, struct chain_list *list)
{
    // if the chain list is empty, put the new chain to the top of the list
    if (list->top == NULL)
        list->top = chain;

    // link and put the new chain to the bottom of the list
    chain->up = list->bottom;
    if (list->bottom != NULL)
        list->bottom->down = chain;
    list->bottom = chain;

    list->longest_eff_length =
        MAX(list->longest_eff_length, chain->effective_length);
    list->total_length += chain->length;
}

void starneig_sever_chain_from_list(
    struct window_chain *chain, struct chain_list *list)
{
    if (chain->up != NULL)
        chain->up->down = chain->down;
    else
        list->top = chain->down;

    if (chain->down != NULL)
        chain->down->up = chain->up;
    else
        list->bottom = chain->up;

    chain->up = NULL;
    chain->down = NULL;

    list->total_length -= chain->length;
}

void starneig_unregister_chain_list(struct chain_list *list)
{
    if (list == NULL)
        return;

    struct window_chain *it = list->top;
    while (it != NULL) {
        starneig_unregister_chain(it);
        it = it->down;
    }
}

void starneig_free_chain_list(struct chain_list *list)
{
    if (list == NULL)
        return;

    struct window_chain *it = list->top;
    while (it != NULL) {
        struct window_chain *next = it->down;
        starneig_free_chain(it);
        it = next;
    }

    free(list);
}
