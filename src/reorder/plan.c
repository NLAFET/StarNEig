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
#include "../common/common.h"
#include "plan.h"

///
/// @brief Creates an empty eigenvalue reordering plan.
///
/// @return reordering plan
///
static struct plan* create_empty_plan()
{
    struct plan *plan = malloc(sizeof(struct plan));
    plan->begin = NULL;
    plan->end = NULL;
    plan->longest_eff_length = 0;
    plan->total_length = 0;
    return plan;
}

///
/// @brief Adds a chain list to a eigenvalue reordering plan.
///
/// @para[in,out] list
///             chain list
///
/// @para[in,out] plan
///             eigenvalue reordering plan
///
static void add_chain_list_to_plan(
    struct chain_list *list, struct plan *plan)
{
    // the plan is empty, put the chain list to the beginning
    if (plan->begin == NULL)
        plan->begin = list;

    // link and put the chain list to the end
    list->prev = plan->end;
    if (plan->end != NULL)
        plan->end->next = list;
    plan->end = list;

    // update chain lengths
    plan->longest_eff_length =
        MAX(plan->longest_eff_length, list->longest_eff_length);
    plan->total_length += list->total_length;
}

///
/// @brief Emulates what happens to the eigenvalue selection and complex
/// eigenvalue distribution bitmaps when a diagonal computation window is
/// processed.
///
/// @param[in] begin
///         first row that belongs to the window
///
/// @param[in] end
///         last row that belongs to the window + 1
///
/// @param[in,out] selected
///         eigenvalue selection vector
///
/// @param[in,out] complex_distr
///         complex eigenvalue distribution vector
///
/// @param[out] swaps
///         returns the total number of involved diagonal block swaps
///
/// @return location of the topmost deselected eigenvalue
///
static int update_bitmaps(
    int begin, int end, int *selected, int *complex_distr, int *swaps)
{
    // a variable to keep track where the next selected eigenvalue should be
    // moved
    int top = begin;

    if (swaps)
        *swaps = 0;

    for (int i = begin; i < end; i++) {

        // is the eigenvalue selected?
        if (selected[i]) {

            if (swaps)
                *swaps += i-top;

            // is it a 2-by-2 tile?
            if (i+1 < end && complex_distr[i+1]) {

                // the eigenvalue is moved to its appropriate place near the
                // upper left corner of the window
                selected[i] = 0;
                selected[i+1] = 0;
                selected[top] = 1;
                selected[top+1] = 1;

                // all other eigenvalue between the original location and the
                // new location are moved downward
                for (int j = i+1; top+1 < j; j--)
                    complex_distr[j] = complex_distr[j-2];

                // mark the eigenvalue as a 2-by-2 tile
                complex_distr[top] = 0;
                complex_distr[top+1] = 1;

                top += 2;
                i++;
            }
            else {

                // a 1x1 tile is processed similarly

                selected[i] = 0;
                selected[top] = 1;

                for (int j = i; top < j; j--)
                    complex_distr[j] = complex_distr[j-1];

                top++;
            }
        }
    }

    return top;
}

///
/// @brief Finds a diagonal window that contains a desired number of selected
/// eigenvalues.
///
/// @param[in] begin          - first row that that should belong to the
///                             diagonal window
/// @param[in] limit          - upper limit for how many eigenvalues are to be
///                             included
/// @param[in] n              - matrix dimension
/// @param[in] selected       - eigenvalue selection vector
/// @param[in] complex_distr  - complex eigenvalue distribution vector
/// @param[out] count         - returns the actual number of selected
///                             eigenvalues that fall within the diagonal window
///
/// @return last row that belongs to the diagonal window + 1
///
static int find_window(
    int begin, int limit, int n, int *selected, int *complex_distr, int *count)
{
    int end = begin;
    int values = 0;

    for (int i = begin; i < n; i++) {

        // is the eigenvalue selected?
        if (selected[i]) {

            // is it 2-by-2 tile?
            if (i < n-1 && complex_distr[i+1]) {

                // stop the search if the upper limit is about to be reached
                if (limit < values+2)
                    break;

                values += 2;
                end = i + 2;
                i++;
            }
            else {

                // 1x1 tiles are processed similarly

                if (limit < values+1)
                    break;
                values++;
                end = i + 1;

            }
        }
    }

    if (count != NULL)
        *count = values;

    return end;
}

///
/// @brief Takes an empty window chain descriptor and fills it with windows.
///
/// @param[in] window_size - window size (-1 => automatic)
/// @param[in] tile_size - tile size
/// @param[in,out] gidx - global index number counter
/// @param[in,out] selected - eigenvalue selection bitmap
/// @param[in,out] complex_distr - complex eigenvalue distribution bitmap
/// @param[in,out] chain - pointer to the chain
///
static void fill_chain(
    int window_size, int tile_size, int *gidx,
    int *selected, int *complex_distr, struct window_chain *chain)
{
    // start from the lower right corner
    int begin = chain->end, end = chain->end;

    for (int idx = 0; chain->begin < begin; idx++) {

        if (0 < window_size)
            begin = MAX(chain->begin, end-window_size);
        else
            // place the window such that upper edge of the window respects the
            // boundaries of the underlying data tiles
            begin = MAX(chain->begin, (divceil(end, tile_size)-2)*tile_size);

        if (begin-1 == chain->begin)
            // re-size the window if the next window is going to be to small
            begin = chain->begin;
        else if (complex_distr[begin] )
            // re-size the window if the upper left corner splits a 2-by-2 tile
            begin++;

        int next_end;

        int swaps;
        next_end = update_bitmaps(begin, end, selected, complex_distr, &swaps);

        starneig_add_window_to_chain_top(
            starneig_create_window(idx, (*gidx)++, begin, end, swaps), chain);

        // place the lower right corner of the next window as high as possible
        end = next_end;
    }
}

///
/// @brief Forms a simple reordering plan that contains a single chain list.
///
///  See reorder.h / starneig_reorder_plan for further documentation.
///
/// @param[in] n - problem dimension
/// @param[in] window_size - window size (-1 => automatic)
/// @param[in] values_per_chain - number of selected eigenvalues per window
///           chain (-1 => automatic)
/// @param[in] tile_size - tile size
/// @param[in,out] selected - eigenvalue selection bitmap
/// @param[in,out] complex_distr - complex eigenvalue distribution bitmap
///
/// @return pointer to the new reordering plan
///
static struct chain_list* form_simple_chain_list(
    int n, int window_size, int values_per_chain, int tile_size,
    int *selected, int *complex_distr)
{
    // calculate how many selected eigenvalues should be included to each window
    // chain
    int limit;
    if (0 < window_size) {
        if (0 < values_per_chain)
            limit = values_per_chain;
        else
            // this seems to be a somewhat optimal value in most situations
            limit = window_size/2;
    }
    else {
        // The fill_chain function will place the windows such that the window
        // size is 2*tile_size (-1) and tries to respect the boundaries of the
        // underlying data tiles. In order to avoid introducing excess MPI
        // communication, we want to be sure that all selected eigenvalues can
        // be fitted to the upper part of a window without spilling over to the
        // preceding tile.
        limit = tile_size-1;
    }

    struct chain_list *list = starneig_create_chain_list();


    // locate first deselected eigenvalue
    int begin = 0;
    while(selected[begin])
        begin++;
    int end = begin;

    int gidx = 0;

    while (1) {
        // compute the location of the lower right corner of the new window
        // chain
        int count;
        end = find_window(
            end, limit, n, selected, complex_distr, &count);

        // stop if there are no remaining selected eigenvalues
        if (count == 0)
            break;

        // create a new window chain and fill it with windows
        struct window_chain *chain = starneig_create_chain(begin, end);
        fill_chain(window_size, tile_size, &gidx,
            selected, complex_distr, chain);

        starneig_add_chain_to_list_bottom(chain, list);

        // place the begin location of the next chain appropriately
        begin += count;
    }

    return list;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_unregister_plan(struct plan *plan)
{
    if (plan == NULL)
        return;

    struct chain_list *it = plan->begin;
    while (it != NULL) {
        starneig_unregister_chain_list(it);
        it = it->next;
    }
}

void starneig_free_plan(struct plan *plan)
{
    if (plan == NULL)
        return;

    struct chain_list *it = plan->begin;
    while (it != NULL) {
        struct chain_list *next = it->next;
        starneig_free_chain_list(it);
        it = next;
    }

    free(plan);
}

struct plan* starneig_formulate_plan(
    int n, int window_size, int values_per_chain, int tile_size,
    int *selected, int *complex_distr)
{
    struct plan *plan = create_empty_plan();
    add_chain_list_to_plan(
        form_simple_chain_list(n, window_size, values_per_chain, tile_size,
            selected, complex_distr),
        plan);

    return plan;
}

struct plan* starneig_formulate_multiplan(
    int n, int window_size, int values_per_chain, int tile_size,
    int *selected, int *complex_distr)
{

    // form an initial chain list (one-part plan) that will serve as a template
    struct chain_list *temp = form_simple_chain_list(
        n, window_size, values_per_chain, tile_size,
        selected, complex_distr);

    //
    // form the actual plan
    //

    struct plan *plan = create_empty_plan();

    // keep splitting window chains until the template chain list becomes empty
    while (temp->top != NULL) {

        struct chain_list *list = starneig_create_chain_list();

        // a variable to keep track where the lower right corner of the most
        // recently added window was located
        int prev_end = 0;

        // go thought all remaining chains in the template chain list
        struct window_chain *it = temp->top;
        while (it != NULL) {

            struct window_chain *chain =
                starneig_create_chain(it->begin, it->end);

            // store the location of the lower right corner of the bottom window
            int next_end = it->bottom->end;

            // start from the bottom window and move upwards
            while (it->bottom != NULL && prev_end <= it->bottom->begin) {

                // if the window does not intersect any of the windows
                // added to the previous chains, remove it from the template
                // chain list and add it to the new window chain
                struct window *window =
                    starneig_pop_window_from_chain_bottom(it);
                starneig_add_window_to_chain_top(window, chain);
            }

            prev_end = divceil(next_end, tile_size)*tile_size;

            if (0 < chain->length) {
                // re-size the chain and add it to the chain list
                chain->begin = chain->top->begin;
                chain->end = chain->bottom->end;
                chain->effective_length = it->effective_length;
                starneig_add_chain_to_list_bottom(chain, list);
            }
            else {
                starneig_free_chain(chain);
            }

            struct window_chain *next = it->down;

            // remove empty chains from the template chain list
            if (it->length == 0) {
                starneig_sever_chain_from_list(it, temp);
                starneig_free_chain(it);
            }

            it = next;
        }

        add_chain_list_to_plan(list, plan);
    }

    starneig_free_chain_list(temp);

    return plan;
}
