///
/// @file This file contains the task insertion engine.
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
#include "insert_engine.h"
#include "../common/common.h"
#include "../common/tasks.h"
#include "tasks.h"
#include <math.h>

#ifdef STARNEIG_ENABLE_MPI
#include <starpu_mpi.h>
#endif

///
/// @brief Calculates priorities for low priority update tasks.
///
///  The update tasks are divided into priority groups and the length of the
///  longest window chain in the plan is used to decide the group size.
///
///  An example with 5 window chains and priority levels [-5,-1]:
///
///   0 1
///   0 1 2 3
///   0 1 2 3 4 5
///   0 1 2 3 4 5 6
///   0 1 2 3 4 5 6 8 9 <= longest chain (10 windows)
///   -----------------
///   -5 -5                There are 10/5 = 2 windows in each group.
///   -4 -4 -5 -5
///   -3 -3 -4 -4 -5 -5
///   -2 -3 -3 -4 -4 -5 -5
///   -1 -1 -2 -2 -3 -3 -4 -4 -5 -5
///
/// @param[in] idx      index number of the related diagonal window
/// @param[in] length   length of the related window chain
/// @param[in] longest  length of the longest window chain in the related plan
///
/// @return a priority between STARPU_MIN_PRIO and STARPU_DEFAULT_PRIO-1
///
static inline int calc_tile_prio(int idx, int length, int longest)
{
    int levels = (long)STARPU_DEFAULT_PRIO-STARPU_MIN_PRIO;

    if (0 < levels) {
        int group_size = divceil(longest, levels);
        return STARPU_DEFAULT_PRIO - (idx + longest - length)/group_size - 1;
    }

    return STARPU_DEFAULT_PRIO;
}

///
/// @brief Inserts a window processing task with the highest priority.
///
/// @param[in] small_window_size - small window size
/// @param[in] small_window_threshold - small window threshold
/// @param[in,out] selected - eigenvalue selection bitmap descriptor
/// @param[in,out] matrix_a - matrix A descriptor
/// @param[in,out] matrix_b - matrix B descriptor
/// @param[in,out] window - window descriptor
/// @param[in,out] mpi
///
static void dummy_insert_window(
    int small_window_size,
    int small_window_threshold,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct window *window,
    mpi_info_t mpi)
{
    starneig_reorder_insert_window(STARPU_MAX_PRIO,
        small_window_size, small_window_threshold, window, selected,
        matrix_a, matrix_b, mpi);
}

///
/// @brief Inserts related right update tasks with the second highest priority.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] window
///         windows descriptor
///
/// @param[in,out] matrix
///         matrix A/B descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void dummy_insert_right_update(
    int height, struct window const *window, starneig_matrix_descr_t matrix,
    mpi_info_t mpi)
{
    // figure out which accumulator matrix should be used
    starpu_data_handle_t operator;
    if (window->lz_h != NULL)
        operator = window->lz_h;
    else
        operator = window->lq_h;

    starneig_insert_right_gemm_update(
        0, window->begin, window->begin, window->end, height,
        MAX(STARPU_MAX_PRIO-1, STARPU_DEFAULT_PRIO), operator, matrix,
        mpi);
}

///
/// @brief Inserts related left update tasks with the default priority.
///
/// @param[in] width
///         width of a single update tasks
///
/// @param[in] window
///         windows descriptor
///
/// @param[in,out] matrix
///         matrix A/B descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void dummy_insert_left_update(
    int width, struct window const *window, starneig_matrix_descr_t matrix,
    mpi_info_t mpi)
{
    starneig_insert_left_gemm_update(
        window->begin, window->end, window->end, STARNEIG_MATRIX_N(matrix),
        width, STARPU_DEFAULT_PRIO, window->lq_h, matrix, mpi);
}

///
/// @brief Inserts Q matrix update tasks with the lowest priority.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] window
///         windows descriptor
///
/// @param[in,out] matrix
///         matrix Q descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void dummy_insert_q_update(
    int height, struct window const *window, starneig_matrix_descr_t matrix,
     mpi_info_t mpi)
{
    starneig_insert_right_gemm_update(
        0, STARNEIG_MATRIX_M(matrix), window->begin, window->end, height,
        STARPU_MIN_PRIO, window->lq_h, matrix, mpi);
}

///
/// @brief Inserts Z matrix update tasks with the lowest priority.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] window
///         windows descriptor
///
/// @param[in,out] matrix
///         matrix Z descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void dummy_insert_z_update(
    int height, struct window const *window, starneig_matrix_descr_t matrix,
    mpi_info_t mpi)
{
    starneig_insert_right_gemm_update(
        0, STARNEIG_MATRIX_M(matrix), window->begin, window->end, height,
        STARPU_MIN_PRIO, window->lz_h, matrix, mpi);
}

///
/// @brief Inserts a diagonal window and its dependencies.
///
/// The idea is to insert only those right-hand side updates (right_gemm_update
/// tasks) that are absolutely necessary for inserting the current diagonal
/// window (reorder_window task). The remaining right-hand side updates should
/// be inserted afterwards by using insert_low_prio_right_updates function.
///
///   +-----------+       aggregated updates
///   |           |               ||              00 right_gemm_update, window 0
///   |           |          +----------+         11 right_gemm_update, window 1
///   |           |          |          |         $$ overlap
///   |        ###|#########11111$$$$000000000
///   |        #  |     111#11111$$$$000000000
///   +-----------+     111#11111$$$$000000000
///            #        111#11111$$$$000000000 __________________
///            #        +--#--------+                 ^^^^
///            #        |  #        |         the right_gemm_update tasks
///            #############        |          below this line are
///             ^       |  window 1 |           already inserted
///             |       |        +--|--------+
///      current window |        |  |        |
///                     +-----------+        |
///                              |  window 0 |
///                              |           |
///                              |           |
///                              +-----------+
///
/// If the scheduler is priority-aware, then the inserted reorder_window and
/// right_gemm_update tasks will get the following priorities:
///
///   MAX <- MAX-1 <- MAX-2 <- MAX-3 <- MAX-4
///    ^___    :        :        :        :
///        '- MAX  <- MAX-1 <- MAX-2 <- MAX-3
///            ^___     :        :        :
///                '-- MAX  <- MAX-1 <- MAX-2
///                     ^___     :        :
///                        '-- MAX  <- MAX-1
///                              ^___     :
///                                  '-- MAX
///
/// In an attempt to avoid introducing new data dependencies in the vertical
/// direction and thus limiting the parallelism, the updates are inserted such
/// that their computation windows follow the boundaries of the underlying data
/// tiles. In some cases this design choice might lengthen the critical path
/// but the problem can be circumvented by choosing the diagonal computation
/// windows in such a that they follow the boundaries of the underlying data
/// tiles.
///
/// @param[in]     small_window_size       small window size
/// @param[in]     small_window_threshold  small window threshold
/// @param[in,out] selected                selection vector
/// @param[in,out] matrix_a                matrix A
/// @param[in,out] matrix_b                matrix B
/// @param[in,out] window                  window
/// @param[in,out] chain                   window chain
/// @param[in,out] mpi              MPI info
///
static void insert_window(
    int small_window_size,
    int small_window_threshold,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct window *window,
    struct window_chain *chain,
    mpi_info_t mpi)
{
    //
    // insert the related right update tasks
    //

    // go through all preceding windows in the chain
    for (struct window *it = chain->bottom; it != window; it = it->up) {

        starpu_data_handle_t operator;
        if (it->lz_h != NULL)
            operator = it->lz_h;
        else
            operator = it->lq_h;

        int prio = MAX(((long)STARPU_MAX_PRIO-STARPU_DEFAULT_PRIO)/2,
            STARPU_MAX_PRIO + it->idx - window->idx);

        // insert A matrix updates
        {
            int begin = ((STARNEIG_MATRIX_RBEGIN(matrix_a) + window->begin) /
                STARNEIG_MATRIX_BM(matrix_a))*STARNEIG_MATRIX_BM(matrix_a) -
                STARNEIG_MATRIX_RBEGIN(matrix_a);

            // the right update task that correspond to the previous diagonal
            // window is a special case and must be dealt differently
            int end;
            if(it == window->down)
                end = window->down->begin;
            else
                end =
                    ((STARNEIG_MATRIX_RBEGIN(matrix_a) + window->down->begin) /
                    STARNEIG_MATRIX_BM(matrix_a))*STARNEIG_MATRIX_BM(matrix_a) -
                    STARNEIG_MATRIX_RBEGIN(matrix_a);

            starneig_insert_right_gemm_update(begin, end, it->begin, it->end, 0,
                prio, operator, matrix_a, mpi);
        }

        // insert B matrix updates
        if (matrix_b != NULL) {
            int begin = ((STARNEIG_MATRIX_RBEGIN(matrix_b) + window->begin) /
                STARNEIG_MATRIX_BM(matrix_b))*STARNEIG_MATRIX_BM(matrix_b) -
                STARNEIG_MATRIX_RBEGIN(matrix_b);

            int end;
            if(it == window->down)
                end = window->down->begin;
            else
                end =
                    ((STARNEIG_MATRIX_RBEGIN(matrix_b) + window->down->begin) /
                    STARNEIG_MATRIX_BM(matrix_b))*STARNEIG_MATRIX_BM(matrix_b) -
                    STARNEIG_MATRIX_RBEGIN(matrix_b);

            starneig_insert_right_gemm_update(begin, end, it->begin, it->end, 0,
                prio, operator, matrix_b, mpi);
        }
    }

    //
    // insert the reorder_window task and give it the highest priority
    //

    starneig_reorder_insert_window(STARPU_MAX_PRIO,
        small_window_size, small_window_threshold, window, selected,
        matrix_a, matrix_b, mpi);
}

///
/// @brief Inserts all windows in a given window chain.
///
/// @param[in]     small_window_size       small window size
/// @param[in]     small_window_threshold  small window threshold
/// @param[in,out] selected                selection vector
/// @param[in,out] matrix_a                matrix A
/// @param[in,out] matrix_b                matrix B
/// @param[in,out] chain                   window chain
/// @param[in,out] mpi              MPI info
///
static void insert_window_chain(
    int small_window_size,
    int small_window_threshold,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct window_chain *chain,
    mpi_info_t mpi)
{
    // insert all windows in the window chain

    for (struct window *it = chain->bottom; it != NULL; it = it->up)
        insert_window(small_window_size, small_window_threshold, selected,
            matrix_a, matrix_b, it, chain, mpi);

    // In order to keep things more consistent, one additional right-hand update
    // is inserted at this point. The update corresponds to the topmost window
    // in the current window chain. The task must be inserted before the next
    // window chain gets inserted because otherwise the ordering of the
    // orthogonal transformations might get violated.

    // The update is not part of the critical path of current window chain but
    // it might be implicitly part of a critical path of one of the following
    // window chain (in some rare cases, the practice of "rounding" to the
    // nearest data tile may introduce new implicit dependencies between window
    // chains). Thus the task is given a high priority.

    starpu_data_handle_t operator;
    if (chain->top->lz_h != NULL)
        operator = chain->top->lz_h;
    else
        operator = chain->top->lq_h;

    {
        int begin = ((STARNEIG_MATRIX_RBEGIN(matrix_a) + chain->top->begin) /
            STARNEIG_MATRIX_BM(matrix_a))*STARNEIG_MATRIX_BM(matrix_a) -
            STARNEIG_MATRIX_RBEGIN(matrix_a);

        starneig_insert_right_gemm_update(
            begin, chain->top->begin, chain->top->begin, chain->top->end, 0,
            MAX(STARPU_DEFAULT_PRIO, STARPU_MAX_PRIO-1), operator, matrix_a,
            mpi);
    }

    if (matrix_b != NULL) {
        int begin = ((STARNEIG_MATRIX_RBEGIN(matrix_b) + chain->top->begin) /
            STARNEIG_MATRIX_BM(matrix_b))*STARNEIG_MATRIX_BM(matrix_b) -
            STARNEIG_MATRIX_RBEGIN(matrix_b);

        starneig_insert_right_gemm_update(
            begin, chain->top->begin, chain->top->begin, chain->top->end, 0,
            MAX(STARPU_DEFAULT_PRIO, STARPU_MAX_PRIO-1), operator, matrix_b,
            mpi);
    }
}

///
/// @brief Inserts all remaining right update tasks.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] longest
///         length of the longest window chain in the plan
///
/// @param[in] chain
///         window chain descriptor
///
/// @param[in,out] matrix
///         matrix A/B descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_right_updates(
    int height, int longest, struct window_chain const *chain,
    starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    // all right updates below this row have already been inserted
    int end = ((STARNEIG_MATRIX_RBEGIN(matrix) + chain->begin) /
        STARNEIG_MATRIX_BM(matrix)) * STARNEIG_MATRIX_BM(matrix) -
        STARNEIG_MATRIX_RBEGIN(matrix);

    // first group of right updates gets the priority (MAX-DEFAULT)/2-1
    int prio = MAX(STARPU_DEFAULT_PRIO,
        ((long)STARPU_MAX_PRIO-STARPU_DEFAULT_PRIO)/2 - 1);

    // go through all chains that are above the current chain
    for (struct window_chain *cit = chain->up; cit != NULL; cit = cit->up) {

        int begin = ((STARNEIG_MATRIX_RBEGIN(matrix) + cit->begin) /
            STARNEIG_MATRIX_BM(matrix)) * STARNEIG_MATRIX_BM(matrix) -
            STARNEIG_MATRIX_RBEGIN(matrix);

        // go though all windows in the current chain
        for (struct window *it = chain->bottom; it != NULL; it = it->up) {

            starpu_data_handle_t operator;
            if (it->lz_h != NULL)
                operator = it->lz_h;
            else
                operator = it->lq_h;

            // insert overlapping right-hand side updates
            starneig_insert_right_gemm_update(
                begin, end, it->begin, it->end, height, prio, operator, matrix,
                mpi);
        }

        end = begin;
        prio = MAX(STARPU_DEFAULT_PRIO, prio-1);
    }

    // insert remaining low priority right update tasks
    for (struct window *it = chain->bottom; it != NULL; it = it->up) {

        starpu_data_handle_t operator;
        if (it->lz_h != NULL)
            operator = it->lz_h;
        else
            operator = it->lq_h;

        prio = calc_tile_prio(it->idx, chain->effective_length, longest);
        starneig_insert_right_gemm_update(
            0, end, it->begin, it->end, height, prio, operator, matrix,
            mpi);
    }
}

///
/// @brief Inserts all remaining low priority right update tasks.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] longest
///         length of the longest window chain in the plan
///
/// @param[in] chain
///         window chain descriptor
///
/// @param[in,out] matrix
///         matrix A/B descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_low_prio_right_updates(
    int height, int longest, struct window_chain const *chain,
    starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    // all right-hand side updates below this row have already been inserted
    int end = ((STARNEIG_MATRIX_RBEGIN(matrix) + chain->begin) /
        STARNEIG_MATRIX_BM(matrix)) * STARNEIG_MATRIX_BM(matrix) -
        STARNEIG_MATRIX_RBEGIN(matrix);

    // inserts all remaining right-hand side updates

    for (struct window *it = chain->bottom; it != NULL; it = it->up) {

        starpu_data_handle_t operator;
        if (it->lz_h != NULL)
            operator = it->lz_h;
        else
            operator = it->lq_h;

        int prio = calc_tile_prio(it->idx, chain->effective_length, longest);
        starneig_insert_right_gemm_update(
            0, end, it->begin, it->end, height, prio, operator, matrix,
            mpi);
    }
}

///
/// @brief Inserts all remaining left update tasks.
///
/// @param[in] width
///         width of a single update tasks
///
/// @param[in] longest
///         length of the longest window chain in the plan
///
/// @param[in] chain
///         window chain descriptor
///
/// @param[in,out] matrix
///         matrix A/B descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_left_updates(
    int width, int longest, struct window_chain const *chain,
    starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    int n = STARNEIG_MATRIX_N(matrix);

    // first group of left update tasks gets the priority (MAX-DEFAULT)/2-1
    int prio = MAX(STARPU_DEFAULT_PRIO,
        ((long)STARPU_MAX_PRIO-STARPU_DEFAULT_PRIO)/2 - 1);

    int begin = ((STARNEIG_MATRIX_CBEGIN(matrix) + chain->begin) /
        STARNEIG_MATRIX_BN(matrix)) * STARNEIG_MATRIX_BN(matrix) -
        STARNEIG_MATRIX_CBEGIN(matrix);

    // go through all chains that follow the current chain
    for (struct window_chain *cit = chain->down; cit != NULL; cit = cit->down) {
        int end = MIN(n,
            divceil(STARNEIG_MATRIX_CBEGIN(matrix) + cit->end,
            STARNEIG_MATRIX_BN(matrix)) * STARNEIG_MATRIX_BN(matrix) -
            STARNEIG_MATRIX_CBEGIN(matrix));

        // go through all windows in the current chain and insert left-hand side
        // updates
        for (struct window *wit = chain->bottom; wit != NULL; wit = wit->up)
            starneig_insert_left_gemm_update(
                wit->begin, wit->end, MAX(begin, wit->end), end, width,
                prio, wit->lq_h, matrix, mpi);

        begin = end;
        prio = MAX(STARPU_DEFAULT_PRIO, prio-1);
    }

    // insert the remaining updates
    for (struct window *wit = chain->bottom; wit != NULL; wit = wit->up) {
        int prio = calc_tile_prio(
            wit->idx, chain->effective_length, longest);
        starneig_insert_left_gemm_update(
            wit->begin, wit->end, MAX(begin, wit->end), n, width, prio,
            wit->lq_h, matrix, mpi);
    }
}

///
/// @brief Inserts all Q matrix update tasks.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] longest
///         length of the longest window chain in the plan
///
/// @param[in] chain
///         window chain descriptor
///
/// @param[in,out] matrix
///         matrix Q descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_q_updates(
    int height, int longest, struct window_chain const *chain,
    starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    // go through all windows in the window chain
    for (struct window *it = chain->bottom; it != NULL; it = it->up) {
        int prio = calc_tile_prio(
            it->idx, chain->effective_length, longest);
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(matrix), it->begin, it->end, height, prio,
            it->lq_h, matrix, mpi);
    }
}

///
/// @brief Inserts all Z matrix update tasks.
///
/// @param[in] height
///         height of a single update tasks
///
/// @param[in] longest
///         length of the longest window chain in the plan
///
/// @param[in] chain
///         window chain descriptor
///
/// @param[in,out] matrix
///         matrix Z descriptor
///
/// @param[in,out] mpi
///         MPI info
///
static void insert_z_updates(
    int height, int longest, struct window_chain const *chain,
    starneig_matrix_descr_t matrix, mpi_info_t mpi)
{
    // go through all windows in the window chain
    for (struct window *it = chain->bottom; it != NULL; it = it->up) {
        int prio = calc_tile_prio(
            it->idx, chain->effective_length, longest);
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(matrix), it->begin, it->end, height, prio,
            it->lz_h, matrix, mpi);
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Processes a diagonal window.
///
/// @param[in] longest - length of the longest window chain in the plan
/// @param[in] conf - configuration structure
/// @param[in] steps - blueprint steps
/// @param[in,out] selected - eigenvalue selection bitmap descriptor
/// @param[in,out] matrix_q - matrix Q descriptor
/// @param[in,out] matrix_z - matrix Z descriptor
/// @param[in,out] matrix_a - matrix A descriptor
/// @param[in,out] matrix_b - matrix B descriptor
/// @param[in,out] window - window descriptor
/// @param[in,out] mpi
///
static void reorder_window(
    int longest,
    struct starneig_engine_conf_t const *conf,
    blueprint_step_t const *steps,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_q,
    starneig_matrix_descr_t matrix_z,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct window *window,
    mpi_info_t mpi)
{
    while (*steps != WINDOW_END) {
        switch (*steps) {
            case DUMMY_WINDOW:
                dummy_insert_window(
                    conf->small_window_size, conf->small_window_threshold,
                    selected, matrix_a, matrix_b, window, mpi);
                break;
            case DUMMY_LEFT_UPDATE:
                dummy_insert_left_update(
                    conf->a_width, window, matrix_a, mpi);
                if (matrix_b != NULL)
                    dummy_insert_left_update(
                        conf->b_width, window, matrix_b, mpi);
                break;
            case DUMMY_RIGHT_UPDATE:
                dummy_insert_right_update(
                    conf->a_height, window, matrix_a, mpi);
                if (matrix_b != NULL)
                    dummy_insert_right_update(
                        conf->b_height, window, matrix_b, mpi);
                break;
            case DUMMY_Q_UPDATE:
                if (matrix_q != NULL)
                    dummy_insert_q_update(
                        conf->q_height, window, matrix_q, mpi);
                if (matrix_z != NULL)
                    dummy_insert_z_update(
                        conf->z_height, window, matrix_z, mpi);
                break;
            case UNREGISTER:
                starneig_unregister_window(window);
                break;
            default:
                starneig_fatal_error("Invalid step.");
        }
        steps++;
    }
}

///
/// @brief Processes a window chain.
///
/// @param[in] longest - length of the longest window chain in the plan
/// @param[in] conf - configuration structure
/// @param[in] steps - blueprint steps
/// @param[in,out] selected - eigenvalue selection bitmap descriptor
/// @param[in,out] matrix_q - matrix Q descriptor
/// @param[in,out] matrix_z - matrix Z descriptor
/// @param[in,out] matrix_a - matrix A descriptor
/// @param[in,out] matrix_b - matrix B descriptor
/// @param[in,out] chain - window chain descriptor
/// @param[in,out] mpi
///
static void process_chain(
    int longest,
    struct starneig_engine_conf_t const *conf,
    blueprint_step_t const *steps,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_q,
    starneig_matrix_descr_t matrix_z,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct window_chain *chain,
    mpi_info_t mpi)
{
    while (*steps != CHAIN_END) {
        switch (*steps) {
            struct window *it;
            case WINDOW_BEGIN:
                it = chain->bottom;
                while (it != NULL) {
                    reorder_window(longest, conf, steps+1, selected, matrix_q,
                        matrix_z, matrix_a, matrix_b, it, mpi);
                    it = it->up;
                }
                while (*(++steps) != WINDOW_END);
                break;
            case WINDOWS:
                insert_window_chain(
                    conf->small_window_size, conf->small_window_threshold,
                    selected, matrix_a, matrix_b, chain, mpi);
                break;
            case LEFT_UPDATES:
                insert_left_updates(
                    conf->a_width, longest, chain, matrix_a, mpi);
                if (matrix_b != NULL)
                    insert_left_updates(
                        conf->b_width, longest, chain, matrix_b, mpi);
                break;
            case RIGHT_UPDATES:
                insert_right_updates(
                    conf->a_height, longest, chain, matrix_a, mpi);
                if (matrix_b != NULL)
                     insert_right_updates(
                        conf->b_height, longest, chain, matrix_b, mpi);
                break;
            case REMAINING_RIGHT_UPDATES:
                insert_low_prio_right_updates(
                    conf->a_height, longest, chain, matrix_a, mpi);
                if (matrix_b != NULL)
                    insert_low_prio_right_updates(
                        conf->b_height, longest, chain, matrix_b, mpi);
                break;
            case Q_UPDATES:
                if (matrix_q != NULL)
                    insert_q_updates(
                        conf->q_height, longest, chain, matrix_q, mpi);
                if (matrix_z != NULL)
                    insert_z_updates(
                        conf->z_height, longest, chain, matrix_z, mpi);
                break;
            case UNREGISTER:
                starneig_unregister_chain(chain);
                break;
            default:
                starneig_fatal_error("Invalid step.");
        }
        steps++;
    }
}

///
/// @brief Processes a chain list.
///
/// @param[in] longest - length of the longest window chain in the plan
/// @param[in] conf - configuration structure
/// @param[in] steps - blueprint steps
/// @param[in,out] selected - eigenvalue selection bitmap descriptor
/// @param[in,out] matrix_q - matrix Q descriptor
/// @param[in,out] matrix_z - matrix Z descriptor
/// @param[in,out] matrix_a - matrix A descriptor
/// @param[in,out] matrix_b - matrix B descriptor
/// @param[in,out] list - chain list descriptor
/// @param[in,out] mpi
///
static void process_chain_list(
    int longest,
    struct starneig_engine_conf_t const *conf,
    blueprint_step_t const *steps,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_q,
    starneig_matrix_descr_t matrix_z,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct chain_list *list,
    mpi_info_t mpi)
{
    while (*steps != CHAIN_LIST_END) {
        switch(*steps) {
            struct window_chain *it;
            case CHAIN_FORWARD:
                it = list->top;
                while (it != NULL) {
                    process_chain(longest, conf, steps+1, selected, matrix_q,
                        matrix_z, matrix_a, matrix_b, it, mpi);
                    it = it->down;
                }
                while (*(++steps) != CHAIN_END);
                break;
            case CHAIN_BACKWARD:
                it = list->bottom;
                while (it != NULL) {
                    process_chain(longest, conf, steps+1, selected, matrix_q,
                        matrix_z, matrix_a, matrix_b, it, mpi);
                    it = it->up;
                }
                while (*(++steps) != CHAIN_END);
                break;
            case UNREGISTER:
                starneig_unregister_chain_list(list);
                break;
            default:
                starneig_fatal_error("Invalid step.");
        }
        steps++;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void starneig_process_plan(
    struct starneig_engine_conf_t const *conf,
    blueprint_step_t const *steps,
    starneig_vector_descr_t selected,
    starneig_matrix_descr_t matrix_q,
    starneig_matrix_descr_t matrix_z,
    starneig_matrix_descr_t matrix_a,
    starneig_matrix_descr_t matrix_b,
    struct plan *plan,
    mpi_info_t mpi)
{
    while (*steps != END) {
        switch(*steps) {
            struct chain_list *it;
            case CHAIN_LIST_BEGIN:
                it = plan->begin;
                while (it != NULL) {
                    process_chain_list(plan->longest_eff_length, conf,
                        steps+1, selected, matrix_q, matrix_z, matrix_a,
                        matrix_b, it, mpi);
                    it = it->next;
                }
                while (*(++steps) != CHAIN_LIST_END);
                break;
            case UNREGISTER:
                starneig_unregister_plan(plan);
                break;
            default:
                starneig_fatal_error("Invalid step.");
        }
        steps++;
    }
}
