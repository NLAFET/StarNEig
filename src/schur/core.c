///
/// @file
///
/// @brief This file contains the main logic of the QR/QZ algorithm.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section DESCRIPTION
///
/// The implementation divides the Hessenberg form into active segments (usually
/// only one active segment at the beginning). The active segments are stored to
/// an active segment list. Each active segment has a state and the main thread
/// scans the active segment list and executes a ''state shift`` function for
/// each active segment. The state shift function may perform computational
/// operations (such as inserting tasks) on the active segment and/or change
/// segment's state. The state shift function may also divide the active segment
/// into sub-segments.
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
#include "core.h"
#include "process_args.h"
#include "segment.h"
#include "tasks.h"
#include "../common/common.h"
#include "../common/utils.h"
#include "../common/tasks.h"
#include "../common/trace.h"
#include "../hessenberg/internal/core.h"
#include <math.h>
#include <time.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#include <starpu_mpi.h>
#endif


///
/// @brief Scans a segment list and calls an appropriate state shift function
/// for each segment.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Error code.
///
static starneig_error_t scan_segment_list(
    struct segment_list *list, struct process_args *args);

///
/// @brief Calls an appropriate state shift function for a given segment.
///
/// @param[in,out] segment
///         segment
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment(
    struct segment *segment, struct process_args *args);

///
/// @brief Window chain direction hint for the insert_updates and the
/// insert_segment_updates functions.
///
enum update_direction_hint {
    UPDATE_DIRECTION_UP,    ///< window chain progresses upwards
    UPDATE_DIRECTION_DOWN,  ///< window chain progresses downwards
    UPDATE_DIRECTION_NONE   ///< no preferred direction
};

///
/// @brief Inserts update tasks that correspond to a given diagonal window.
///
/// @param[in] begin
///         First row/column that belongs to the diagonal window.
///
/// @param[in] end
///         Last row/column that belongs to the diagonal window + 1.
///
/// @param[in] lQ_h
///         Local left-hand size transformation matrix.
///
/// @param[in] lZ_h
///         Local right-hand size transformation matrix.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @param[in] direction
///         Window chain direction hint.
///
static void insert_updates(
    int begin, int end, starpu_data_handle_t lQ_h, starpu_data_handle_t lZ_h,
    struct process_args *args, enum update_direction_hint direction)
{
    if (lZ_h == NULL)
        lZ_h = lQ_h;

    int right_prio, left_prio;
    switch (direction) {
        case UPDATE_DIRECTION_UP:
            right_prio = MAX(args->default_prio, args->max_prio-1);
            left_prio = args->default_prio;
            break;
        case UPDATE_DIRECTION_DOWN:
            right_prio = args->default_prio;
            left_prio = MAX(args->default_prio, args->max_prio-1);
            break;
        default:
            right_prio = args->default_prio;
            left_prio = args->default_prio;
    }

    // update A

    starneig_insert_right_gemm_update(0, begin, begin, end, args->a_height,
        right_prio, lZ_h, args->matrix_a, args->mpi);

    starneig_insert_left_gemm_update(
        begin, end, end, STARNEIG_MATRIX_N(args->matrix_a), args->a_width,
        left_prio, lQ_h, args->matrix_a, args->mpi);

    // update B

    if (args->matrix_b != NULL) {
        starneig_insert_right_gemm_update(0, begin, begin, end, args->b_height,
            right_prio, lZ_h, args->matrix_b, args->mpi);

        starneig_insert_left_gemm_update(
            begin, end, end, STARNEIG_MATRIX_N(args->matrix_b), args->b_width,
            left_prio, lQ_h, args->matrix_b, args->mpi);
    }

    // update Q

    if (args->matrix_q != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_q), begin, end, args->q_height,
            args->min_prio, lQ_h, args->matrix_q, args->mpi);

    // update Z

    if (args->matrix_z != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_z), begin, end, args->z_height,
            args->min_prio, lZ_h, args->matrix_z, args->mpi);
}

///
/// @brief Inserts update tasks that correspond to a given diagonal window. The
/// segment size is taken into account when assigning priorities. Updates
/// that fall outside the segment are given lower priority.
///
/// @param[in] begin
///         First row/column that belongs to the diagonal window.
///
/// @param[in] end
///         Last row/column that belongs to the diagonal window + 1.
///
/// @param[in] lQ_h
///         Local left-hand size transformation matrix.
///
/// @param[in] lZ_h
///         Local right-hand size transformation matrix.
///
/// @param[in] segment
///         Segment.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @param[in] direction
///         Window chain direction hint.
///
static void insert_segment_updates(
    int begin, int end, starpu_data_handle_t lQ_h, starpu_data_handle_t lZ_h,
    struct segment const *segment, struct process_args *args,
    enum update_direction_hint direction)
{
    if (lZ_h == NULL)
        lZ_h = lQ_h;

    int off_prio = MAX(args->min_prio, args->default_prio-1);
    int right_prio, left_prio;
    switch (direction) {
        case UPDATE_DIRECTION_UP:
            right_prio = MAX(args->default_prio, args->max_prio-1);
            left_prio = args->default_prio;
            break;
        case UPDATE_DIRECTION_DOWN:
            right_prio = args->default_prio;
            left_prio = MAX(args->default_prio, args->max_prio-1);
            break;
        default:
            right_prio = args->default_prio;
            left_prio = args->default_prio;
    }

    #define update_matrix(matrix_x, x_height, x_width) { \
        int vert_cut = starneig_matrix_cut_vectically_up( \
            segment->begin, matrix_x); \
        int hor_cut = starneig_matrix_cut_horizontally_right( \
            segment->end, matrix_x); \
        \
        starneig_insert_right_gemm_update( \
            0, vert_cut, begin, end, x_height, off_prio, lZ_h, \
            matrix_x, args->mpi); \
        starneig_insert_right_gemm_update( \
            vert_cut, begin, begin, end, x_height, right_prio, lZ_h, \
            matrix_x, args->mpi); \
        \
        starneig_insert_left_gemm_update( \
            begin, end, end, hor_cut, x_width, left_prio, \
            lQ_h, matrix_x, args->mpi); \
        starneig_insert_left_gemm_update( \
            begin, end, hor_cut, STARNEIG_MATRIX_N(matrix_x), x_width,  \
            off_prio, lQ_h, matrix_x, args->mpi); \
    }

    // update A

    update_matrix(args->matrix_a, args->a_height, args->a_width);

    // update B

    if (args->matrix_b != NULL)
        update_matrix(args->matrix_b, args->b_height, args->b_width);

    #undef update_matrix

    // update Q

    if (args->matrix_q != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_q), begin, end, args->q_height,
            args->min_prio, lQ_h, args->matrix_q, args->mpi);

    // update Z

    if (args->matrix_z != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_z), begin, end, args->z_height,
            args->min_prio, lZ_h, args->matrix_z, args->mpi);
}

///
/// @brief Inserts update tasks that correspond to a given diagonal window. The
/// segment size is taken into account when assigning priorities. Updates
/// that fall outside the segment are given lower priority.
///
/// @param[in] begin
///         First row/column that belongs to the diagonal window.
///
/// @param[in] end
///         Last row/column that belongs to the diagonal window + 1.
///
/// @param[in] top
///         location of the top left corner of the last bulge chasing window
///
/// @param[in] lQ_h
///         Local left-hand size transformation matrix.
///
/// @param[in] lZ_h
///         Local right-hand size transformation matrix.
///
/// @param[in] segment
///         Segment.
///
/// @param[in,out] args
///         Segment processing arguments.
///
static void insert_reverse_updates(
    int begin, int end, int top, starpu_data_handle_t lQ_h,
    starpu_data_handle_t lZ_h, struct segment const *segment,
    struct process_args *args)
{
    if (lZ_h == NULL)
        lZ_h = lQ_h;

    int low_prio = MAX(args->min_prio, args->default_prio-1);
    int medium_prio = args->default_prio;
    int high_prio = MAX(args->default_prio, args->max_prio-1);
    int max_prio = args->max_prio;

    #define update_matrix(matrix_x, x_height, x_width) { \
        int vert_cut = \
            starneig_matrix_cut_vectically_up(segment->begin, matrix_x); \
        int top_cut = \
            starneig_matrix_cut_vectically_up(top, matrix_x); \
        int hor_cut = \
            starneig_matrix_cut_horizontally_right(segment->end, matrix_x); \
        \
        starneig_insert_right_gemm_update( \
            top_cut, begin, begin, end, STARNEIG_MATRIX_BM(matrix_x), \
            max_prio, lZ_h, matrix_x, args->mpi); \
        \
        starneig_insert_left_gemm_update( \
            begin, end, end, hor_cut, x_width, \
            high_prio, lQ_h, matrix_x, args->mpi); \
        starneig_insert_right_gemm_update( \
            vert_cut, top_cut, begin, end, x_height, \
            medium_prio, lZ_h, matrix_x, args->mpi); \
        \
        starneig_insert_left_gemm_update( \
            begin, end, hor_cut, STARNEIG_MATRIX_N(matrix_x), x_width, \
            low_prio, lQ_h, matrix_x, args->mpi); \
        starneig_insert_right_gemm_update(0, vert_cut, begin, end, x_height, \
            low_prio, lZ_h, matrix_x, args->mpi); \
    }

    // update A

    update_matrix(args->matrix_a, args->a_height, args->a_width);

    // update B

    if (args->matrix_b != NULL)
        update_matrix(args->matrix_b, args->b_height, args->b_width);

    #undef update_matrix

    // update Q

    if (args->matrix_q != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_q), begin, end, args->q_height,
            args->min_prio, lQ_h, args->matrix_q, args->mpi);

    // update Z

    if (args->matrix_z != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_z), begin, end, args->z_height,
            args->min_prio, lZ_h, args->matrix_z, args->mpi);
}

///
/// @brief Inserts update tasks that correspond to a given AED window. The
/// segment size is taken into account when assigning priorities. Updates
/// that fall outside the segment are given lower priority.
///
/// @param[in] begin
///         First row/column that belongs to the diagonal window.
///
/// @param[in] end
///         Last row/column that belongs to the diagonal window + 1.
///
/// @param[in] lQ_h
///         Local left-hand size transformation matrix.
///
/// @param[in] lZ_h
///         Local right-hand size transformation matrix.
///
/// @param[in] segment
///         Segment.
///
/// @param[in,out] args
///         Segment processing arguments.
///
static void insert_aed_updates(
    int begin, int end, starpu_data_handle_t lQ_h, starpu_data_handle_t lZ_h,
    struct segment const *segment, struct process_args *args)
{
    if (lZ_h == NULL)
        lZ_h = lQ_h;

    int off_prio = MAX(args->min_prio, args->default_prio-1);
    int right_prio = MAX(args->default_prio, args->max_prio-1);

    int aed_window_size = evaluate_parameter(
        segment->end - segment->begin, args->aed_window_size);

    #define update_matrix(matrix_x) { \
        int x_height = STARNEIG_MATRIX_BM(matrix_x); \
        int x_width = STARNEIG_MATRIX_BN(matrix_x); \
        \
        int off_cut = starneig_matrix_cut_vectically_up( \
            segment->begin, matrix_x); \
        int aed_cut = starneig_matrix_cut_vectically_up( \
            MAX(segment->begin, begin-aed_window_size), matrix_x); \
        \
        starneig_insert_right_gemm_update( \
            0, off_cut, begin, end, x_height, off_prio, lZ_h, \
            matrix_x, args->mpi); \
        starneig_insert_right_gemm_update( \
            off_cut, aed_cut, begin, end, x_height, right_prio, lZ_h, \
            matrix_x, args->mpi); \
        starneig_insert_right_gemm_update( \
            aed_cut, begin, begin, end, x_height, args->max_prio, lZ_h, \
            matrix_x, args->mpi); \
        \
        starneig_insert_left_gemm_update( \
            begin, end, end, STARNEIG_MATRIX_N(matrix_x), x_width,  \
            off_prio, lQ_h, matrix_x, args->mpi); \
    }

    // update A

    update_matrix(args->matrix_a);

    // update B

    if (args->matrix_b != NULL)
        update_matrix(args->matrix_b);

    #undef update_matrix

    // update Q

    if (args->matrix_q != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_q), begin, end,
            STARNEIG_MATRIX_BM(args->matrix_q), args->min_prio, lQ_h,
            args->matrix_q, args->mpi);

    // update Z

    if (args->matrix_z != NULL)
        starneig_insert_right_gemm_update(
            0, STARNEIG_MATRIX_M(args->matrix_z), begin, end,
            STARNEIG_MATRIX_BM(args->matrix_z), args->min_prio, lZ_h,
            args->matrix_z, args->mpi);
}

///
/// @brief Inserts tasks that push infinite eigenvalues to the top of the
/// segment and deflate them.
///
/// @param[in] infs
///         An array that has all infinite eigenvalues marked in it.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
static void insert_push_inf_top(
    int *infs, struct segment *segment, struct process_args *args)
{
    int tile_size = STARNEIG_MATRIX_BN(args->matrix_a);

    int begin = segment->begin;
    int end = segment->begin;

    // repeat until all chains have been processed
    while (1) {

        // place the window chain
        int in_chain = 0;
        for (int i = end; in_chain+3 < tile_size && i < segment->end; i++) {
            if (infs[i-segment->begin]) {
                in_chain += 2;
                end = MIN(segment->end, i+2);
                i++;
            }
        }

        // quit if the chain is empty
        if (in_chain == 0)
            break;

        int wbegin = MAX(begin,
            starneig_matrix_cut_vectically_up(end, args->matrix_a) -
            STARNEIG_MATRIX_BM(args->matrix_a));
        int wend = end;

        int in_window = 0;
        for (int i = wbegin; i < wend; i++) {
            if (infs[i-segment->begin]) {
                in_window += 2;
                i++;
            }
        }
        memset(
            &infs[wbegin-segment->begin], 0, (wend-wbegin)*sizeof(infs[0]));

        while(1) {

            starpu_data_handle_t lQ_h, lZ_h;
            starneig_schur_insert_push_inf_top(
                wbegin, wend, wbegin == begin, wend == segment->end,
                args->max_prio, args->thres_inf, args->matrix_a, args->matrix_b,
                &lQ_h, &lZ_h, args->mpi);

            insert_segment_updates(
                wbegin, wend, lQ_h, lZ_h, segment, args, UPDATE_DIRECTION_UP);

            starpu_data_unregister_submit(lQ_h);
            starpu_data_unregister_submit(lZ_h);

            // quit if this was the topmost window in the chain
            if (wbegin == begin)
                break;

            for (int i = wbegin; i < wend-1; i++) {
                if (infs[i-segment->begin]) {
                    in_window += 2;
                    i++;
                }
            }
            memset(
                &infs[wbegin-segment->begin], 0, (wend-wbegin)*sizeof(infs[0]));

            wend = MIN(end, wbegin + in_window + 1);
            wbegin = MAX(begin, wbegin-STARNEIG_MATRIX_BM(args->matrix_a));
        }

        // advance downwards
        begin += in_chain/2;
    }

    // zero sub-diagonal entries may mess things up...
    // segment->begin = begin;
}

///
/// @brief Inserts bulge chasing tasks using a fixed window size.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
static void insert_bulges_fixed(
    struct segment *segment, struct process_args *args)
{
    // initialize aftermath vector
    if (segment->bulges_aftermath == NULL)
        segment->bulges_aftermath = starneig_init_matching_vector_descr(
            args->matrix_a, sizeof(bulge_chasing_aftermath_t), NULL, args->mpi);

    int aed_window_size = evaluate_parameter(
        segment->end - segment->begin, args->aed_window_size);
    int requested_shifts = evaluate_parameter(
        segment->end - segment->begin, args->shift_count);
    int window_size = evaluate_parameter(
        segment->end - segment->begin, args->bulges_window_size);
    int shifts_per_window = evaluate_parameter(
        segment->end - segment->begin, args->bulges_shifts_per_window);

    int total_shifts = (MIN(segment->computed_shifts, requested_shifts)/2)*2;
    int total_chains = divceil(total_shifts, shifts_per_window);
    int jump = 3*(shifts_per_window/2)+1;

    int top = - (total_chains-1) * window_size;

    while (top < segment->end) {
        int i = (total_chains-1)*shifts_per_window;
        for (int j = 0; j < total_chains; j++) {
            int shifts = MIN(shifts_per_window, total_shifts - i);

            int begin = top + 2*j*jump;
            int middle = top + (2*j+1)*jump;
            int end = top + (2*j+2)*jump;

            if (segment->begin < begin || end < segment->end) {
                if (middle <= segment->begin || segment->end < middle) {
                    i -= shifts_per_window;
                    continue;
                }
            }

            int wbegin;
            if (begin <= segment->begin)
                wbegin = segment->begin;
            else
                wbegin = middle - 3*(shifts/2) - 1;
            int wend = MIN(segment->end, end);

            // infer the bulge chasing mode
            bulge_chasing_mode_t mode;
            if (wbegin == segment->begin && wend == segment->end)
                mode = BULGE_CHASING_MODE_FULL;
            else if (wbegin == segment->begin)
                mode = BULGE_CHASING_MODE_INTRODUCE;
            else if (wend == segment->end)
                mode = BULGE_CHASING_MODE_FINALIZE;
            else
                mode = BULGE_CHASING_MODE_CHASE;

            // insert the window and the related updates

            starpu_data_handle_t lQ_h, lZ_h;
            starneig_schur_insert_push_bulges(
                wbegin, wend, i, i+shifts, mode, args->max_prio,
                args->thres_a, args->thres_b, args->thres_inf,
                segment->shifts_real, segment->shifts_imag,
                j == 0 ? segment->bulges_aftermath : NULL,
                args->matrix_a, args->matrix_b, &lQ_h, &lZ_h,
                args->mpi);

            insert_reverse_updates(wbegin, wend,
                MAX(segment->begin,  MIN(segment->end - aed_window_size, top)),
                lQ_h, lZ_h, segment, args);

            if (lZ_h != NULL && lZ_h != lQ_h)
                starpu_data_unregister_submit(lZ_h);
            starpu_data_unregister_submit(lQ_h);

            i -= shifts_per_window;
        }
        top += jump;
    }

    segment->peak_submitted = starpu_task_nsubmitted();
    segment->peak_time = starpu_timing_now();
    segment->slope = NAN;

#ifdef STARNEIG_ENABLE_MPI
    // gather the aftermath vector to all MPI nodes
    if (args->mpi != NULL) {
        int world_size = starneig_mpi_get_comm_size();
        for (int i = 0; i < world_size; i++)
            starneig_gather_segment_vector_descr(
                i, segment->begin, segment->end, segment->bulges_aftermath);
    }
#endif
}

///
/// @brief Inserts bulge chasing tasks using "rounded" window size.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
static void insert_bulges_rounded(
    struct segment *segment, struct process_args *args)
{
    // initialize aftermath vector
    if (segment->bulges_aftermath == NULL)
        segment->bulges_aftermath = starneig_init_matching_vector_descr(
            args->matrix_a, sizeof(bulge_chasing_aftermath_t), NULL, args->mpi);

    int aed_window_size = evaluate_parameter(
        segment->end - segment->begin, args->aed_window_size);
    int requested_shifts = evaluate_parameter(
            segment->end - segment->begin, args->shift_count);
    int shifts_per_window = MIN(
        evaluate_parameter(
                segment->end - segment->begin, args->bulges_shifts_per_window),
        2*((STARNEIG_MATRIX_BM(args->matrix_a)-1)/3));
    shifts_per_window = (shifts_per_window/2)*2;

    int total_shifts = (MIN(segment->computed_shifts, requested_shifts)/2)*2;
    int total_chains = divceil(total_shifts, shifts_per_window);

    int top = starneig_matrix_cut_vectically_up(
        segment->begin, args->matrix_a) +
        (2 - 2*total_chains)*STARNEIG_MATRIX_BM(args->matrix_a);

    while (top < segment->end) {
        int i = (total_chains-1)*shifts_per_window;
        for (int j = 0; j < total_chains; j++) {
            int shifts = MIN(shifts_per_window, total_shifts - i);

            int begin = top + 2*j*STARNEIG_MATRIX_BM(args->matrix_a);
            int middle = top + (2*j+1)*STARNEIG_MATRIX_BM(args->matrix_a);
            int end = top + (2*j+2)*STARNEIG_MATRIX_BM(args->matrix_a);

            if (segment->begin < begin || end < segment->end) {
                if (middle <= segment->begin || segment->end < middle) {
                    i -= shifts_per_window;
                    continue;
                }
            }

            int wbegin;
            if (begin <= segment->begin)
                wbegin = segment->begin;
            else
                wbegin = middle - 3*(shifts/2) - 1;
            int wend = MIN(segment->end, end);

            // infer the bulge chasing mode
            bulge_chasing_mode_t mode;
            if (wbegin == segment->begin && wend == segment->end)
                mode = BULGE_CHASING_MODE_FULL;
            else if (wbegin == segment->begin)
                mode = BULGE_CHASING_MODE_INTRODUCE;
            else if (wend == segment->end)
                mode = BULGE_CHASING_MODE_FINALIZE;
            else
                mode = BULGE_CHASING_MODE_CHASE;

            // insert the window and the related updates

            starpu_data_handle_t lQ_h, lZ_h;
            starneig_schur_insert_push_bulges(
                wbegin, wend, i, i+shifts, mode, args->max_prio,
                args->thres_a, args->thres_b, args->thres_inf,
                segment->shifts_real, segment->shifts_imag,
                j == 0 ? segment->bulges_aftermath : NULL,
                args->matrix_a, args->matrix_b, &lQ_h, &lZ_h,
                args->mpi);

            insert_reverse_updates(wbegin, wend,
                MAX(segment->begin, MIN(segment->end - aed_window_size, top)),
                lQ_h, lZ_h, segment, args);

            if (lZ_h != NULL && lZ_h != lQ_h)
                starpu_data_unregister_submit(lZ_h);
            starpu_data_unregister_submit(lQ_h);

            i -= shifts_per_window;
        }
        top += STARNEIG_MATRIX_BM(args->matrix_a);
    }

    segment->peak_submitted = starpu_task_nsubmitted();
    segment->peak_time = starpu_timing_now();
    segment->slope = NAN;

#ifdef STARNEIG_ENABLE_MPI
    // gather the aftermath vector to all MPI nodes
    if (args->mpi != NULL) {
        int world_size = starneig_mpi_get_comm_size();
        for (int i = 0; i < world_size; i++)
            starneig_gather_segment_vector_descr(
                i, segment->begin, segment->end, segment->bulges_aftermath);
    }
#endif
}

///
/// @brief Performs deflation process finalization.
///
///  Outcomes:
///   - If the AED process generated enough shifts => bulge chasing tasks are
///     inserted and the matching aftermath vector tiles are gathered to
///     all MPI nodes. The segment is marked as SEGMENT_BULGES.
///   - Otherwise, the segment is marked as SEGMENT_NEW.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_deflate_finalize(
    struct segment *segment, struct process_args *args)
{
    int my_rank = starneig_mpi_get_comm_rank();
    int owner = starneig_get_tile_owner_matrix_descr(
        0, 0, segment->aed_args.matrix_a);

    //
    // acquire and check the AED status
    //

#ifdef STARNEIG_ENABLE_MPI
    if (args->mpi != NULL)
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), segment->aed_status_h);
#endif

    starpu_data_acquire(segment->aed_status_h, STARPU_R);
    struct aed_status *status = (struct aed_status *)
        starpu_variable_get_local_ptr(segment->aed_status_h);

    if (status->status != AED_STATUS_SUCCESS) {
        starneig_verbose("Parallel AED has failed.");
        segment->status = SEGMENT_FAILURE;
        goto cleanup;
    }

    // the AED window is badded from the top left corner
    int padded_size = segment->end - segment->aed_begin;

    int requested_shifts = MIN(0.30*padded_size, evaluate_parameter(
        segment->end - segment->begin, args->shift_count));

    int nibble = evaluate_parameter(
        segment->end - segment->begin, args->aed_nibble);

    int perform_bulge_chasing =
        status->converged == 0 || (
            requested_shifts/2 <= status->computed_shifts &&
            status->converged < 0.01*nibble*(padded_size-1)
        );

    //
    // extract shifts from the matrix pencil
    //

    if (perform_bulge_chasing && my_rank == owner)
        starneig_schur_insert_extract_shifts(
            1, status->computed_shifts+1, args->max_prio,
            segment->aed_args.matrix_a, segment->aed_args.matrix_b,
            segment->shifts_real, segment->shifts_imag, NULL);

    //
    // if the AED process managed to deflate eigenvalues, ...
    //
    if (0 < status->converged) {

        starpu_data_handle_t lQ;
        starpu_matrix_data_register(
            &lQ, -1, 0, padded_size, padded_size, padded_size, sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
        if (args->mpi)
            starpu_mpi_data_register_comm(
                lQ, args->mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif

        starpu_data_handle_t lZ = lQ;
        if (args->matrix_b != NULL) {
            starpu_matrix_data_register(
                &lZ, -1, 0, padded_size, padded_size, padded_size,
                sizeof(double));
#ifdef STARNEIG_ENABLE_MPI
            if (args->mpi)
                starpu_mpi_data_register_comm(
                    lZ, args->mpi->tag_offset++, owner,
                    starneig_mpi_get_comm());
#endif
        }

        if (my_rank == owner) {

            //
            // embed the spike to the padded AED window
            //

            starneig_schur_insert_embed_spike(
                padded_size-status->converged, args->max_prio,
                segment->aed_deflate_base, segment->aed_args.matrix_a);

            //
            // reduce the non-deleflated part to Hessenberg-triangular form
            //

            int hessenberg_prio_high, hessenberg_prio_normal,
                hessenberg_prio_low;
            if (perform_bulge_chasing) {
                hessenberg_prio_high =
                    MAX(args->default_prio, args->max_prio-2);
                hessenberg_prio_normal =
                    MAX(args->default_prio, args->max_prio-3);
                hessenberg_prio_low =
                    MAX(args->default_prio, args->max_prio-4);
            }
            else {
                hessenberg_prio_high = args->max_prio;
                hessenberg_prio_normal =
                    MAX(args->default_prio, args->max_prio-1);
                hessenberg_prio_low =
                    MAX(args->default_prio, args->max_prio-1);
            }

            if (padded_size-status->converged < 1000 || args->matrix_b != NULL)
            {

                starneig_verbose(
                    "Performing a sequential Hessenberg reduction.");

                starpu_data_handle_t _lQ, _lZ;
                starneig_schur_insert_small_hessenberg(
                    0, padded_size-status->converged, hessenberg_prio_high,
                    segment->aed_args.matrix_a, segment->aed_args.matrix_b,
                    &_lQ, &_lZ, NULL);

                insert_updates(
                    0, padded_size-status->converged, _lQ, _lZ,
                    &segment->aed_args, UPDATE_DIRECTION_NONE);

                if (_lZ != NULL && _lZ != _lQ)
                    starpu_data_unregister_submit(_lZ);
                starpu_data_unregister_submit(_lQ);

            }
            else {

                starneig_verbose(
                    "Performing a parallel Hessenberg reduction.");

                int panel_width = divceil(224,
                    STARNEIG_MATRIX_BN(segment->aed_args.matrix_a)) *
                        STARNEIG_MATRIX_BN(segment->aed_args.matrix_a);

                starneig_hessenberg_insert_tasks(
                    panel_width, 0, padded_size-status->converged,
                    hessenberg_prio_high, hessenberg_prio_normal,
                    hessenberg_prio_low, segment->aed_args.matrix_q,
                    segment->aed_args.matrix_a, false, NULL);
            }

            //
            // copy the local transformation matrices to separete data handles
            //

            starneig_insert_copy_matrix_to_handle(
                0, STARNEIG_MATRIX_M(segment->aed_args.matrix_q),
                0, STARNEIG_MATRIX_N(segment->aed_args.matrix_q),
                args->max_prio, segment->aed_args.matrix_q, lQ, NULL);

            if (lZ != lQ)
                starneig_insert_copy_matrix_to_handle(
                    0, STARNEIG_MATRIX_M(segment->aed_args.matrix_z),
                    0, STARNEIG_MATRIX_N(segment->aed_args.matrix_z),
                    args->max_prio, segment->aed_args.matrix_z, lZ, NULL);
        }

        starneig_verbose("Deflated %d eigenvalues.", status->converged);

        //
        // embed the AED window back to the matrix pencil
        //

        starneig_insert_copy_matrix(
            0, 0, segment->aed_begin, segment->aed_begin,
            padded_size, padded_size, args->max_prio,
            segment->aed_args.matrix_a, args->matrix_a, args->mpi);

        if (args->matrix_b != NULL)
            starneig_insert_copy_matrix(
                0, 0, segment->aed_begin, segment->aed_begin,
                padded_size, padded_size, args->max_prio,
                segment->aed_args.matrix_b, args->matrix_b, args->mpi);

        //
        // insert update tasks
        //

        insert_aed_updates(
            segment->aed_begin, segment->end, lQ, lZ, segment, args);

        if (lZ != lQ)
            starpu_data_unregister_submit(lZ);
        starpu_data_unregister_submit(lQ);

        //
        // resize the segment
        //

        segment->end -= status->converged;
        segment->aed_failed = 0;
    }
    else {
        segment->aed_failed++;
    }

    if (perform_bulge_chasing) {

        //
        // insert bulge chasing tasks and mark the segment as SEGMENT_BULGES
        //

        segment->computed_shifts = status->computed_shifts;
        segment->status = SEGMENT_BULGES;
        if (args->bulges_window_placement == BULGES_WINDOW_PLACEMENT_FIXED)
            insert_bulges_fixed(segment, args);
        else
            insert_bulges_rounded(segment, args);
    }
    //
    // otherwise, ...
    //
    else {

        //
        // mark the segment as SEGMENT_NEW to begin a new iteration
        //

        segment->computed_shifts = 0;
        segment->status = SEGMENT_NEW;
    }

    //
    // cleanup
    //

cleanup:

    starpu_data_release(segment->aed_status_h);
    starpu_data_unregister_submit(segment->aed_status_h);
    segment->aed_status_h = NULL;

    starneig_free_matrix_descr(segment->aed_args.matrix_a);
    segment->aed_args.matrix_a = NULL;

    starneig_free_matrix_descr(segment->aed_args.matrix_b);
    segment->aed_args.matrix_b = NULL;

    starneig_free_matrix_descr(segment->aed_args.matrix_q);
    segment->aed_args.matrix_q = NULL;

    starneig_free_matrix_descr(segment->aed_args.matrix_z);
    segment->aed_args.matrix_z = NULL;

    if (segment->aed_deflate_status_h != NULL)
        starpu_data_unregister_submit(segment->aed_deflate_status_h);
    segment->aed_deflate_status_h = NULL;

    if (segment->aed_deflate_inducer_h != NULL)
        starpu_data_unregister_submit(segment->aed_deflate_inducer_h);
    segment->aed_deflate_inducer_h = NULL;

    starneig_free_vector_descr(segment->aed_deflate_base);
    segment->aed_deflate_base = NULL;

    starneig_free_vector_descr(segment->shifts_real);
    segment->shifts_real = NULL;

    starneig_free_vector_descr(segment->shifts_imag);
    segment->shifts_imag = NULL;

    return segment->status;
}

///
/// @brief Performs a deflation step.
///
///  Outcomes:
///   - If the deflation process has not finished => the segment retains the
///     SEGMENT_AED_DEFLATE status.
///   - Otherwise, the perform_deflate_finalize function is called.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_deflate_step(
    struct segment *segment, struct process_args *args)
{
    segment->status = SEGMENT_AED_DEFLATE;

    int tile_size = STARNEIG_MATRIX_BM(segment->aed_args.matrix_a);

    // the AED window is badded from the top left corner with one row/column
    int padded_size = segment->end - segment->aed_begin;

    // if the deflation process has not begun, ...
    if (segment->aed_deflate_status_h == NULL) {

        starneig_verbose("Beginning deflation checks.");

        // initialize deflation tracking variables
        segment->aed_deflate_top = 1;
        segment->aed_deflate_bottom =
            STARNEIG_MATRIX_M(segment->aed_args.matrix_a);

        // extract the spike inducer
        segment->aed_deflate_inducer_h = starneig_get_elem_from_matrix_descr(
            1, 0, segment->aed_args.matrix_a, NULL);

        // extract the spike base
        starneig_schur_insert_form_spike(
            args->max_prio, segment->aed_args.matrix_q,
            &segment->aed_deflate_base);
    }
    // otherwise a deflation task has already been inserted, ...
    else {
        // acquire the outcome of the previous deflate task
        starpu_data_acquire(segment->aed_deflate_status_h, STARPU_R);
        struct deflate_status deflate_status = *((struct deflate_status *)
            starpu_variable_get_local_ptr(segment->aed_deflate_status_h));
        starpu_data_release(segment->aed_deflate_status_h);

        // record progress
        segment->aed_deflate_bottom = deflate_status.end;
        if (deflate_status.begin <= segment->aed_deflate_top)
            segment->aed_deflate_top = deflate_status.end;

        int undeflated = deflate_status.end - deflate_status.begin;

        // if the undeflated blocks are not at their final locations and there
        // are many of them, ...
        if (segment->aed_deflate_top < deflate_status.begin &&
        tile_size <= undeflated) {

            // process the undeflated blocks in batches
            int jump = MIN(tile_size-2, undeflated/2+1);
            for (int i = 0; i < undeflated; i += jump) {

                int batch_size = MIN(jump, undeflated - i);

                // Calculate where the first reordering window should be placed.
                // The window is padded from all sides with one row/column.
                int end = MIN(padded_size, MIN(deflate_status.end+1,
                    deflate_status.begin + i+jump + 1));
                int begin = MAX(1, MAX(segment->aed_deflate_top-1,
                    ((end-1)/tile_size - 1) * tile_size));

                // create and initialize status tracking structure

                starpu_data_handle_t new_status_h;
                starpu_variable_data_register(
                    &new_status_h, -1, 0, sizeof(struct deflate_status));

                starpu_data_acquire(new_status_h, STARPU_W);
                struct deflate_status *new_status =
                    (struct deflate_status *)
                    starpu_variable_get_local_ptr(new_status_h);
                new_status->inherited = 1;
                new_status->begin = deflate_status.begin + i;
                new_status->end =
                    MIN(deflate_status.end, deflate_status.begin + i+jump);
                starpu_data_release(new_status_h);

                // move the undeflated blocks
                for (;;) {

                    // insert reordering window
                    starpu_data_handle_t lQ_h, lZ_h;
                    starneig_schur_insert_deflate(begin, end, 0, args->max_prio,
                        args->thres_a, segment->aed_deflate_inducer_h,
                        new_status_h, segment->aed_deflate_base,
                        segment->aed_args.matrix_a, segment->aed_args.matrix_b,
                        &lQ_h, &lZ_h);

                    // insert related updates
                    insert_updates(begin, end, lQ_h, lZ_h, &segment->aed_args,
                        UPDATE_DIRECTION_UP);

                    if (lZ_h != NULL && lZ_h != lQ_h)
                        starpu_data_unregister_submit(lZ_h);
                    starpu_data_unregister_submit(lQ_h);

                    // last/topmost reordering window?
                    if (begin <= segment->aed_deflate_top)
                        break;

                    // place the next window (remember padding)
                    end = begin + batch_size + 2;
                    begin = MAX(1, MAX(segment->aed_deflate_top-1,
                        ((end-1)/tile_size - 1) * tile_size));
                }

                starpu_data_unregister_submit(new_status_h);

                segment->aed_deflate_top += batch_size;
            }

            // the next deflation window does not contain any undeflated
            // blocks and starts with a fresh status tracking structure
            starpu_data_unregister_submit(segment->aed_deflate_status_h);
            segment->aed_deflate_status_h = NULL;
        }
    }

    // if there are unchecked blocks left, ...
    if (segment->aed_deflate_top < segment->aed_deflate_bottom) {

        // Place the deflation window. The deflation windows are padded only
        // from the top left corner.
        int end = segment->aed_deflate_bottom;
        int begin = MAX(1, MAX(segment->aed_deflate_top-1,
            ((end-1) / tile_size - 1) * tile_size));

        // if the status tracking structure does not exists, create one
        if (segment->aed_deflate_status_h == NULL) {
            starpu_variable_data_register(
                &segment->aed_deflate_status_h, -1, 0,
                sizeof(struct deflate_status));
            starpu_data_acquire(
                segment->aed_deflate_status_h, STARPU_W);
            struct deflate_status *status = (struct deflate_status *)
                starpu_variable_get_local_ptr(
                    segment->aed_deflate_status_h);
            status->inherited = 0;
            status->begin = 0;
            status->end = 0;
            starpu_data_release(segment->aed_deflate_status_h);
        }

        // insert deflation window
        starpu_data_handle_t lQ_h, lZ_h;
        starneig_schur_insert_deflate(begin, end, 1, args->max_prio,
            args->thres_a, segment->aed_deflate_inducer_h,
            segment->aed_deflate_status_h, segment->aed_deflate_base,
            segment->aed_args.matrix_a, segment->aed_args.matrix_b,
            &lQ_h, &lZ_h);

        // insert updates
        insert_updates(begin, end, lQ_h, lZ_h, &segment->aed_args,
            UPDATE_DIRECTION_UP);

        if (lZ_h != NULL && lZ_h != lQ_h)
            starpu_data_unregister_submit(lZ_h);
        starpu_data_unregister_submit(lQ_h);
    }
    // otherwise all blocks have been checked, ...
    else {

        starneig_verbose("Finished deflation checks.");

        // update AED status

        starpu_data_acquire(segment->aed_status_h, STARPU_RW);
        struct aed_status *status = (struct aed_status *)
            starpu_variable_get_local_ptr(segment->aed_status_h);

        status->status = AED_STATUS_SUCCESS;
        status->converged = padded_size - segment->aed_deflate_bottom;
        status->computed_shifts = segment->aed_deflate_bottom - 1;

        starpu_data_release(segment->aed_status_h);

        // finalize the deflation process
        perform_deflate_finalize(segment, args);
    }

    return segment->status;
}

///
/// @brief Performs a single-shot large AED.
///
///  Outcomes:
///   - In the end, the perform_deflate_finalize function is called. Other MPI
///     ranks should call the perform_deflate_finalize function.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_single_shot_aed(
    struct segment *segment, struct process_args *args)
{
    // reduce the AED window to Schur form
    while (segment->children->top != NULL) {
        starneig_error_t ret =
            scan_segment_list(segment->children, &segment->aed_args);
        if (ret != STARNEIG_SUCCESS) {
            starneig_verbose("Large AED related QR/QZ failed.");
            return perform_deflate_finalize(segment, args);
        }
    }

    // cleanup
    starneig_free_segment_list(segment->children);
    segment->children = NULL;

    // deflate
    segment->status = SEGMENT_AED_DEFLATE;
    while (segment->status == SEGMENT_AED_DEFLATE)
        perform_deflate_step(segment, args);

    return segment->status;
}

///
/// @brief Performs a small QR algorithm.
///
///  Outcomes:
///   - A small QR window task and related update tasks are inserted and the
///     segment is marked as SEGMENT_SMALL.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_small(
    struct segment *segment, struct process_args *args)
{
    // free the aftermath vector

    if (segment->bulges_aftermath != NULL) {
        starneig_free_vector_descr(segment->bulges_aftermath);
        segment->bulges_aftermath = NULL;
    }

    // insert a small QR task

    starpu_data_handle_t lQ_h, lZ_h;
    starneig_schur_insert_small_schur(
        segment->begin, segment->end, args->max_prio,
        args->thres_a, args->thres_b, args->thres_inf,
        args->matrix_a, args->matrix_b, &segment->small_status_h,
        &lQ_h, &lZ_h, args->mpi);

    // insert related update tasks

    insert_segment_updates(
        segment->begin, segment->end, lQ_h, lZ_h, segment, args,
        UPDATE_DIRECTION_NONE);

    if (lZ_h != NULL && lZ_h != lQ_h)
        starpu_data_unregister_submit(lZ_h);
    starpu_data_unregister_submit(lQ_h);

    // gather the small QR task state to all MPI nodes

#ifdef STARNEIG_ENABLE_MPI
    if (args->mpi != NULL)
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), segment->small_status_h);
#endif

    segment->status = SEGMENT_SMALL;
    return segment->status;
}

///
/// @brief Performs a small AED.
///
///  Outcomes:
///   - A AED window task is inserted and the segment is marked as
///     SEGMENT_AED_SMALL.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_small_aed(
    int window_size, struct segment *segment, struct process_args *args)
{
    segment->aed_begin =
        MAX(segment->begin, segment->end - window_size - 1);

    // setup the shift vectors

    int owner = starneig_get_elem_owner_matrix_descr(
        segment->aed_begin, segment->aed_begin, args->matrix_a);

    segment->shifts_real = starneig_init_vector_descr(
        segment->end-segment->aed_begin, segment->end-segment->aed_begin,
        sizeof(double), starneig_single_owner_vector_descr, &owner,
        args->mpi);

    segment->shifts_imag = starneig_init_vector_descr(
        segment->end-segment->aed_begin, segment->end-segment->aed_begin,
        sizeof(double), starneig_single_owner_vector_descr, &owner,
        args->mpi);

    // insert a AED window task

    starneig_schur_insert_aggressively_deflate(
        segment->aed_begin, segment->end, args->max_prio,
        args->thres_a, args->thres_b, args->thres_inf,
        args->matrix_a, args->matrix_b,
        segment->shifts_real, segment->shifts_imag, &segment->aed_status_h,
        &segment->aed_small_lQ_h, &segment->aed_small_lZ_h, args->mpi);

    // gather the AED window task state to all MPI nodes

#ifdef STARNEIG_ENABLE_MPI
    if (args->mpi != NULL)
        starpu_mpi_get_data_on_all_nodes_detached(
            starneig_mpi_get_comm(), segment->aed_status_h);
#endif

    segment->status = SEGMENT_AED_SMALL;
    return segment->status;
}

///
/// @brief Begins a large AED.
///
///  Outcomes:
///   - 7 new sub-problem is created and the segment is marked as
///     SEGMENT_AED_SCHUR. In MPI mode, the perform_single_shot_aed or
///     perform_deflate_finalize is called.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_large_aed(
    int window_size, struct segment *segment, struct process_args *args)
{
    int my_rank = starneig_mpi_get_comm_rank();

    segment->aed_begin =
        MAX(segment->begin, segment->end - window_size - 1);

    int owner = starneig_get_elem_owner_matrix_descr(
        segment->aed_begin, segment->aed_begin, args->matrix_a);

    starpu_variable_data_register(
        &segment->aed_status_h, -1, 0, sizeof(struct aed_status));
#ifdef STARNEIG_ENABLE_MPI
    if (args->mpi != NULL)
        starpu_mpi_data_register_comm(segment->aed_status_h,
            args->mpi->tag_offset++, owner, starneig_mpi_get_comm());
#endif

    if (my_rank == owner) {
        starpu_data_acquire(segment->aed_status_h, STARPU_W);
        struct aed_status *status = (struct aed_status *)
            starpu_variable_get_local_ptr(segment->aed_status_h);
            status->status = AED_STATUS_FAILURE;
        starpu_data_release(segment->aed_status_h);
    }

    // copy the AED window to a separate matrix

    // this is based on some experiments performed on Intel(R) Core(TM) i5-6600
    int tile_size = MAX(24, ((0.025*(segment->end-segment->aed_begin)+11)/8)*8);

    starneig_matrix_descr_t matrix_a = starneig_init_matrix_descr(
        segment->end-segment->aed_begin, segment->end-segment->aed_begin,
        tile_size, tile_size, -1, -1, sizeof(double),
        starneig_single_owner_matrix_descr, &owner, args->mpi);
    STARNEIG_EVENT_INHERIT(matrix_a, args->matrix_a);
    STARNEIG_EVENT_ADD_OFFSET(matrix_a, segment->aed_begin, segment->aed_begin);

    starneig_insert_copy_matrix(
        segment->aed_begin, segment->aed_begin, 0, 0,
        segment->end-segment->aed_begin, segment->end-segment->aed_begin,
        args->max_prio, args->matrix_a, matrix_a, args->mpi);

    starneig_matrix_descr_t matrix_b = NULL;
    if (args->matrix_b != NULL) {
        matrix_b = starneig_init_matrix_descr(
            segment->end-segment->aed_begin, segment->end-segment->aed_begin,
            tile_size, tile_size, -1, -1, sizeof(double),
            starneig_single_owner_matrix_descr, &owner, args->mpi);
        STARNEIG_EVENT_INHERIT(matrix_b, args->matrix_b);
        STARNEIG_EVENT_ADD_OFFSET(
            matrix_b, segment->aed_begin, segment->aed_begin);

        starneig_insert_copy_matrix(
            segment->aed_begin, segment->aed_begin, 0, 0,
            segment->end-segment->aed_begin, segment->end-segment->aed_begin,
            args->max_prio, args->matrix_b, matrix_b, args->mpi);
    }

    // create shift vectors

    segment->shifts_real = starneig_init_vector_descr(
        segment->end-segment->aed_begin, segment->end-segment->aed_begin,
        sizeof(double), starneig_single_owner_vector_descr, &owner,
        args->mpi);

    segment->shifts_imag = starneig_init_vector_descr(
        segment->end-segment->aed_begin, segment->end-segment->aed_begin,
        sizeof(double), starneig_single_owner_vector_descr, &owner,
        args->mpi);

    starneig_verbose("Rank %d is going to perform a parallel AED.", owner);

    // only the owner is going to insert tasks
    if (my_rank == owner) {

        // initialize local left-hand side transformation matrix

        starneig_matrix_descr_t matrix_q = starneig_init_matrix_descr(
            segment->end-segment->aed_begin,
            segment->end-segment->aed_begin,
            tile_size, tile_size, -1, -1, sizeof(double), NULL, NULL, NULL);

        starneig_insert_set_to_identity(args->max_prio, matrix_q, NULL);

        // initialize local right-hand side transformation matrix
        starneig_matrix_descr_t matrix_z = NULL;
        if (args->matrix_b != NULL) {
            matrix_z = starneig_init_matrix_descr(
                segment->end-segment->aed_begin,
                segment->end-segment->aed_begin,
                tile_size, tile_size, -1, -1, sizeof(double), NULL, NULL, NULL);

            starneig_insert_set_to_identity(args->max_prio, matrix_z, NULL);
        }

        // create sub-problem

        segment->children = starneig_create_segment_list();
        starneig_add_segment_to_list_bottom(starneig_create_segment(
            SEGMENT_NEW, 1, segment->end-segment->aed_begin),
            segment->children);

        starneig_build_process_args_from(
            args, matrix_q, matrix_z, matrix_a, matrix_b, &segment->aed_args);

        segment->aed_args.mpi = NULL;

        segment->aed_args.min_prio =
            MIN(args->max_prio, args->default_prio+1);
        segment->aed_args.max_prio = args->max_prio;
        segment->aed_args.default_prio =
            (segment->aed_args.min_prio + segment->aed_args.max_prio) / 2;

        segment->status = SEGMENT_AED_SCHUR;

        if (args->mpi != NULL)
            perform_single_shot_aed(segment, args);
    }
    else {
        starneig_build_process_args_from(
            args, NULL, NULL, matrix_a, matrix_b, &segment->aed_args);
        segment->aed_args.mpi = NULL;
        perform_deflate_finalize(segment, args);
    }

    return segment->status;
}

///
/// @brief Perform a bulge chasing aftermath check.
///
///  The function checks whether the segment has deflated or contains infinite
///  eigenvalues.
///
///  Outcomes:
///   - If the segment did not deflate, the segment is marked as SEGMENT_NEW.
///   - Otherwise, all sub-segments are added to a segment list and marked as
///     SEGMENT_NEW. The current segment is marked as SEGMENT_CHILDREN.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status perform_aftermath_check(
    struct segment *segment, struct process_args *args)
{
    struct segment_list *list = NULL;

    int deflated = 0;

    int *infinities = NULL;

    // The code can track two segment "candidates" simultaneously. These
    // variables are used to track where the segment candidates begin and end.
    // At the beginning, both segment candidates will be zero in size.

    int prev_begin = segment->begin;
    int prev_end = segment->begin;
    int end = segment->begin;

    // Process all intersecting aftermath vector tiles. The matching tiles
    // were gathered when the bulge chasing tasks were inserted.

    int ds_begin = STARNEIG_VECTOR_TILE_IDX(
        segment->begin, segment->bulges_aftermath);
    int ds_end = STARNEIG_VECTOR_TILE_IDX(
        segment->end - 1, segment->bulges_aftermath) + 1;
    for (int i = ds_begin; i < ds_end; i++) {

        // acquire the matching aftermath vector tile

        starpu_data_handle_t handle =
            starneig_get_tile_from_vector_descr(i, segment->bulges_aftermath);
        starpu_data_acquire(handle, STARPU_R);
        int *aftermath = (int *) starpu_vector_get_local_ptr(handle);

        int _begin = MAX(0, STARNEIG_VECTOR_IN_TILE_IDX(
            segment->begin, i, segment->bulges_aftermath));
        int _end = MIN(STARNEIG_VECTOR_BM(segment->bulges_aftermath),
            STARNEIG_VECTOR_IN_TILE_IDX(
                segment->end, i, segment->bulges_aftermath));

        // scan the current tile
        for (int j = _begin; j < _end; j++) {

            // check infinities
            if (aftermath[j] & BULGE_CHASING_AFTERMATH_INFINITY) {
                if (infinities == NULL) {
                    infinities =
                        malloc((segment->end-segment->begin)*sizeof(int));
                    memset(infinities, 0,
                        (segment->end-segment->begin)*sizeof(int));
                }

                int loc =
                    STARNEIG_VECTOR_EXT_IDX(i, j, segment->bulges_aftermath);

                starneig_verbose("Zero right-hand side entry at row %d.", loc);

                infinities[loc-segment->begin] = 1;
            }

            // skip the first row of the segment
            if (STARNEIG_VECTOR_EXT_IDX(i, j, segment->bulges_aftermath)
            == segment->begin)
                continue;

            // check deflation
            if (aftermath[j] & BULGE_CHASING_AFTERMATH_DEFLATED) {

                deflated = 1;

                // enlarge the bottommost segment candidate
                end = STARNEIG_VECTOR_EXT_IDX(i, j, segment->bulges_aftermath);

                starneig_verbose("Deflated at row %d.",
                    STARNEIG_VECTOR_EXT_IDX(i, j, segment->bulges_aftermath));

                // deflate 1-by-1 blocks directly
                if (end - prev_begin == 1) {

                    // both secment candidates are now empty
                    prev_begin = end;
                    prev_end = end;
                }

                // if the two segment candidates are small when merged, ...
                else if (end - prev_begin < 32) {

                    // merge the segment candidates. The merged segment
                    // candidate now becomes the topmost segment candidate. The
                    // bottommost segment candidate will be zero in size.
                    prev_end = end;
                }

                // if the two segment candidates are large when merged ...
                else {

                    // create a new segment from the topmost segment
                    // candidate

                    if (1 < prev_end - prev_begin) {
                        struct segment *new = starneig_create_segment(
                            SEGMENT_NEW, prev_begin, prev_end);
                        new->iter = segment->iter;

                        if (infinities != NULL)
                            insert_push_inf_top(
                                infinities+prev_begin-segment->begin, new,
                                args);
                        process_segment(new, args);

                        if (list == NULL)
                            list = starneig_create_segment_list();
                        starneig_add_segment_to_list_bottom(new, list);
                    }

                    // if the bottommost segment is small
                    if ( end - prev_end < 32) {

                        // The bottommost segment candidate now becomes the
                        // topmost segment candidate. The bottommost segment
                        // candidate will be zero in size.
                        prev_begin = prev_end;
                        prev_end = end;
                    }

                    // if the bottommost segment is large
                    else {

                        // create a new segment from the bottomost segment
                        // candidate

                        if (1 < end - prev_end) {
                            struct segment *new = starneig_create_segment(
                                SEGMENT_NEW, prev_end, end);
                            new->iter = segment->iter;

                            if (infinities != NULL)
                                insert_push_inf_top(
                                    infinities+prev_end-segment->begin, new,
                                    args);
                            process_segment(new, args);

                            if (list == NULL)
                                list = starneig_create_segment_list();
                            starneig_add_segment_to_list_bottom(new, list);
                        }

                        // both secment candidates are now empty
                        prev_begin = end;
                        prev_end = end;
                    }
                }
            }
        }

        starpu_data_release(handle);
    }

    end = segment->end;

    // if deflation occurred, ...
    if (deflated) {

        // add the remaining segment candidates to the segment list

        if (1 < prev_end - prev_begin) {
            struct segment *new = starneig_create_segment(
                SEGMENT_NEW, prev_begin, prev_end);
            new->iter = segment->iter;

            if (infinities != NULL) {
                insert_push_inf_top(
                    infinities+prev_begin-segment->begin, new, args);
                new->status = SEGMENT_BOOTSTRAP;
            }
            process_segment(new, args);

            if (list == NULL)
                list = starneig_create_segment_list();
            starneig_add_segment_to_list_bottom(new, list);
        }

        if (1 < end - prev_end) {
            struct segment *new = starneig_create_segment(
                SEGMENT_NEW, prev_end, end);
            new->iter = segment->iter;

            if (infinities != NULL) {
                insert_push_inf_top(
                    infinities+prev_end-segment->begin, new, args);
                new->status = SEGMENT_BOOTSTRAP;
            }
            process_segment(new, args);

            if (list == NULL)
                list = starneig_create_segment_list();
            starneig_add_segment_to_list_bottom(new, list);
        }

        // add the segment list to the current segment

        segment->children = list;

        // mark the segment as SEGMENT_CHILDREN

        segment->status = SEGMENT_CHILDREN;
    }
    else {
        if (infinities != NULL) {
            insert_push_inf_top(infinities, segment, args);
            segment->status = SEGMENT_BOOTSTRAP;
        }
        else {
            segment->status = SEGMENT_NEW;
        }
    }

    free(infinities);

    return segment->status;
}

static void extract_aftermath(
    int size, int rbegin, int cbegin, int m, int n, int ldA, int ldB,
    void const *arg, void const *_A, void const *_B, void **masks)
{
    double thres_inf = *((double *)arg);
    double const *A = _A;
    double const *B = _B;
    bulge_chasing_aftermath_t *mask = masks[0];

    if (cbegin == 0)
        mask[0] = BULGE_CHASING_AFTERMATH_NONE;

    for (int i = 0 < cbegin ? 0 : 1; i < size; i++) {
        mask[i] = BULGE_CHASING_AFTERMATH_NONE;
        if (A[(cbegin+i-1)*ldA+rbegin+i] == 0.0)
            mask[i] |= BULGE_CHASING_AFTERMATH_DEFLATED;
    }

    if (B != NULL) {
        for (int i = 0; i < size; i++) {
            if (fabs(B[(cbegin+i)*ldB+rbegin+i]) < thres_inf)
                mask[i] |= BULGE_CHASING_AFTERMATH_INFINITY;
        }
    }
}

///
/// @brief Processes a segment with state SEGMENT_BOOTSTRAP.
///
///  Outcomes:
///   - The function detects whether the matrix pencil has deflated or contains
///     infinite eigenvalues. The perform_aftermath_check() function is called.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment_bootstrap(
    struct segment *segment, struct process_args *args)
{
    segment->bulges_aftermath = starneig_init_matching_vector_descr(
        args->matrix_a, sizeof(bulge_chasing_aftermath_t), NULL, args->mpi);

    starneig_insert_scan_diagonal(
        0, STARNEIG_MATRIX_M(args->matrix_a), 0, 0, 0, 1, 0, args->max_prio,
        extract_aftermath, &args->thres_inf, args->matrix_a, args->matrix_b,
        args->mpi, segment->bulges_aftermath, NULL);

#ifdef STARNEIG_ENABLE_MPI
    // gather the deflation check vector to all MPI nodes
    if (args->mpi != NULL) {
        int world_size = starneig_mpi_get_comm_size();
        for (int i = 0; i < world_size; i++)
            starneig_gather_segment_vector_descr(
                i, segment->begin, segment->end, segment->bulges_aftermath);
    }
#endif

    return perform_aftermath_check(segment, args);
}

///
/// @brief Processes a segment with state SEGMENT_NEW.
///
///  The function starts a new iteration.
///
///  Outcomes:
///   - If the segment is small => perform_small is called.
///   - If the segment is large and the AED window is small => perform_small_aed
///     is called..
///   - If the segment is large and the AED window is large => perform_large_aed
///     is called.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment_new(
    struct segment *segment, struct process_args *args)
{
    int segment_size = segment->end - segment->begin;

    // segments of the size 1 have already converged
    if (segment_size < 2) {
        segment->status = SEGMENT_CONVERGED;
        return SEGMENT_CONVERGED;
    }

    // iteration limit check
    if (args->iteration_limit <= segment->iter) {
        segment->status = SEGMENT_FAILURE;
        return SEGMENT_FAILURE;
    }

    int small_limit = evaluate_parameter(segment_size, args->small_limit);

    int aed_parallel_soft_limit =
        evaluate_parameter(segment_size, args->aed_parallel_soft_limit);
    int aed_parallel_hard_limit =
        evaluate_parameter(segment_size, args->aed_parallel_hard_limit);
    aed_parallel_soft_limit =
        MAX(aed_parallel_soft_limit, aed_parallel_hard_limit);

    int aed_window_size;
    if (segment->iter == 0)
        aed_window_size = MIN(0.30*segment_size,
            evaluate_parameter(segment_size, args->shift_count));
    else
        aed_window_size = MIN(0.30*segment_size,
            evaluate_parameter(segment_size, args->aed_window_size));

    if (1 < segment->iter && 0 < segment->aed_failed) {
        starneig_verbose(
            "Encountered %d failed AEDs. Resizing the AED window.",
            segment->aed_failed);
        aed_window_size = MIN(segment_size/2,
            aed_window_size * (1.0 + 0.05*segment->aed_failed));
    }

    if (isnan(segment->slope)) {
        int submitted = starpu_task_nsubmitted();
        double time = starpu_timing_now();
        segment->slope =
            (submitted-segment->peak_submitted) / (time-segment->peak_time);
    }

    // if the new segment is small, ...
    if (segment_size < small_limit) {
        perform_small(segment, args);
    }
    // if the AED window size is below the soft limit, ...
    else if (aed_window_size < aed_parallel_soft_limit) {
        // if the AED window size is below the hard limit, ...
        if (aed_window_size < aed_parallel_hard_limit) {
            perform_small_aed(aed_window_size, segment, args);
        }
        // if the AED window size is above the hard limit and we are running in
        // distributed memory, ...
        else if (args->mpi != NULL) {
            perform_large_aed(aed_window_size, segment, args);
        }
        // if the AED window size is above the hard limit and we are running in
        // shared memory, ...
        else {

            // predict sequential AED finish time

            double task_length = starneig_predict_aggressively_deflate(
                args->matrix_b != NULL, aed_window_size);
            if (isnan(task_length))
                task_length = 0.0;

            int submitted = starpu_task_nsubmitted();
            double time = starpu_timing_now();
            int prediction = segment->peak_submitted +
                segment->slope*((time-segment->peak_time)+task_length);

            // if there submitted tasks left and the sequential AED is expected
            // to finish before the workers become idle, ...
            if (0 < submitted && (task_length == 0.0 || 0 < prediction))
                perform_small_aed(aed_window_size, segment, args);
            else
                perform_large_aed(aed_window_size, segment, args);
        }
    }
    else {
        perform_large_aed(aed_window_size, segment, args);
    }

    segment->iter++;

    return segment->status;
}

///
/// @brief Processes a segment with state SEGMENT_SMALL.
///
///  The function checks whether the related small QR window task has completed.
///
///  Outcomes:
///   - If the small QR window failed to converge => the segment is marked as
///     SEGMENT_FAILURE.
///   - If the small QR window task converged => the segment is marked as
///     SEGMENT_CONVERGED.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment_small(
    struct segment *segment, struct process_args *args)
{
    starpu_data_acquire(segment->small_status_h, STARPU_R);

    struct small_schur_status const * status =
        (struct small_schur_status const *) starpu_variable_get_local_ptr(
            segment->small_status_h);

    if (status->converged < segment->end - segment->begin) {
        segment->status = SEGMENT_FAILURE;
    }
    else {
        segment->status = SEGMENT_CONVERGED;
    }

    starpu_data_release(segment->small_status_h);
    starpu_data_unregister_submit(segment->small_status_h);
    segment->small_status_h = NULL;

    return segment->status;
}

///
/// @brief Processes a segment with state SEGMENT_AED_SMALL.
///
///  The function checks whether the related AED window tasks has completed.
///
///  Outcomes:
///   - If the AED window task generated enough shifts => bulge chasing tasks
///     are inserted and the matching deflation check vector tiles are gathered
///     to all MPI nodes. The segment is marked as SEGMENT_BULGES.
///     Otherwise the segment is marked as SEGMENT_NEW.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment_aed_small(
    struct segment *segment, struct process_args *args)
{
    starpu_data_acquire(segment->aed_status_h, STARPU_R);

    struct aed_status const *status = (struct aed_status const *)
        starpu_variable_get_local_ptr(segment->aed_status_h);

    // the AED window is badded from the top left corner
    int padded_size = segment->end - segment->aed_begin;

    int requested_shifts = MIN(0.30*padded_size, evaluate_parameter(
        segment->end - segment->begin, args->shift_count));

    int nibble = evaluate_parameter(
        segment->end - segment->begin, args->aed_nibble);

    int perform_bulge_chasing =
        status->converged == 0 || (
            requested_shifts/2 <= status->computed_shifts &&
            status->converged < 0.01*nibble*(padded_size-1)
        );

    // if AED managed to converge eigenvalues, ...
    if (0 < status->converged) {

        // insert the related updates and re-size the segment
        insert_aed_updates(segment->aed_begin, segment->end,
            segment->aed_small_lQ_h, segment->aed_small_lZ_h, segment, args);

        segment->end -= status->converged;

        segment->aed_failed = 0;
    }
    else {
        segment->aed_failed++;
    }

    starneig_verbose("Deflated %d eigenvalues.", status->converged);

    starpu_data_unregister_submit(segment->aed_small_lQ_h);
    if (segment->aed_small_lZ_h != NULL &&
    segment->aed_small_lZ_h != segment->aed_small_lQ_h)
        starpu_data_unregister_submit(segment->aed_small_lZ_h);
    segment->aed_small_lQ_h = NULL;
    segment->aed_small_lZ_h = NULL;

    if (perform_bulge_chasing) {

        //
        // insert bulge chasing tasks and mark the segment as SEGMENT_BULGES
        //

        segment->computed_shifts = status->computed_shifts;
        segment->status = SEGMENT_BULGES;
        if (args->bulges_window_placement == BULGES_WINDOW_PLACEMENT_FIXED)
            insert_bulges_fixed(segment, args);
        else
            insert_bulges_rounded(segment, args);
    }
    //
    // otherwise, ...
    //
    else {

        //
        // mark the segment as SEGMENT_NEW to begin a new iteration
        //

        segment->computed_shifts = 0;
        segment->status = SEGMENT_NEW;
    }

    starpu_data_release(segment->aed_status_h);
    starpu_data_unregister_submit(segment->aed_status_h);
    segment->aed_status_h = NULL;

    starneig_free_vector_descr(segment->shifts_real);
    segment->shifts_real = NULL;

    starneig_free_vector_descr(segment->shifts_imag);
    segment->shifts_imag = NULL;

    return segment->status;
}

///
/// @brief Processes a segment with state SEGMENT_AED_SCHUR.
///
///  Outcomes:
///   - If the children is not empty, scan_segment_list is called.
///   - If children becomes/is empty, the segment is marked as
///     SEGMENT_AED_DEFLATE and the perform_deflate_step function is called.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment_aed_schur(
    struct segment *segment, struct process_args *args)
{
    {
        starneig_error_t ret =
            scan_segment_list(segment->children, &segment->aed_args);
        if (ret != STARNEIG_SUCCESS)
            return perform_deflate_finalize(segment, args);
    }

    // if the QR step has finished, ...
    if (segment->children->top == NULL) {

        // cleanup
        starneig_free_segment_list(segment->children);
        segment->children = NULL;

        // perform first deflation step
        perform_deflate_step(segment, args);
    }

    return segment->status;
}

///
/// @brief Processes a segment with state SEGMENT_AED_DEFLATE.
///
///  Outcomes:
///   - The perform_deflate_step function is called.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_segment_aed_deflate(
    struct segment *segment, struct process_args *args)
{
    return perform_deflate_step(segment, args);
}

///
/// @brief Processes a segment with state SEGMENT_BULGES.
///
/// @param[in,out] list
///         Segment list.
///
/// @param[in,out] args
///         Segment processing arguments.
///
/// @return Segment status.
///
static enum segment_status process_bulges(
    struct segment *segment, struct process_args *args)
{
    return perform_aftermath_check(segment, args);
}

static char const * status_to_str(enum segment_status status) {
    switch (status) {
        case SEGMENT_BOOTSTRAP:
            return "SEGMENT_BOOTSTRAP";
        case SEGMENT_NEW:
            return "SEGMENT_NEW";
        case SEGMENT_SMALL:
            return "SEGMENT_SMALL";
        case SEGMENT_AED_SMALL:
            return "SEGMENT_AED_SMALL";
        case SEGMENT_AED_SCHUR:
            return "SEGMENT_AED_SCHUR";
        case SEGMENT_AED_DEFLATE:
            return "SEGMENT_AED_DEFLATE";
        case SEGMENT_BULGES:
            return "SEGMENT_BULGES";
        case SEGMENT_CHILDREN:
            return "SEGMENT_CHILDREN";
        case SEGMENT_CONVERGED:
            return "SEGMENT_CONVERGED";
        case SEGMENT_FAILURE:
            return "SEGMENT_FAILURE";
        default:
            return "MYSTERY";
    }
}

static enum segment_status process_segment(
    struct segment *segment, struct process_args *args)
{
    enum segment_status old_status = segment->status;
    int old_begin = segment->begin;
    int old_end = segment->end;

    if (segment->status == SEGMENT_AED_SCHUR) {
        starneig_verbose("Segment [%d,%d[, %s ===>",
            segment->begin, segment->end, status_to_str(segment->status));
        starneig_verbose(
            "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv");
    }

    switch (segment->status) {
        case SEGMENT_BOOTSTRAP:
            // ===> SEGMENT_BULGES
            process_segment_bootstrap(segment, args);
            break;
        case SEGMENT_NEW:
            // ===> SEGMENT_SMALL or SEGMENT_AED_SMALL or SEGMENT_AED_SCHUR
            process_segment_new(segment, args);
            break;
        case SEGMENT_SMALL:
            // ===> SEGMENT_SMALL or SEGMENT_CONVERGED (or SEGMENT_FAILURE)
            process_segment_small(segment, args);
            break;
        case SEGMENT_AED_SMALL:
            // ===> SEGMENT_AED_SMALL or SEGMENT_NEW or SEGMENT_BULGES
            process_segment_aed_small(segment, args);
            break;
        case SEGMENT_AED_SCHUR:
            // ===> SEGMENT_AED_SCHUR or SEGMENT_AED_DEFLATE
            process_segment_aed_schur(segment, args);
            break;
        case SEGMENT_AED_DEFLATE:
            // ===> SEGMENT_AED_DEFLATE or SEGMENT_NEW or SEGMENT_BULGES
            process_segment_aed_deflate(segment, args);
            break;
        case SEGMENT_BULGES:
            // ===> SEGMENT_BULGES or SEGMENT_NEW or SEGMENT_CHILDREN
            process_bulges(segment, args);
            break;
        case SEGMENT_CHILDREN:
            // the current segment will be replaced by it's children
        case SEGMENT_CONVERGED:
            // the current segment will be removed
        case SEGMENT_FAILURE:
            // the current segment is reported as a failure
            break;
        default:
            STARNEIG_ASSERT(0);
    }

    if (old_status == SEGMENT_AED_SCHUR) {
        starneig_verbose(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
        starneig_verbose(" ===> [%d,%d[, %s.",
            segment->begin, segment->end, status_to_str(segment->status));
    }
    else {
        starneig_verbose("Segment [%d,%d[, %s ===> [%d,%d[, %s.",
            old_begin, old_end, status_to_str(old_status),
            segment->begin, segment->end, status_to_str(segment->status));
    }

    return segment->status;
}

static starneig_error_t scan_segment_list(
    struct segment_list *list, struct process_args *args)
{
    // loop over the segments
    struct segment *iter = list->top;
    while (iter != NULL) {

        // process segment
        process_segment(iter, args);

        // if the segment converged, ...
        if (iter->status == SEGMENT_CONVERGED) {
//            starneig_message("Segment [%d,%d[ converged with %d iterations.",
//                iter->begin, iter->end, iter->iter);
            // remove it from the list and move to the next segment
            struct segment *next = iter->down;
            starneig_remove_segment_from_list(iter, list);
            starneig_free_segment(iter);
            iter = next;
        }
        // if the segment has children, ...
        else if (iter->status == SEGMENT_CHILDREN) {
            // replace the segment with them and continue from the topmost
            // child
            struct segment *next = iter->down;
            starneig_replace_segment_with_list(iter, iter->children, list);
            iter = next;
        }
        // if the segment caused an failure, ...
        else if (iter->status == SEGMENT_FAILURE) {
            // stop scan and report the failure
            return STARNEIG_DID_NOT_CONVERGE;
        }
        // otherwise,
        else {
            // move to the next segment
            iter = iter->down;
        }
    }

    return STARNEIG_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

starneig_error_t starneig_schur_insert_tasks(
    struct starneig_schur_conf const *conf,
    starneig_matrix_descr_t Q,
    starneig_matrix_descr_t Z,
    starneig_matrix_descr_t A,
    starneig_matrix_descr_t B,
    starneig_vector_descr_t real,
    starneig_vector_descr_t imag,
    starneig_vector_descr_t beta,
    mpi_info_t mpi)
{
    starneig_error_t ret = STARNEIG_SUCCESS;
    struct segment_list *list = NULL;

    //
    // check threshold arguments
    //

    if (conf->left_threshold != STARNEIG_SCHUR_DEFAULT_THRESHOLD &&
    conf->left_threshold != STARNEIG_SCHUR_NORM_STABLE_THRESHOLD &&
    conf->left_threshold != STARNEIG_SCHUR_LAPACK_THRESHOLD &&
    conf->left_threshold <= 0.0) {
        starneig_error("Invalid left threshold.");
        ret = STARNEIG_INVALID_CONFIGURATION;
        goto cleanup;
    }

    if (conf->right_threshold != STARNEIG_SCHUR_DEFAULT_THRESHOLD &&
    conf->right_threshold != STARNEIG_SCHUR_NORM_STABLE_THRESHOLD &&
    conf->right_threshold != STARNEIG_SCHUR_LAPACK_THRESHOLD &&
    conf->right_threshold <= 0.0) {
        starneig_error("Invalid right threshold.");
        ret = STARNEIG_INVALID_CONFIGURATION;
        goto cleanup;
    }

    if (conf->inf_threshold != STARNEIG_SCHUR_DEFAULT_THRESHOLD &&
    conf->inf_threshold != STARNEIG_SCHUR_NORM_STABLE_THRESHOLD &&
    conf->inf_threshold <= 0.0) {
        starneig_error("Invalid infinity threshold.");
        ret = STARNEIG_INVALID_CONFIGURATION;
        goto cleanup;
    }

    //
    // compute norms if necessary
    //

    starpu_data_handle_t norm_a_h = NULL;
    if (conf->left_threshold == STARNEIG_SCHUR_DEFAULT_THRESHOLD ||
    conf->left_threshold == STARNEIG_SCHUR_NORM_STABLE_THRESHOLD)
        norm_a_h =
            starneig_schur_insert_compute_norm(STARPU_MAX_PRIO, A, mpi);

    starpu_data_handle_t norm_b_h = NULL;
    if (B != NULL) {
        if (conf->right_threshold == STARNEIG_SCHUR_DEFAULT_THRESHOLD ||
        conf->right_threshold == STARNEIG_SCHUR_NORM_STABLE_THRESHOLD ||
        conf->inf_threshold == STARNEIG_SCHUR_DEFAULT_THRESHOLD ||
        conf->inf_threshold == STARNEIG_SCHUR_NORM_STABLE_THRESHOLD)
            norm_b_h =
                starneig_schur_insert_compute_norm(STARPU_MAX_PRIO, B, mpi);
    }

    //
    // set thresholds
    //

    double norm_a = 0.0;
    if (norm_a_h != NULL) {
        starpu_data_acquire(norm_a_h, STARPU_R);
        norm_a = *((double *) starpu_data_get_local_ptr(norm_a_h));
        starpu_data_release(norm_a_h);
        starpu_data_unregister(norm_a_h);
    }

    double norm_b = 0.0;
    if (norm_b_h != NULL) {
        starpu_data_acquire(norm_b_h, STARPU_R);
        norm_b = *((double *) starpu_data_get_local_ptr(norm_b_h));
        starpu_data_release(norm_b_h);
        starpu_data_unregister(norm_b_h);
    }

    double thres_a = 0.0;
    if (conf->left_threshold == STARNEIG_SCHUR_DEFAULT_THRESHOLD ||
    conf->left_threshold == STARNEIG_SCHUR_NORM_STABLE_THRESHOLD) {
        thres_a = dlamch("Precision") * norm_a;
    }
    else if (conf->left_threshold == STARNEIG_SCHUR_LAPACK_THRESHOLD) {
        thres_a = 0.0;
    }
    else {
        thres_a = conf->left_threshold;
    }

    double thres_b = 0.0;
    if (B != NULL) {
        if (conf->right_threshold == STARNEIG_SCHUR_DEFAULT_THRESHOLD ||
        conf->right_threshold == STARNEIG_SCHUR_NORM_STABLE_THRESHOLD) {
            thres_b = dlamch("Precision") * norm_b;
        }
        else if (conf->right_threshold == STARNEIG_SCHUR_LAPACK_THRESHOLD) {
            thres_b = 0.0;
        }
        else {
            thres_b = conf->right_threshold;
        }
    }

    double thres_inf = 0.0;
    if (B != NULL) {
        if (conf->inf_threshold == STARNEIG_SCHUR_DEFAULT_THRESHOLD ||
        conf->inf_threshold == STARNEIG_SCHUR_NORM_STABLE_THRESHOLD) {
            thres_inf = dlamch("Precision") * norm_b;
        }
        else {
            thres_inf = conf->inf_threshold;
        }
    }

    //
    // build arguments
    //

    struct process_args args;
    ret = starneig_build_process_args(
        conf, Q, Z, A, B, thres_a, thres_b, thres_inf, mpi, &args);

    if (ret != STARNEIG_SUCCESS)
        goto cleanup;

    starneig_message("Using AED windows size %d.", (int)
        evaluate_parameter(STARNEIG_MATRIX_N(A), args.aed_window_size));
    starneig_message("Using %d shifts.", (int)
        evaluate_parameter(STARNEIG_MATRIX_N(A), args.shift_count));

    //
    // prepare for the bootstrap process
    //

    list = starneig_create_segment_list();
    starneig_add_segment_to_list_top(starneig_create_segment(
        SEGMENT_BOOTSTRAP, 0, STARNEIG_MATRIX_M(A)), list);

    //
    // main loop
    //

    while (list->top != NULL) {
        ret = scan_segment_list(list, &args);
        if (ret != STARNEIG_SUCCESS)
            goto cleanup;
    }

    //
    // extract eigenvalues
    //

    if (real != NULL && imag != NULL)
        starneig_insert_extract_eigenvalues(
            STARPU_MAX_PRIO, A, B, real, imag, beta, mpi);

cleanup:

    //
    // clean up
    //

    starneig_free_segment_list(list);

    return ret;
}
