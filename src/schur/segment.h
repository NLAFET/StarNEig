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

#ifndef STARNEIG_SCHUR_SEGMENT_H
#define STARNEIG_SCHUR_SEGMENT_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "process_args.h"
#include <starpu.h>

///
/// @brief Segment status enumerator.
///
///  At an any given moment, a segment is in one of the following states:
///
enum segment_status {
    SEGMENT_EMPTY = 0,   ///< empty segment
    SEGMENT_BOOTSTRAP,   ///< segment is waiting for a bootstrap sequence
    SEGMENT_NEW,         ///< segment should go through a new iteration
    SEGMENT_SMALL,       ///< small (sequential) QR algorithm in progress
    SEGMENT_AED_SMALL,   ///< small (sequential) AED in progress
    SEGMENT_AED_SCHUR,   ///< AED window is being reduced to Schur form
    SEGMENT_AED_DEFLATE, ///< AED window is being deflated
    SEGMENT_BULGES,      ///< bulge chasing in progress
    SEGMENT_CHILDREN,    ///< segment has been divided into sub-segments
    SEGMENT_CONVERGED,   ///< segment has converged
    SEGMENT_FAILURE      ///< an error has occurred while processing the segment
};

struct segment_list;

///
/// @brief Segment structure.
///
struct segment {

    /// segment status
    enum segment_status status;

    /// segment iteration counter
    int iter;

    /// first row/column that belongs to the segment
    int begin;

    /// last row/column that belongs to the segment + 1
    int end;

    /// when the segment is in the state SEGMENT_SMALL, this handle
    /// encapsulates the matching small QR window task status structure
    starpu_data_handle_t small_status_h;

    /// when the segment is in the state SEGMENT_AED_*, this variable
    /// stores the first row/column that belongs the padded AED window
    int aed_begin;

    /// stores the number of failed AEDs
    int aed_failed;

    /// Allocator for AED related tasks. Used when the segment is in the states
    /// SEGMENT_AED_SCHUR and SEGMENT_AED_DEFLATE.
    struct allocator *aed_allocator;

    /// when the segment is in the state SEGMENT_AED_*, this handle
    /// encapsulates the matching AED window task status structure
    starpu_data_handle_t aed_status_h;

    /// when the segment is in the state SEGMENT_AED_SMALL, this handle
    /// encapsulates the matching left-hand side AED transformation matrix
    starpu_data_handle_t aed_small_lQ_h;

    /// when the segment is in the state SEGMENT_AED_SMALL, this handle
    /// encapsulates the matching right-hand side AED transformation matrix
    starpu_data_handle_t aed_small_lZ_h;

    /// when the segment is in the state SEGMENT_AED_SCHUR or
    /// SEGMENT_AED_DEFLATE, this variable stores the segment processing
    /// arguments for the matching AED sub-matrix
    struct process_args aed_args;

    /// when the segment is in the state SEGMENT_AED_DEFLATE, this
    //// variable stores the location where the next batch of undeflated
    /// blocks should  be moved inside the padded AED window
    int aed_deflate_top;

    /// when the segment is in the state SEGMENT_AED_DEFLATE, this
    /// variable stores the location of the topmost deflated block inside
    /// the padded AED window
    int aed_deflate_bottom;

    /// when the segment is in the state SEGMENT_AED_DEFLATE, this handle
    /// encapsulates the matching deflate window task status structure
    starpu_data_handle_t aed_deflate_status_h;

    /// when the segment is in the state SEGMENT_AED_DEFLATE, this handle
    /// encapsulates the spike inducer (the sub-diagonal entry to the
    /// left of the AED window)
    starpu_data_handle_t aed_deflate_inducer_h;

    /// when the segment is in the state SEGMENT_AED_DEFLATE, this handle
    /// encapsulates the spike base (the first row from AED
    /// transformation  matrix)
    starneig_vector_descr_t aed_deflate_base;

    /// when the segment is in the state SEGMENT_BULGES, this variable
    /// stores the number of computed shifts
    int computed_shifts;

    /// when the segment is in the state SEGMENT_BULGES, this vectors
    /// stores the real parts of the computed shifts
    starneig_vector_descr_t shifts_real;

    /// when the segment is in the state SEGMENT_BULGES, this vectors
    /// stores the imaginary parts of the computed shifts
    starneig_vector_descr_t shifts_imag;

    /// bulge chasing aftermath vector
    starneig_vector_descr_t bulges_aftermath;

    /// peaks submitted task count (recorded just after the bulges have
    /// been inserted)
    int peak_submitted;

    /// time when pead_submitted got recorded
    double peak_time;

    /// calculated submitted tasks slope (i.e. the rate at which the
    /// tasks get consumed)
    double slope;

    /// sub-segments
    struct segment_list *children;

    /// previous/upper segment
    struct segment *up;

    /// next/lower segment
    struct segment *down;
};

///
/// @brief Segment list.
///
struct segment_list {
    struct segment *top;    ///< first/topmost segment
    struct segment *bottom; ///< last/bottom segment
};

///
/// @brief Creates a new segment.
///
/// @param[in] status  segment status
/// @param[in] begin   first row that belongs to the segment
/// @param[in] end     last row that belongs to the segment + 1
///
/// @return new segment
///
struct segment * starneig_create_segment(
    enum segment_status status, int begin, int end);

///
/// @brief Moves the contents of a segment to a new segment.
///
///  The old segment retains the list pointers (up and down). The new segment is
///  not linked (up and down are NULL).
///
/// @param[in,out] segment  segment whose contents are to be copied
///
/// @return new segment with identical contents
///
struct segment * starneig_move_segment(struct segment *segment);

///
/// @brief Frees a previously allocated segment.
///
/// @param[in,out] segment  segment to be freed
///
void starneig_free_segment(struct segment *segment);

///
/// @brief Creates an empty segment list.
///
/// @return empty segment list
///
struct segment_list * starneig_create_segment_list();

///
/// @brief Adds a segment to the beginning/top of a segment list.
///
/// @param[in,out] segment  segment to be added to the segment list
/// @param[in,out] list     segment list
///
void starneig_add_segment_to_list_top(
    struct segment *segment, struct segment_list *list);

///
/// @brief Adds a segment to the end/bottom of a segment list.
///
/// @param[in,out] segment  segment to be added segment list
/// @param[in,out] end      segment list
///
void starneig_add_segment_to_list_bottom(
    struct segment *segment, struct segment_list *list);

///
/// @brief Removes a segment from a segment list.
///
/// @param[in,out] segment  segment to be removed from the segment list
/// @param[in,out] list     segment list
///
void starneig_remove_segment_from_list(
    struct segment *segment, struct segment_list *list);

///
/// @brief Replaces a segment with a segment list.
///
/// @param[in,out] segment   segment to be replaced
/// @param[in,out] sub_list  segment list that replaces the segment
/// @param[in,out] list      segment list that contains the segment
///
void starneig_replace_segment_with_list(struct segment *segment,
    struct segment_list *sublist, struct segment_list *list);

///
/// @brief Frees a previously allocated segment list.
///
/// @param[in,out] list  segment list to be freed
///
void starneig_free_segment_list(struct segment_list *list);

#endif
