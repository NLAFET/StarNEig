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

#ifndef STARNEIG_COMMON_TRACE_H
#define STARNEIG_COMMON_TRACE_H

#include <starneig_config.h>
#include <starneig/configuration.h>

#ifdef STARNEIG_ENABLE_EVENTS

#include "tiles.h"

///
/// @brief Data type for event color.
///
typedef unsigned char * event_color;

///
/// @brief Blue event color.
///
extern const event_color starneig_event_blue;

///
/// @brief Green event color.
///
extern const event_color starneig_event_green;

///
/// @brief Red event color.
///
extern const event_color starneig_event_red;

///
/// @brief Initialized event traces.
///
void starneig_event_init();

///
/// @brief Frees event traces.
///
void starneig_event_free();

///
/// @brief Begins an event.
///
/// @param[in] pi
///         The packing info that defines the windows.
///
/// @param[in] color
///         The trace color.
///
void starneig_event_begin(
    struct packing_info const *pi, const event_color color);

///
/// @brief Ends an event.
///
void starneig_event_end();

///
/// @brief Stores the trace into a file.
///
/// @param[in] n
///         The matrix dimension.
///
/// @param[in] file_name
///         The file name.
///
void starneig_event_store(int n, char const *file_name);

#define STARNEIG_EVENT_INIT() \
    starneig_event_init()
#define STARNEIG_EVENT_FREE() \
    starneig_event_free()
#define STARNEIG_EVENT_BEGIN(pi, color) \
    starneig_event_begin(pi, color)
#define STARNEIG_EVENT_END() \
    starneig_event_end()
#define STARNEIG_EVENT_STORE(n, filename) \
    starneig_event_store(n, filename)
#define STARNEIG_EVENT_SET_LABEL(matrix, label) \
if (matrix != NULL) { \
    matrix->event_enabled = 1; \
    matrix->event_label = label; \
}
#define STARNEIG_EVENT_INHERIT(matrix, source) \
if (matrix != NULL && source != NULL) { \
    matrix->event_enabled = source->event_enabled; \
    matrix->event_label = source->event_label; \
    matrix->event_roffset = source->event_roffset; \
    matrix->event_coffset = source->event_coffset; \
}
#define STARNEIG_EVENT_ADD_OFFSET(matrix, roffset, coffset) \
if (matrix != NULL) { \
    matrix->event_roffset += roffset; \
    matrix->event_coffset += coffset; \
}

#else

#define STARNEIG_EVENT_INIT() {}
#define STARNEIG_EVENT_FREE() {}
#define STARNEIG_EVENT_BEGIN(pi, color)  {}
#define STARNEIG_EVENT_END() {}
#define STARNEIG_EVENT_STORE(n, filename) {}
#define STARNEIG_EVENT_SET_LABEL(matrix, label) {}
#define STARNEIG_EVENT_INHERIT(matrix, source) {}
#define STARNEIG_EVENT_ADD_OFFSET(matrix, roffset, coffset) {}

#endif

#endif
