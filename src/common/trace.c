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
#include "trace.h"
#include "common.h"
#include <time.h>
#include <string.h>
#include <starpu.h>

#define MAX_EVENTS 1000000

struct event {
    char label;
    float begin;
    float end;
    int rbegin;
    int rend;
    int cbegin;
    int cend;
    unsigned char color[3];
};

static struct event *events[STARPU_NMAXWORKERS];
static int event_counts[STARPU_NMAXWORKERS];
static int event_active[STARPU_NMAXWORKERS];
static struct timespec event_base;

const event_color starneig_event_blue = (unsigned char[]) { 0, 0, 128 };
const event_color starneig_event_green = (unsigned char[]) { 0, 128, 0 };
const event_color starneig_event_red = (unsigned char[]) { 128, 0, 0 };

void starneig_event_init()
{
    for (int i = 0; i < STARPU_NMAXWORKERS; i++) {
        free(events[i]);
        events[i] = NULL;
        event_counts[i] = 0;
        event_active[i] = 0;
    }

    int worker_count = starpu_worker_get_count();
    for (int i = 0; i < worker_count; i++)
        events[i] = malloc(MAX_EVENTS*sizeof(struct event));

    clock_gettime(CLOCK_REALTIME, &event_base);
}

void starneig_event_free()
{
    for (int i = 0; i < STARPU_NMAXWORKERS; i++) {
        free(events[i]);
        events[i] = NULL;
    }
}

void starneig_event_begin(
    struct packing_info const *pi, const event_color color)
{
    if (!pi->event_enabled)
        return;

    int worker_id = starpu_worker_get_id();

    if (event_counts[worker_id] + 1 < MAX_EVENTS) {
        struct event *event = &events[worker_id][event_counts[worker_id]];

        event->label = pi->event_label;

        struct timespec start;
        clock_gettime(CLOCK_REALTIME, &start);
        event->begin =
            (start.tv_sec-event_base.tv_sec) * 1E+3 +
            (start.tv_nsec-event_base.tv_nsec) * 1E-6;

        event->rbegin = pi->event_roffset + pi->roffset;
        event->rend = pi->event_roffset + pi->roffset + (pi->rend - pi->rbegin);
        event->cbegin = pi->event_coffset + pi->coffset;
        event->cend = pi->event_coffset + pi->coffset + (pi->cend - pi->cbegin);
        memcpy(event->color, color, 3*sizeof(unsigned char));

        event_active[worker_id] = 1;
    }
}

void starneig_event_end()
{
    int worker_id = starpu_worker_get_id();

    if (!event_active[worker_id])
        return;

    if (event_counts[worker_id]+1 < MAX_EVENTS) {
        struct timespec start;
        clock_gettime(CLOCK_REALTIME, &start);
        events[worker_id][event_counts[worker_id]].end =
            (start.tv_sec-event_base.tv_sec) * 1E+3 +
            (start.tv_nsec-event_base.tv_nsec) * 1E-6;
        event_counts[worker_id]++;
    }

    event_active[worker_id] = 0;
}

void starneig_event_store(int n, char const *file_name)
{
    int worker_count = starpu_worker_get_count();

    int total_events = 0;
    for (int i = 0; i < worker_count; i++)
        total_events += event_counts[i];

    struct event *_events = malloc(total_events*sizeof(struct event));

    int _offset = 0;
    for (int i = 0; i < worker_count; i++) {
        memcpy(
            _events+_offset, events[i], event_counts[i]*sizeof(struct event));
        _offset += event_counts[i];
    }

    float begin = 1.0/0.0, end = 0.0;
    for (int i = 0; i < total_events; i++) {
        begin = MIN(begin, _events[i].begin);
        end = MAX(end, _events[i].end);
    }

    FILE *file = fopen(file_name, "wb");

    fwrite(&n, sizeof(n), 1, file);
    fwrite(&total_events, sizeof(total_events), 1, file);
    fwrite(&begin, sizeof(begin), 1, file);
    fwrite(&end, sizeof(end), 1, file);
    fwrite(_events, sizeof(struct event), total_events, file);
    fclose(file);

    free(_events);
}
