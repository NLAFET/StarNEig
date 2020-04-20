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

#include <algorithm>
#include <CImg.h>
using namespace cimg_library;

#define H 720
#define W (2*(H)+10)

//#define H 1024
//#define W 1024

enum fill { UPPER, FULL };

static const unsigned char gray[] = { 240, 240, 240 };

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

void draw_window(
    int rbegin, int rend, int cbegin, int cend, int m, int n,
    int f_rbegin, int f_rend, int f_cbegin, int f_cend,
    unsigned char const *color, float weight, CImg<unsigned char>& frame)
{
    int height = f_rend - f_rbegin;
    int width = f_cend - f_cbegin;

    frame.draw_rectangle(
        f_cbegin+(1.0*cbegin/n)*width, f_rbegin+(1.0*rbegin/m)*height,
        f_cbegin+(1.0*cend/n)*width, f_rbegin+(1.0*rend/m)*height, color, weight);
}

void draw_between(
    char label, enum fill fill, double t_begin, double t_end,
    struct event const *events, int event_count, int m, int n,
    int f_rbegin, int f_rend, int f_cbegin, int f_cend,
    CImg<unsigned char>& frame)
{
    if (fill == UPPER)
        frame.draw_triangle(
            f_cbegin, f_rbegin, f_cend, f_rbegin, f_cend, f_rend, gray, 1.0);
    else
        frame.draw_rectangle(f_cbegin, f_rbegin, f_cend, f_rend, gray, 1.0);

    for (int i = 0; i < event_count; i++) {
        if (events[i].label == label && t_begin < events[i].end &&
            events[i].begin <= t_end) {

            float weight =
                std::max<float>(0.33,
                    std::min<float>(t_end, events[i].end) -
                    std::max<float>(t_begin, events[i].begin)
                ) / (t_end-t_begin);

            draw_window(
                events[i].rbegin, events[i].rend,
                events[i].cbegin, events[i].cend,
                m, n, f_rbegin, f_rend, f_cbegin, f_cend,
                events[i].color, weight, frame);
        }
    }
}

int main(int argc, char **argv)
{
    FILE *file = fopen(argv[1], "rb");
    int frames = atoi(argv[2]);

    int n, total_events;
    float begin, end;
    fread(&n, sizeof(n), 1, file);
    fread(&total_events, sizeof(total_events), 1, file);
    fread(&begin, sizeof(begin), 1, file);
    fread(&end, sizeof(end), 1, file);

    struct event *events = new struct event[total_events];
    fread(events, sizeof(struct event), total_events, file);

    fclose(file);

    for (int i = 0; i < frames; i++) {
        CImg<unsigned char> frame(W, H, 1, 3, 255);

        float _begin = begin + i*(end-begin)/frames;
        float _end = begin + (i+1)*(end-begin)/frames;
/*
        draw_between('A', UPPER, _begin, _end, events, total_events, n, n,
            0, H/2-5, 0, W/2-5, frame);
        draw_between('B', UPPER, _begin, _end, events, total_events, n, n,
            H/2+5, H, 0, W/2-5, frame);
        draw_between('Q', FULL,  _begin, _end, events, total_events, n, n,
            0, H/2-5, W/2+5, W, frame);
        draw_between('Z', FULL,  _begin, _end, events, total_events, n, n,
            H/2+5, H, W/2+5, W, frame);
*/
        draw_between('A', UPPER, _begin, _end, events, total_events, n, n,
            0, H, 0, W/2-5, frame);
        draw_between('Q', FULL,  _begin, _end, events, total_events, n, n,
            0, H, W/2+5, W, frame);

        char filename[100];
        sprintf(filename, "frame_%05d.png", i);
        frame.save_png(filename);
    }

    delete events;

    return 0;
}
