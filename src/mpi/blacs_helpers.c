///
/// @file
///
/// @brief This file contains various BLACS helper functions.
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
#include <starneig/blacs_helpers.h>
#include "../common/common.h"

__attribute__ ((visibility ("default")))
void starneig_blacs_pinfo(int *my_rank, int *rank_count)
{
    extern void blacs_pinfo_(int *, int *);
    blacs_pinfo_(my_rank, rank_count);
}

__attribute__ ((visibility ("default")))
int starneig_blacs_get(
    starneig_blacs_context_t context, starneig_blacs_query_id_t query)
{
    extern void blacs_get_(int const *, int const *, int *);

    int val;
    blacs_get_(&context, &query, &val);
    return val;
}

__attribute__ ((visibility ("default")))
starneig_blacs_context_t starneig_blacs_gridinit(
    starneig_blacs_context_t system_context, char *order,
    int rows, int cols)
{
    extern void blacs_gridinit_(int *, char const *, int const *, int const *);
    blacs_gridinit_(&system_context, order, &rows, &cols);
    return system_context;
}

__attribute__ ((visibility ("default")))
void starneig_blacs_gridinfo(
    starneig_blacs_context_t context, int *rows, int *cols, int *row, int *col)
{
    extern void blacs_gridinfo_(int const *, int *, int *, int *, int *);
    blacs_gridinfo_(&context, rows, cols, row, col);
}

__attribute__ ((visibility ("default")))
void starneig_blacs_pcoord(
    starneig_blacs_context_t context, int process, int *row, int *col)
{
    extern void blacs_pcoord_(int const *, int const *, int *, int *);
    blacs_pcoord_(&context, &process, row, col);
}

__attribute__ ((visibility ("default")))
void starneig_blacs_gridexit(starneig_blacs_context_t context)
{
    extern void blacs_gridexit_(int const *);
    blacs_gridexit_(&context);
}

__attribute__ ((visibility ("default")))
void starneig_blacs_exit(int cont)
{
    extern void blacs_exit_(int const *);
    blacs_exit_(&cont);
}

__attribute__ ((visibility ("default")))
int starneig_blacs_numroc(
    int n, int nb, int iproc, int isrcproc, int nprocs)
{
    extern int numroc_(
        int const *, int const *, int const *, int const *, int const *);
    return numroc_(&n, &nb, &iproc, &isrcproc, &nprocs);
}

__attribute__ ((visibility ("default")))
int starneig_blacs_descinit(
    struct starneig_blacs_descr *descr, int m, int n, int sm, int sn,
    int irsrc, int icsrc, starneig_blacs_context_t context, int ld)
{
    extern void descinit_(
        struct starneig_blacs_descr *, int const *, int const *,
        int const *, int const *, int const *, int const *, int const *,
        int const *, int *);

    int info;
    descinit_(descr, &m, &n, &sm, &sn, &irsrc, &icsrc, &context, &ld, &info);
    return info;
}

// deprecated
__attribute__ ((visibility ("default")))
int starneig_numroc(
    int n, int nb, int iproc, int isrcproc, int nprocs)
{
    starneig_warning("starneig_numroc has been deprecated.");
    return starneig_blacs_numroc(n, nb, iproc, isrcproc, nprocs);
}

// deprecated
__attribute__ ((visibility ("default")))
int starneig_descinit(
    struct starneig_blacs_descr *descr, int m, int n, int sm, int sn,
    int irsrc, int icsrc, starneig_blacs_context_t context, int ld)
{
    starneig_warning("starneig_descinit has been deprecated.");
    return starneig_descinit(descr, m, n, sm, sn, irsrc, icsrc, context, ld);
}
