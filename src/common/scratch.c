///
/// @file
///
/// @brief This file contains code that implements a scratch buffer cache.
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
#include "scratch.h"

struct handle_list {
    int in_use;
    starpu_data_handle_t handle;
    struct handle_list *next;
};

static struct handle_list *hmatrices = NULL;
static struct handle_list *vmatrices = NULL;

static starpu_data_handle_t get_matrix(
    int m, int n, size_t elemsize, struct handle_list **list)
{
    if (*list == NULL) {
        *list = malloc(sizeof(struct handle_list));
        (*list)->in_use = 0;
        (*list)->handle = NULL;
        (*list)->next = NULL;
    }

    // this assumes that the last element in the list is empty
    struct handle_list *iter = *list;
    while (iter->in_use)
        iter = iter->next;

    if (iter->handle != NULL) {
        int _m = starpu_matrix_get_nx(iter->handle);
        int _n = starpu_matrix_get_ny(iter->handle);
        size_t _elemsize = starpu_matrix_get_elemsize(iter->handle);

        if (_m < m || _n < n || _elemsize != elemsize) {
            starpu_data_unregister_submit(iter->handle);
            iter->handle = NULL;
        }
    }

    if (iter->handle == NULL)
        starpu_matrix_data_register(&iter->handle, -1, 0, m, m, n, elemsize);

    iter->in_use = 1;

    if (iter->next == NULL) {
        iter->next = malloc(sizeof(struct handle_list));
        iter->next->in_use = 0;
        iter->next->handle = NULL;
        iter->next->next = NULL;
    }

    return iter->handle;
}

static void flush_list(struct handle_list **list)
{
    struct handle_list *iter = *list;
    while (iter != NULL) {
        iter->in_use = 0;
        iter = iter->next;
    }
}

static void unregister_list(struct handle_list **list)
{
    struct handle_list *iter = *list;
    while (iter != NULL) {
        struct handle_list *next = iter->next;
        if (iter->handle != NULL)
            starpu_data_unregister_submit(iter->handle);
        free(iter);
        iter = next;
    }

    *list = NULL;
}

starpu_data_handle_t starneig_scratch_get_matrix(int m, int n, size_t elemsize)
{
    if (m < n)
        return get_matrix(m, n, elemsize, &hmatrices);
    else
        return get_matrix(m, n, elemsize, &vmatrices);
}

void starneig_scratch_flush()
{
    flush_list(&hmatrices);
    flush_list(&vmatrices);
}

void starneig_scratch_unregister()
{
    unregister_list(&hmatrices);
    unregister_list(&vmatrices);
}
