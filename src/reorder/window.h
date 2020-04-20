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

#ifndef STARNEIG_REORDER_WINDOW_H
#define STARNEIG_REORDER_WINDOW_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <starpu.h>

///
/// @brief Diagonal computation window.
///
///  Each window has two index numbers:
///   - unique local index number inside a window chain
///   - unique global index number (used in StarPU-MPI)
///
struct window {
    int idx;                   ///< local index number
    int gidx;                  ///< global index number
    int begin;                 ///< first row that belongs to the window
    int end;                   ///< last row that belongs to the window + 1
    int swaps;                 ///< total number of diagonal blocks swaps
    starpu_data_handle_t lq_h; ///< handle to the corresponding local Q matrix
    starpu_data_handle_t lz_h; ///< handle to the corresponding local Z matrix
    struct window *up;         ///< window above the current window
    struct window *down;       ///< window below the current window
};

///
/// @brief Window chain.
///
///  The length field stores the actual length of the chain and it should not be
///  modified by hand. The effective_length field is used when a chain is split
///  into multiple sub-chains and it stores the length of the original chain.
///
struct window_chain {
    int begin;                 ///< first row that belongs to the window chain
    int end;                   ///< last row that belongs to the window chain +1
    int length;                ///< chain length
    int effective_length;      ///< effective chain length
    struct window *top;        ///< last/topmost window in the chain
    struct window *bottom;     ///< first/bottom window in the chain
    struct window_chain *up;   ///< previous window chain
    struct window_chain *down; ///< next window chain
};

///
/// @brief Chains list.
///
struct chain_list {
    struct window_chain *top;    ///< first/topmost window chain
    struct window_chain *bottom; ///< last/bottom window chain
    struct chain_list *next;     ///< next chain list
    struct chain_list *prev;     ///< previous chain list
    int longest_eff_length;      ///< longest chain (effective length)
    int total_length;            ///< combined length all chains
};

///
/// @brief Creates a new diagonal computation window.
///
///  Data fields lq_h, lz_h, up, and down are initialized to NULL.
///
/// @param[in] idx   - local index number
/// @param[in] gidx  - global index number
/// @param[in] begin - first row that belongs to the new window
/// @param[in] end   - last row that belongs to the new window + 1
/// @param[in] swaps - total number of involved diagonal block swaps
///
/// @return diagonal computation window
///
struct window* starneig_create_window(
    int idx, int gidx, int begin, int end, int swaps);

///
/// @brief Unregisters StarPU resources that are linked to a given diagonal
/// computation window.
///
/// @param[in,out] window - diagonal computation window
///
void starneig_unregister_window(struct window *window);

///
/// @brief Frees a previously allocated diagonal computation window and
/// unregisters the related StarPU resources.
///
/// @param[in,out] window - diagonal computation window
///
void starneig_free_window(struct window *window);

///
/// @brief Creates an empty window chain.
///
/// @param[in] begin - first row that belongs to the new window chain
/// @param[in] end   - last row that belongs to the new window chain + 1
///
/// @return window chain
///
struct window_chain* starneig_create_chain(int begin, int end);

///
/// @brief Adds a diagonal computation window to the "top" of a window chain.
///
/// @param[in,out] window - diagonal computation window
/// @param[in,out] chain - window chain
///
void starneig_add_window_to_chain_top(
    struct window *window, struct window_chain *chain);

///
/// @brief Removes a diagonal computation window from the "bottom" of a window
/// chain.
///
/// @param[in,out] chain - window chain
///
/// @return diagonal computation window
///
struct window* starneig_pop_window_from_chain_bottom(
    struct window_chain *chain);

///
/// @brief Unregisters StarPU resources that are linked to a given window chain.
///
/// @param[in,out] window - window chain
///
void starneig_unregister_chain(struct window_chain *chain);

///
/// @brief Frees a previously allocated window chain and unregisters the related
/// StarPU resources.
///
/// @param[in,out] chain - window chain
///
void starneig_free_chain(struct window_chain *chain);

///
/// @brief Creates an empty chain list.
///
/// @return chain list
///
struct chain_list* starneig_create_chain_list();

///
/// @brief Adds a window chain to the "bottom" of a chain list.
///
/// @param[in,out] chain - window chain
/// @param[in,out] list - chain list
///
void starneig_add_chain_to_list_bottom(
    struct window_chain *chain, struct chain_list *list);

///
/// @brief Removes a window chain from a chain list.
///
/// @param[in,out] chain - window chain
/// @param[in,out] list - chain list
///
void starneig_sever_chain_from_list(
    struct window_chain *chain, struct chain_list *list);

///
/// @brief Unregisters StarPU resources that are linked to a given chain list.
///
/// @param[in,out] list - chain list
///
void starneig_unregister_chain_list(struct chain_list *list);

///
/// @brief Frees a previously allocated chain list and unregisters the related
/// StarPU resources.
///
/// @param[in,out] list - chain list
///
void starneig_free_chain_list(struct chain_list *list);

#endif
