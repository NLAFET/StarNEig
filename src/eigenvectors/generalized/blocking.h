///
/// @file
///
/// @brief Header file
///
/// @author Carl Christian K. Mikkelsen (spock@cs.umu.se), Umeå University
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

#ifndef STARNEIG_EIGVEC_GEN_BLOCKING_H_
#define STARNEIG_EIGVEC_GEN_BLOCKING_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Count mini-blocks of a quasi-upper triangular matrix.
///
/// @param[in] m
///         Dimension of matrix.
///
/// @param[in] a
///         Array containing matrix.
///
/// @param[in] lda
///         Leading dimension of array a.
///
/// @return The number of 1-by-1 and 2-by-2 blocks found along the main
/// diagonal.
///
int starneig_eigvec_gen_count_blocks(int m, double *a, size_t lda);

///
/// @brief Map the mini-block structure of a quasi-upper triangular matrix
///
/// @param[in] m
///         dimension of matrix
///
/// @param[in] a
///         array containing matrix
///
/// @param[in] lda
///         leading dimension of array a
///
/// @param[out] blocks
///         array of length at least numBlocks+1
///
/// @param[in] numBlocks
///         maximum number of blocks to find
///
/// @return the number of 1-by-1 and 2-by-2 blocks mapped along the main
/// diagonal
///
int starneig_eigvec_gen_find_blocks(
    int m, double *a, size_t lda, int *bp, int numBlocks);

#endif // STARNEIG_EIGVEC_GEN_BLOCKING_H_
