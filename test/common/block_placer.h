///
/// @file This file contains code that places a set of 1-by-1 and 2-by-2 blocks
/// to the diagonal of a matrix A or a matrix pencil (A,B).
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

#ifndef STARNEIG_TEST_COMMON_BLOCK_PLACER_H
#define STARNEIG_TEST_COMMON_BLOCK_PLACER_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "pencil.h"

///
/// @brief Places a set of 1-by-1 and 2-by-2 blocks to the diagonal of a matrix
/// A or a matrix pencil (A,B).
///
/// @param[in] real
///         The real parts of the eigenvalues.
///
/// @param[in] imag
///         The imaginary parts of the eigenvalues.
///
/// @param[in] beta
///         The eigenvalue scaling factors.
///
/// @param[in,out] A
///         Matrix A.
///
/// @param[in,out] B
///         Matrix B.
///
void block_placer(
    double *real, double *imag, double *beta, matrix_t A, matrix_t B);

#endif // STARNEIG_TEST_COMMON_BLOCK_PLACER_H
