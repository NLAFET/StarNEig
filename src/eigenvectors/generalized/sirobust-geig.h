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

#ifndef STARNEIG_EIGVEG_GEN_SIROBUST_GEIG_H_
#define STARNEIG_EIGVEG_GEN_SIROBUST_GEIG_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Computes selected generalized eigenvectors from real Schur forms
///
/// @param[in] m the dimension of matrices S, T
/// @param[in] s array containing the matrix S
/// @param[in] lds leading dimension of array s
/// @param[in] t array containing the matrix T
/// @param[in] ldt leading dimension of array t
/// @param[in] select LAPACK style selection array of length at least m
/// @param[out] y array large enough to store an m by n matrix
/// @param[in] ldy leading dimension of y
/// @param[in] mb number of rows pr. block row of Y (target value)
/// @param[in] nb number of colums pr. block column of Y (target value)
///
/// The different tilings used will never split a 2-by-2 block.
/// Tiles are expanded/reduce by one row/column prevent the splitting of
/// a 2-by-2 block or the separationg of the real and imaginary part of a
/// complex eigenvector.
///
int starneig_eigvec_gen_sinew(
    int m, double *s, size_t lds, double *t, size_t ldt, int *select,
    double *y, size_t ldy, int mb, int nb);

#endif // STARNEIG_EIGVEG_GEN_SIROBUST_GEIG_H_
