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

#ifndef STARNEIG_EIGVEG_GEN_UTILS_H_
#define STARNEIG_EIGVEG_GEN_UTILS_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

// Controls how many rows and columns are printed
#define maxrow 20
#define maxcol 12

///
/// @brief Prints double arrays nicely on the screen
///
/// @param[in] m number of rows of matrix A
/// @param[in] n number of columns of matrix A
/// @param[in] a array containing matrix A
/// @param[in] lda leading dimension of matrix A
/// @param[in] format valid C format specification
///
void starneig_eigvec_gen_ddm(
    int m, int n, double *a, size_t lda, char *format);

///
/// @brief Prints integer arrays nicely on the screen
///
/// @param[in] m  number of rows of matrix A
/// @param[in] n  number of columns of matrix A
/// @param[in] a  array containing matrix A
/// @param[in] lda  leading dimension of matrix A
/// @param[in] format  valid C format specification
///
void starneig_eigvec_gen_ddmi(int m, int n, int *a, size_t lda, char *format);

///
/// @brief  Fill a matrix with zeros
///
/// @param[in] m  number of rows of matrix
/// @param[in] n  number of columns of matrix
/// @param[in] a  array containing matrix
/// @param[in] lda  leading dimension of array a
///
void starneig_eigvec_gen_zeros(int m, int n, double *a, size_t lda);

///
/// @brief  Fill a matrix with ones
///
/// @param[in] m  number of rows of matrix
/// @param[in] n  number of columns of matrix
/// @param[in] a  array containing matrix
/// @param[in] lda  leading dimension of array a
///
void starneig_eigvec_gen_ones(int m, int n, double *a, size_t lda);

#endif // STARNEIG_EIGVEG_GEN_UTILS_H_
