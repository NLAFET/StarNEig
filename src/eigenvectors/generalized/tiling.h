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

#ifndef STARNEIG_EIGVEG_GEN_TILING_H_
#define STARNEIG_EIGVEG_GEN_TILING_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Removes duplicate entries from an *increasing* array
///
/// The duplicated entries of an increasing array c
/// ~~~
///   c[0] <= c[1] <= c[2] <= .... <= c[m-1]
/// ~~~
/// are overwritten leaving a strictly increasing array
/// ~~~
///   c[0] <  c[1] <  c[2] < ... < c[n-1].
/// ~~~
/// The number n of distinct entries are returned
///
/// @param[in] m - the length of the array
/// @param[in,out] l - the array
///
/// @return the number of distinct entries in the array
///
int starneig_eigvec_gen_remove_duplicates(int m, int *l);

///
/// @brief Maps the structure of a quasi-upper triangular matrix A.
///
/// The array l is filled such that l[0]=0 and
/// ~~~
///    l[j]=1   if A(j,j-1)!=0
///    l[j]=0   if A(j,j-1) =0
/// ~~~
/// for 0 < j < m.
///
/// @param[in] m  the dimension of the matrix A
/// @param[in] a  array containing the matrix A
/// @param[in] lda  the leading dimension of the array a
/// @param[out] l  array of length at least m
///
void starneig_eigvec_gen_find_left(int m, double *a, size_t lda, int *l);

///
/// @brief Counts the number of selected eigenvectors
///
/// @param[in] m  the dimension of the matrix
/// @param[in] l map of the quasi-upper triangular structure of the matrix
/// @param[in] select LAPACK style selection array
///
/// @return the number of selected eigenvalues
///
int starneig_eigvec_gen_count_selected(int m, int *l, int *select);

///
/// @brief Find the global index of every selected eigenvalue
///
/// @param[in] m  the dimension of the matrix
/// @param[in] l map of the quasi-upper triangular structure of the matrix
/// @param[in] select LAPACK style selection array
/// @param[out] map map[j] is global index of the jth selected eigenvector
///
/// @return the number n of selected eigenvalues
///
int starneig_eigvec_gen_find_selected(int m, int *l, int *select, int *map);

///
/// @brief Find tiling which covers quasi-upper triangular matrix without
/// splitting 2-by-2 blocks
///
/// ~~~
/// xxxxx|xxx|xxxxx|xxx|xxxx|  <--- a[0]
///  xxxx|xxx|xxxxx|xxx|xxxx|
///   xxx|xxx|xxxxx|xxx|xxxx|
///    xx|xxx|xxxxx|xxx|xxxx|
///    xx|xxx|xxxxx|xxx|xxxx|
/// -------------------------
///      |xxx|xxxxx|xxx|xxxx|  <--- a[1]
///      | xx|xxxxx|xxx|xxxx|
///      | xx|xxxxx|xxx|xxxx|
/// -------------------------
///      |   |xxxxx|xxx|xxxx|  <--- a[2]
///      |   |xxxxx|xxx|xxxx|
///      |   |  xxx|xxx|xxxx|
///      |   |   xx|xxx|xxxx|
///      |   |   xx|xxx|xxxx|
/// -------------------------
///      |   |     |xxx|xxxx|  <--- a[3]
///      |   |     |xxx|xxxx|
///      |   |     |  x|xxxx|
/// -------------------------
///      |   |     |   |xxxx|  <--- a[4]
///      |   |     |   |xxxx|
///      |   |     |   |  xx|
///      |   |     |   |  xx|
/// -------------------------
///                            <--- a[5]
/// ~~~
///
/// @param[in] m - the dimension of the matrix
/// @param[in] mb - the target tile size
/// @param[in] l - left looking array of length at least m describing the
/// quasi-upper triangular structure
/// @param[out] ap - array of length at least divceil(m,mb)+1.
///         On exit, ap[j] is the starting index of the jth partition.
///
/// @return the number M of diagonal tiles necessary
///
int starneig_eigvec_gen_practical_row_tiling(int m, int mb, int *l, int *ap);

///
/// @brief Find tile columns which reflects the users selection of eigenvalues
///
/// Example: no eigenvalues have been selected from the last tile column.
/// ~~~
/// b[0]
/// |   b[1]
/// |   | b[2]
/// |   | |     b[3]
/// |   | |     |
/// |   | |     |  b[4]=b[5]
/// |   | |     |  |
/// xxx|x|xxxxx|x||  <--- a[0]
///  xx|x|xxxxx|x||
///   x|x|xxxxx|x||
///    |x|xxxxx|x||
///    |x|xxxxx|x||
/// ---------------
///    |x|xxxxx|x||  <--- a[1]
///    | |xxxxx|x||
///    | |xxxxx|x||
/// ---------------
///    | |xxxxx|x||  <--- a[2]
///    | |xxxxx|x||
///    | |  xxx|x||
///    | |   xx|x||
///    | |   xx|x||
/// ---------------
///    | |     |x||  <--- a[3]
///    | |     |x||
///    | |     |x||
/// ---------------
///    | |     | ||  <--- a[4]
///    | |     | ||
///    | |     | ||
///    | |     | ||
/// ---------------
///                  <--- a[5]
/// ~~~
/// @param[in] m  the length of m
/// @param[in] select  LAPACK style selection array
/// @param[in] l  left looking array as generated by FindLeft()
/// @param[in] ap  indices of the tiles used to cover S, T
/// @param[in] M  number of tiles used to cover S, T
/// @param[out] bp  array of length at least M+1.
///         On exit, bp[j] is the starting index of the jth partition.
///
/// @return the number of selected eigenvalues
///
int starneig_eigvec_gen_induced_column_tiling(
    int m, int *select, int *l, int M, int *ap, int *cp);

///
/// @brief Find tile columns which cover the matrix of selected eigenvectors
/// without splitting two vectors representing the real and imaginary part of
/// a complex eigenvector
///
/// The kth tile column starts at cp[k] and cp[N]=n
///
/// ~~~
/// c[0]   c[1] c[2]
/// |      |    |
/// xxxxxx|xxxx|  <-- a[0]
///  xxxxx|xxxx|
///   xxxx|xxxx|
///    xxx|xxxx|
///    xxx|xxxx|
/// ------------
///    xxx|xxxx|  <-- a[2]
///     xx|xxxx|
///     xx|xxxx|
/// ------------
///     xx|xxxx|  <-- a[3]
///     xx|xxxx|
///       |xxxx|
///       | xxx|
///       | xxx|
/// ------------
///       |   x|  <-- a[4]
///       |   x|
///       |   x|
/// ------------
///       |    |  <-- a[5]
///       |    |
///       |    |
///       |    |
/// ------------
///               <-- a[6]
/// ~~~
/// @param[in] n  the number of selected eigenvalues
/// @param[in] nb  the target width for the tile columns
/// @param[in] map  the global indices of all selected eigenvalues
/// @param[in] l  left looking array as generated by FindLeft()
/// @param[out] cp  array of length at least divceil(n,nb)+1.
///         On exit, cp[j] is the starting index of the jth partition.
///
/// @return the number of tile columns, N <= divceil(n,nb)+1
///
int starneig_eigvec_gen_practical_column_tiling(
    int n, int nb, int *map, int *l, int *cp);

#endif // STARNEIG_EIGVEG_GEN_TILING_H_
