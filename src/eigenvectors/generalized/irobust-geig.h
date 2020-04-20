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

#ifndef STARNEIG_EIGVEC_GEN_IROBUST_GEIG_H_
#define STARNEIG_EIGVEC_GEN_IROBUST_GEIG_H_

#include <starneig_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Auxilliary routine for rescaling a tile of a matrix
///
/// @param[in] m  number of rows in tile A
/// @param[in] n  number of cols in tile A and number of rows of matrix SCAL
/// @param[in] k  number of cols in matrix SCAL
/// @param[in,out] a array containing tile A
/// @param[in] lda leading dimension of array a
/// @param[in] scal array contain matrix SCAL
/// @param[in] lds leading dimension of array scal
/// @param[in] idx the scaling factors of tile A are stored in SCAL(:,idx)
///
void starneig_eigvec_gen_int_consistent_scaling(
    int m, int n, int k, double *a, size_t lda, int *scal, size_t lds, int idx);

///
/// @brief Mini-block column norms and scalings
///
/// Computes the infinity norm of the mini-block columns of X, scaling if
/// necessary to prevent them from exceeding Omega.
///
/// @param[in] m number of rows of matrix X
/// @param[in] n number of columns of X
/// @param[in] alphai array dictating the column structure of X
///        If alpha[j]==0, then column j forms a mini-block column
///        if alpha[j]!=0, then columns j:j+1 form a mini-block column
/// @param[in] x array containing matrix X
/// @param[in] ldx leading dimension of array x
/// @param[in, out] xscal array of scaling factors for all columns of X.
///        On entry, xscal[j] is the original scaling factor of the jth column.
///        On exit,  xscal[j] is the updated scaling factor of the jth column.
/// @param[out] xnorm array
///        On exit, xnorm[j] is the infinity norm of the mini-block column
///        which contains the jth column.
///
void starneig_eigvec_gen_int_mini_block_column_norms_and_scalings(
    int m, int n, double *alphai, double *x, size_t ldx, int *xscal,
    double *xnorm);

///
/// @brief Robust solution of a single shifted quasi-upper triangular system
///
/// @param[in] m dimension of the problem
/// @param[in] s array containing quasi-upper triangular matrix S
/// @param[in] lds leading dimension of array s
/// @param[in] cs generalised column majorants for matrix S
/// @param[in] t array containing triangular matrix T
/// @param[in] ldt leading dimension of array t
/// @param[in] ct generalised column majorants for matrix T
/// @param[in] blocks offset of all mini-blocks along the diagonals of S, T
/// @param[in] numBlocks number of mini-blocks along the diagonals of S, T
/// @param[in] alphar shift is (alphar + i*alphai)/beta
/// @param[in] alphai shift is (alphar + i*alphai)/beta
/// @param[in] beta shift is (alphar + i*alphai)/beta
/// @param[in,out] f array containing the right-hand side/solution
/// @param[in] ldf leading dimension of array f
/// @param[in,out] scal array of scaling factors.
///         On entry, scal[j] applies to the jth column of RHS.
///         On entry, scal[j] applies to the jth column of the solution.
/// @param[in,out] norm array of upper bounds.
///         On entry, norm[j] bounds infinity norm of jth column of RHS.
///         On exit, norm[j] bounds infinity norm of jth column of solution.
/// @param[out] work scratch buffer of length at least MAX(4,m).
///         On exit, the original content has been overwritten
///
int starneig_eigvec_gen_int_robust_single_shift_solve(
    int m, double *s, size_t lds, double *cs, double *t, size_t ldt, double *ct,
	int *blocks, int numBlocks, double alphar, double alphai, double beta,
	double *f, size_t ldf, int *scal, double *norm, double *work);

///
/// @brief Robust solution of a multi-shifted quasi-upper triangular system
/// S*X*D-T*X*B = F
///
/// @param[in] m dimension of the problem
/// @param[in] n the number of columns and the number shifts
/// @param[in] s array containing quasi-upper triangular matrix S
/// @param[in] lds leading dimension of array s
/// @param[in] cs generalised column majorants for matrix S
/// @param[in] t array containing triangular matrix T
/// @param[in] ldt leading dimension of array t
/// @param[in] ct generalised column majorants for matrix T
/// @param[in] blocks offset of all mini-blocks along the diagonals of S, T
/// @param[in] numBlocks number of mini-blocks along the diagonals of S, T
/// @param[in] alphar real array
/// @param[in] alphai real array
/// @param[in] beta real array
/// @param[in,out] f
///         On entry, the right-hand side.
///         On exit, the solution.
/// @param[in] ldf leading dimension of array f
/// @param[in,out] scal array of scaling factors.
///         On entry, scaling factors for the columns of right hand-side.
///         On exit, scaling factor for the columns of the solution.
/// @param[in,out] norm array of mini-block column norms.
///         On entry, infinity norms of mini-block columns of RHS.
///         On exit, infinity norms of mini-block columns of solution.
/// @param[out] work scratch buffer of length at least MAX(4,m).
///         On exit, overwritten by intermediate calculations.
///
int starneig_eigvec_gen_int_robust_multi_shift_solve(
    int m, int n, double *s, size_t lds, double *cs, double *t, size_t ldt,
    double *ct, int *blocks, int numBlocks, double* alphar, double* alphai,
    double* beta, double *f, size_t ldf, int *scal, double *norm);

///
/// @brief Robust multi-shift linear update
/// \f$Y(:,j) = Y(:,j) - (beta[j]*S-(alphar[j] + i*alphai[j])*T)*X(:,j)\f$
/// in real arithmetic
///
/// @param[in] m  number of rows of S, T, Y
/// @param[in] n  number of shifts and number of columns of Y
/// @param[in] k  number of columns of S and T, number of rows of X
///
/// @param[in] s  array containing matrix S
/// @param[in] lds  leading dimension of array s
/// @param[in] snorm  infinity norm of matrix S
///
/// @param[in] t array containing matrix T
/// @param[in] ldt leading dimension of array t
/// @param[in] tnorm infinity norm of matrix T
///
/// @param[in] alphar array of length at least n
/// @param[in] alphai array of length at least n
/// @param[in] beta array of length at least n
///
/// @param[in] x array containing matrix X
/// @param[in] ldx leading dimension of x
/// @param[in] xscal array of scaling factors of mini-block columns of X
/// @param[in] xnorm array of infinity norms of mini-block columns of X
///
/// @param[in,out] y array containing matrix y
/// @param[in] ldy leading dimension of array y
/// @param[in,out] yscal array of scaling factors of mini-block columns of Y.
///         On entry, yscal[j] is the scaling of the jth column of Y.
///         On exit, yscal[j] is the new scaling factor of the jth column of Y.
/// @param[in,out] ynorm array of infinity norms of mini-block columns of Y.
///         On entry, yscal[j] is an upper bound for the mini-block column which
///         contains the jth column of Y.
///         On exit, yscal[j] is the infinity norm of the mini-block column
///         which contains the jth column.
///
int starneig_eigvec_gen_int_robust_multi_shift_update(
    int m, int n, int k, double *s, size_t lds, double snorm, double *t,
    size_t ldt, double tnorm, double *alphar, double *alphai, double *beta,
	 double *x, size_t ldx, int *xscal, double *xnorm, double *y, size_t ldy,
     int *yscal, double *ynorm);

///
/// @brief Solves for the relevant portion an m by n tile Y
///
/// @param[in] m number of rows in tile Y
/// @param[in] n number of columns int tile Y
///
/// @param[in] s array containing matrix S
/// @param[in] lds leading dimension of array s
/// @param[in] cs array of mini-block column majorants for S
///
/// @param[in] t array containing matrix T
/// @param[in] ldt leading dimension of array t
/// @param[in] ct array of mini-block column majorants for T
///
/// @param[in] blocks array mapping mini-block structure of S
/// @param[in] numBlocks number of mini-blocks of S
///
/// @param[in] alphar array containing real(alpha)
/// @param[in] alphai array containing imag(alpha)
/// @param[in] beta array containg beta
/// @param[in] map array mapping columns to selected eigenvalues
///
/// @param[in] ap0 global index, start of tiles S, T
/// @param[in] ap1 global index, end+1 of tiles S, T
/// @param[in] bp0 global index, start of work region
/// @param[in] bp1 global index, start of multi-shift region
/// @param[in] cp0 global index, start of tile Y
/// @param[in] cp1 global index, end+1 of tile Y
///
/// @param[in,out] y array containing tile Y
/// @param[in] ldy leading dimension of array y
/// @param[in,out] yscal array of scaling factors for columns of tile Y
/// @param[in,out] ynorm array of mini-block column norms of tile Y
///
/// @param[in,out] work workspace
///
void starneig_eigvec_gen_irobust_solve_task(
    int m, int n, double *s, size_t lds, double *cs, double *t, size_t ldt,
    double *ct, int *blocks, int numBlocks, double *alphar, double *alphai,
    double *beta, int *map, int ap0, int ap1, int bp0, int bp1, int cp0,
    int cp1, double *y, size_t ldy, int *yscal, double *ynorm, double *work);

///
/// @brief Updates relevant parts of tile Y using corresponding parts tile X.
///
/// The user's selection of eigenvalues induces a partitioning of Y which is
/// described using arrays: ap (rows) and bp (columns). There is also a
/// practical partitioning descriped using arrays: ap (rows) and cp (columns).
/// The indiced partitioning specifies which columns should be update. The
/// practical partitioning identifies which columns intercept the current tile
///
/// This is is an example where the active region starts inside the tile, i.e
/// cp0 < ap0. Therefore there are zero operations which should be avoided.
/// ~~~
///
///              cp0       ap0                 cp1
///               |         |                   |
///           ----------------------------------------------
///               |        |                    |
///               | do not |    update this     |
///               | update |      region        |
///               |        |                    |
///           ----------------------------------------------
///               <--- tile spans cp0:cp1-1 --->
/// ~~~
/// Updates are done from MAX(cp0,ap0) and to the end of the tile.
///
/// @param[in] m number of rows in tile Y
/// @param[in] n number of columns in tile Y
/// @param[in] k internal dimension in matrix matrix multiplication
///
/// @param[in] s array containing tile S
/// @param[in] lds leading dimension of array s
/// @param[in] snorm infinity norm of tile S
///
/// @param[in] t array containing tile T
/// @param[in] ldt leading dimension of array t
/// @param[in] tnorm infinity norm of tile T
///
/// @param[in] alphar array containing real(alpha)
/// @param[in] alphai array containing imag(alpha)
/// @param[in] beta array containg beta
///
/// @param[in] bp0 global column index, start of work region
/// @param[in] bp1 gloabl column index, start of multi-shift region
/// @param[in] cp0 global column index, start of tiles X, Y
/// @param[in] cp1 global column index, end+1 of tiles X, Y
///
/// @param[in] x array containing tile X
/// @param[in] ldx leading dimension of array x
/// @param[in] xscal array containing scaling factors for columns of X
/// @param[in] xnorm array containing mini-block column nors for columns of X
/// @param[in,out] y array containing tile Y
/// @param[in] ldy leading dimension of array y
/// @param[in,out] yscal array of scaling factors for columns of Y
/// @param[in,out] ynorm array of mini-block column nors for columns of Y
///
void starneig_eigvec_gen_irobust_update_task(
    int m, int n, int k, double *s, size_t lds, double snorm, double *t,
    size_t ldt, double tnorm, double *alphar, double *alphai, double *beta,
	int bp0, int bp1, int cp0, int cp1, double *x, size_t ldx, int *xscal,
    double *xnorm, double *y, size_t ldy, int *yscal, double *ynorm);

#endif // STARNEIG_EIGVEC_GEN_IROBUST_GEIG_H_
