///
/// @file
///
/// @brief This file contains code that is used in the Schur reduction CPU
/// codelets.
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

#ifndef STARNEIG_SCHUR_CPU_UTILS_H
#define STARNEIG_SCHUR_CPU_UTILS_H

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include <stddef.h>

///
/// @brief Deflates infinite eigenvalues from the top.
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] begin
///         The first row that belongs to the active region.
///
/// @param[in] end
///         The last row that belongs to the active region + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B.
///         On exit, the matrix ~B.
///
/// @return The first row that belongs to the remaining active region.
///
int starneig_deflate_inf_top(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B);

///
/// @brief Deflates infinite eigenvalues from the bottom.
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] begin
///         The first row that belongs to the active region.
///
/// @param[in] end
///         The last row that belongs to the active region + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B.
///         On exit, the matrix ~B.
///
/// @return The last row that belongs to the remaining active region + 1.
///
int starneig_deflate_inf_bottom(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B);

///
/// @brief Moves infinite eigenvalues top the top.
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] begin
///         The first row that belongs to the active region.
///
/// @param[in] end
///         The last row that belongs to the active region + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B.
///         On exit, the matrix ~B.
///
/// @param[in] deflate
///         If non-zero, then the infinite eigenvalues are deflated.
///
/// @return The number of moved infinite eigenvalues.
///
int starneig_push_inf_top(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B, int deflate);

///
/// @brief Reorders a matrix pencil Q (A,B) Z^T such that a single diagonal
/// block is moved from one location to another. Produces an updated matrix
/// pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] from
///         The original location of the diagonal block
///
/// @param[in] to
///         The location where the diagonal block is to be moved.
///
/// @param[in] n
///         The order of matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of matrix Z.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in] ldB
///         The leading dimension of matrix B.
///
/// @param[in] lwork
///         The size of the workspace buffer. If B != NULL, then the workspace
///         should be at least 4*n+16. Otherwise, the workspace should be at
///         least n.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @param[out] work
///         The workspace buffer.
///
/// @return The location where the diagonal block was actually moved.
///
int starneig_move_block(
    int from, int to, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double *Q, double *Z, double *A, double *B, double *work);

///
/// @brief Chases a set of bulges across a matrix pencil Q (A,B) Z^T. Produces
/// an updated matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] mode
///         The bulge chasing mode.
///
/// @param[in] shifts
///         The number of shifts to use.
///
/// @param[in] n
///         The order of matrices Q, Z, A and B.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Z.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] real
///         Shifts (real parts).
///
/// @param[in] imag
///         Shifts (imaginary parts).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
void starneig_push_bulges(
    bulge_chasing_mode_t mode, int shifts, int n,
    int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double const *real, double const *imag,
    double *Q, double *Z, double *A, double *B);

///
/// @brief Performs AED on a matrix pencil Q (A,B) Z^T. May produce an updated
/// matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] n
///         The order of matrices Q, Z, A and B.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Z.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[out] real
///         Returns the shifts (real parts).
///
/// @param[out] imag
///         Returns the shifts (imaginary parts).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q if 0 < converged.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z if 0 < converged.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A if 0 < converged.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B if 0 < converged.
///
/// @param[out] unconverged
///         Returns the number of unconverged eigenvalues / shifts
///
/// @param[out] converged
///         Returns the number of converged eigenvalues.
///
void starneig_aggressively_deflate(
    int n,  int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double *real, double *imag, double *Q, double *Z, double *A, double *B,
    int *unconverged, int *converged);

///
/// @brief Reduces a matrix pencil Q (A,B) Z^T to Schur form. Produces an
/// updated matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] n
///         The order of matrices Q, Z, A and B.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Z.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[out] real
///         Returns the eigenvalues (real parts).
///
/// @param[out] imag
///          Returns the eigenvalues (imaginary parts).
///
/// @param[out] beta
///          Returns the eigenvalues (beta parts).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @return The leftmost column that has been reduced to Schur form.
///
int starneig_schur_reduction(
    int n, int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double *real, double *imag, double *beta,
    double *Q, double *Z, double *A, double *B);

///
/// @brief Reduces a matrix pencil Q (A,B) Z^T to Hessenberg-triangular form.
/// Produces an updated matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] n
///         The order of matrices Q, Z, A and B.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Z.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @return The leftmost column that has been reduced to Hessenberg form.
///
int starneig_hessenberg_reduction(
    int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B);

///
/// @brief Extract eigenvalues from a matrix pencil (A,B).
///
/// @param[in] n
///         The order of the matrices A and B.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B. If NULL, then it is assumed that B = I.
///
/// @param[out] real
///         Returns the real parts of the eigenvalues.
///
/// @param[out] imag
///         Returns the imaginary parts of the eigenvalues.
///
/// @param[out] beta
///         Returns the beta parts of the eigenvalues.
///
void starneig_extract_eigenvalues(int n, int ldA, int ldB,
    double const *A, double const *B, double *real, double *imag, double *beta);

///
/// @brief Extracts shifts from a matrix pencil (A,B).
///
///  The smallest eigenvalues are reordered to the beginning of the buffers and
///  shuffled into pairs of real eigenvalues and pairs of complex conjugate
///  eigenvalues. Zero and infinite eigenvalues are removed.
///
/// @param[in] n
///         The order of the matrices A and B.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B. If NULL, then it is assumed that B = I.
///
/// @param[out] real
///         Returns the real parts of the shifts.
///
/// @param[out] imag
///         Returns the imaginary parts of the shifts.
///
/// @return The number of valid shifts.
///
int starneig_extract_shifts(int n, int ldA, int ldB,
    double const *A, double const *B, double *real, double *imag);

#endif // STARNEIG_SCHUR_CPU_UTILS_H
