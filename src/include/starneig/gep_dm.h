///
/// @file
///
/// @brief This file contains distributed memory interface functions for
/// generalized eigenvalue problems.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
/// @author Lars Karlsson (larsk@cs.umu.se), Umeå University
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

#ifndef STARNEIG_GEP_DM_H
#define STARNEIG_GEP_DM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

#ifndef STARNEIG_ENABLE_MPI
#error "This header should be included only when STARNEIG_ENABLE_MPI is defined."
#endif

#include <starneig/error.h>
#include <starneig/expert.h>
#include <starneig/distr_matrix.h>

///
/// @defgroup starneig_dm_gep Distributed Memory / Generalized EVP
///
/// @brief Functions for solving non-symmetric generalized eigenvalue problems
/// on distributed memory systems.
///
/// @{
///

///
/// @name Computational functions
/// @{
///

#ifdef STARNEIG_GEP_DM_HESSENBERGTRIANGULAR

///
/// @brief Computes a Hessenberg-triangular decomposition of a general matrix
/// pencil.
///
/// @attention This function is a wrapper for several ScaLAPACK subroutines.
/// The function exists if @ref STARNEIG_GEP_DM_HESSENBERGTRIANGULAR is defined.
///
/// @param[in,out] A
///         On entry, the general matrix \f$A\f$.
///         On exit, the upper Hessenberg matrix \f$H\f$.
///
/// @param[in,out] B
///         On entry, the general matrix \f$B\f$.
///         On exit, the upper triangular matrix \f$R\f$.
///
/// @param[in,out] Q
///         On entry, the orthogonal matrix \f$Q\f$.
///         On exit, the product matrix \f$Q * U_1\f$.
///
/// @param[in,out] Z
///         On entry, the orthogonal matrix \f$Z\f$.
///         On exit, the product matrix \f$Z * U_2\f$.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
///
starneig_error_t starneig_GEP_DM_HessenbergTriangular(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t B,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z);

#endif

///
/// @brief Computes a generalized Schur decomposition given a
/// Hessenberg-triangular decomposition.
///
/// @param[in,out] H
///         On entry, the upper Hessenberg matrix \f$H\f$.
///         On exit, the Schur matrix \f$S\f$.
///
/// @param[in,out] R
///         On entry, the upper triangular matrix \f$R\f$.
///         On exit, the upper triangular matrix \f$T\f$.
///
/// @param[in,out] Q
///         On entry, the orthogonal matrix \f$Q\f$.
///         On exit, the product matrix \f$Q * U_1\f$.
///
/// @param[in,out] Z
///         On entry, the orthogonal matrix \f$Z\f$.
///         On exit, the product matrix \f$Z * U_2\f$.
///
/// @param[out] real
///         An array of the same size as \f$H\f$ containing the real parts of
///         the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] imag
///         An array of the same size as \f$H\f$ containing the imaginary parts
///         of the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] beta
///         An array of the same size as \f$H\f$ containing the \f$\beta\f$
///         values of computed generalized eigenvalues.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
/// @ref STARNEIG_DID_NOT_CONVERGE if the QZ algorithm failed to converge.
///
starneig_error_t starneig_GEP_DM_Schur(
    starneig_distr_matrix_t H,
    starneig_distr_matrix_t R,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[]);

///
/// @brief Reorders selected generalized eigenvalues to the top left corner of a
/// generalized Schur decomposition.
///
/// @param[in,out] selected
///         The selection array.
///         On entry, the initial positions of the selected generalized
///         eigenvalues.
///         On exit, the final positions of all correctly placed selected
///         generalized eigenvalues. In case of failure, the number of 1's in
///         the output may be less than the number of 1's in the input.
///
/// @param[in,out] S
///         On entry, the Schur matrix \f$S\f$.
///         On exit, the updated Schur matrix \f$\hat{S}\f$.
///
/// @param[in,out] T
///         On entry, the upper triangular \f$T\f$.
///         On exit, the updates upper triangular matrix \f$\hat{T}\f$.
///
/// @param[in,out] Q
///         On entry, the orthogonal matrix \f$Q\f$.
///         On exit, the product matrix \f$Q * U_1\f$.
///
/// @param[in,out] Z
///         On entry, the orthogonal matrix \f$Z\f$.
///         On exit, the product matrix \f$Z * U_2\f$.
///
/// @param[out] real
///         An array of the same size as \f$S\f$ containing the real parts of
///         the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] imag
///         An array of the same size as \f$S\f$ containing the imaginary parts
///         of the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] beta
///         An array of the same size as \f$S\f$ containing the \f$\beta\f$
///         values of computed generalized eigenvalues.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
/// @ref STARNEIG_PARTIAL_REORDERING if the generalized Schur form is not
/// fully reordered.
///
/// @see starneig_GEP_DM_Select
///
starneig_error_t starneig_GEP_DM_ReorderSchur(
    int selected[],
    starneig_distr_matrix_t S,
    starneig_distr_matrix_t T,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[]);

#ifdef STARNEIG_GEP_DM_REDUCE

///
/// @brief Computes a (reordered) generalized Schur decomposition given a
/// general matrix pencil.
///
/// @attention This function uses several ScaLAPACK subroutines. The function
/// exists if @ref STARNEIG_GEP_DM_REDUCE is defined.
///
/// @param[in,out] A
///         On entry, the general matrix \f$A\f$.
///         On exit, the Schur matrix \f$S\f$.
///
/// @param[in,out] B
///         On entry, the general matrix \f$B\f$.
///         On exit, the upper triangular matrix \f$T\f$.
///
/// @param[in,out] Q
///         On entry, the orthogonal matrix \f$Q\f$.
///         On exit, the product matrix \f$Q * U_1\f$.
///
/// @param[in,out] Z
///         On entry, the orthogonal matrix \f$Z\f$.
///         On exit, the product matrix \f$Z * U_2\f$.
///
/// @param[out] real
///         An array of the same size as \f$A\f$ containing the real parts of
///         the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] imag
///         An array of the same size as \f$A\f$ containing the imaginary parts
///         of the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] beta
///         An array of the same size as \f$A\f$ containing the \f$\beta\f$
///         values of computed generalized eigenvalues.
///
/// @param[in] predicate
///         A function that takes a (complex) generalized eigenvalue as input
///         and returns non-zero if it should be selected. For complex conjugate
///         pairs of generalized eigenvalues, the predicate is called only for
///         the generalized eigenvalue with positive imaginary part and the
///         corresponding \f$2 \times 2\f$ block is either selected or
///         deselected. The reordering step is skipped if the argument is a NULL
///         pointer.
///
/// @param[in] arg
///         An optional argument for the predicate function.
///
/// @param[out] selected
///         The final positions of all correctly placed selected generalized
///         eigenvalues.
///
/// @param[out] num_selected
///         The number of selected generalized eigenvalues (a complex conjugate
///         pair is counted as two selected generalized eigenvalues).
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
/// @ref STARNEIG_DID_NOT_CONVERGE if the QZ algorithm failed to converge.
/// @ref STARNEIG_PARTIAL_REORDERING if the generalized Schur form is not
/// fully reordered.
///
starneig_error_t starneig_GEP_DM_Reduce(
    starneig_distr_matrix_t A,
    starneig_distr_matrix_t B,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[],
    int (*predicate)(double real, double imag, double beta, void *arg),
    void *arg,
    int selected[],
    int *num_selected);

#endif

///
/// @brief Computes a generalized eigenvector for each selected generalized
/// eigenvalue.
///
/// @param[in] selected
///         The selection array specifying the locations of the selected
///         generalized eigenvalues. The number of 1's in the array is the same
///         as the number of columns in \f$X\f$.
///
/// @param[in] S
///         The Schur matrix \f$S\f$.
///
/// @param[in] T
///         The upper triangular matrix \f$T\f$.
///
/// @param[in] Z
///         The orthogonal matrix \f$Z\f$.
///
/// @param[out] X
///         A matrix with \f$n\f$ rows and one column for each selected
///         generalized eigenvalue. The columns represent the computed
///         generalized eigenvectors as previously described.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
///
/// @see starneig_GEP_DM_Select
///
/// @todo This interface function is not implemented.
///
starneig_error_t starneig_GEP_DM_Eigenvectors(
    int selected[],
    starneig_distr_matrix_t S,
    starneig_distr_matrix_t T,
    starneig_distr_matrix_t Z,
    starneig_distr_matrix_t X);

///
/// @}
///

///
/// @name Helper functions
/// @{
///

///
/// @brief Generates a selection array for a Schur-triangular matrix pencil
/// using a user-supplied predicate function.
///
/// @param[in] S
///         The Schur matrix \f$S\f$.
///
/// @param[in] T
///         The upper triangular matrix \f$T\f$.
///
/// @param[in] predicate
///         A function that takes a (complex) generalized eigenvalue as input
///         and returns non-zero if it should be selected. For complex conjugate
///         pairs of generalized eigenvalues, the predicate is called only for
///         the generallized eigenvalue with positive imaginary part and the
///         corresponding \f$2 \times 2\f$ block is either selected or
///         deselected.
///
/// @param[in] arg
///         An optional argument for the predicate function.
///
/// @param[out] selected
///         The selection array. Both elements of a selected complex conjugate
///         pair are set to 1.
///
/// @param[out] num_selected
///         The number of selected generalized eigenvalues (a complex conjugate
///         pair is counted as two selected generalized eigenvalues).
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
///
starneig_error_t starneig_GEP_DM_Select(
    starneig_distr_matrix_t S,
    starneig_distr_matrix_t T,
    int (*predicate)(double real, double imag, double beta, void *arg),
    void *arg,
    int selected[],
    int *num_selected);

///
/// @}
///

///
/// @name Expert computational functions
/// @{
///

/// @brief Computes a generalized Schur decomposition given a
/// Hessenberg-triangular decomposition.
///
/// @param[in] conf
///         Configuration structure.
///
/// @param[in,out] H
///         On entry, the upper Hessenberg matrix \f$H\f$.
///         On exit, the Schur matrix \f$S\f$.
///
/// @param[in,out] R
///         On entry, the upper triangular matrix \f$R\f$.
///         On exit, the upper triangular matrix \f$T\f$.
///
/// @param[in,out] Q
///         On entry, the orthogonal matrix \f$Q\f$.
///         On exit, the product matrix \f$Q * U_1\f$.
///
/// @param[in,out] Z
///         On entry, the orthogonal matrix \f$Z\f$.
///         On exit, the product matrix \f$Z * U_2\f$.
///
/// @param[out] real
///         An array of the same size as \f$H\f$ containing the real parts of
///         the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] imag
///         An array of the same size as \f$H\f$ containing the imaginary parts
///         of the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] beta
///         An array of the same size as \f$H\f$ containing the \f$\beta\f$
///         values of computed generalized eigenvalues.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
///
/// @see starneig_GEP_DM_Schur
/// @see starneig_schur_conf
/// @see starneig_schur_init_conf
///
starneig_error_t starneig_GEP_DM_Schur_expert(
    struct starneig_schur_conf *conf,
    starneig_distr_matrix_t H,
    starneig_distr_matrix_t R,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[]);

///
/// @brief Reorders selected generalized eigenvalues to the top left corner of a
/// generalized Schur decomposition.
///
/// @param[in] conf
///         Configuration structure.
///
/// @param[in,out] selected
///         The selection array.
///
/// @param[in,out] S
///         On entry, the Schur matrix \f$S\f$.
///         On exit, the updated Schur matrix \f$\hat{S}\f$.
///
/// @param[in,out] T
///         On entry, the upper triangular \f$T\f$.
///         On exit, the updates upper triangular matrix \f$\hat{T}\f$.
///
/// @param[in,out] Q
///         On entry, the orthogonal matrix \f$Q\f$.
///         On exit, the product matrix \f$Q * U_1\f$.
///
/// @param[in,out] Z
///         On entry, the orthogonal matrix \f$Z\f$.
///         On exit, the product matrix \f$Z * U_2\f$.
///
/// @param[out] real
///         An array of the same size as \f$S\f$ containing the real parts of
///         the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] imag
///         An array of the same size as \f$S\f$ containing the imaginary parts
///         of the \f$\alpha\f$ values of the computed generalized eigenvalues.
///
/// @param[out] beta
///         An array of the same size as \f$S\f$ containing the \f$\beta\f$
///         values of computed generalized eigenvalues.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
///
/// @see starneig_GEP_DM_ReorderSchur
/// @see starneig_GEP_DM_Select
/// @see starneig_reorder_conf
/// @see starneig_reorder_init_conf
///
starneig_error_t starneig_GEP_DM_ReorderSchur_expert(
    struct starneig_reorder_conf *conf,
    int selected[],
    starneig_distr_matrix_t S,
    starneig_distr_matrix_t T,
    starneig_distr_matrix_t Q,
    starneig_distr_matrix_t Z,
    double real[], double imag[], double beta[]);

///
/// @brief Computes a generalized eigenvector for each selected generalized
/// eigenvalue.
///
/// @param[in] conf
///         Configuration structure.
///
/// @param[in] selected
///         The selection array specifying the locations of the selected
///         generalized eigenvalues. The number of 1's in the array is the same
///         as the number of columns in \f$X\f$.
///
/// @param[in] S
///         The Schur matrix \f$S\f$.
///
/// @param[in] T
///         The upper triangular matrix \f$T\f$.
///
/// @param[in] Z
///         The orthogonal matrix \f$Z\f$.
///
/// @param[out] X
///         A matrix with \f$n\f$ rows and one column for each selected
///         generalized eigenvalue. The columns represent the computed
///         generalized eigenvectors as previously described.
///
/// @return @ref STARNEIG_SUCCESS (0) on success. Negative integer -i when i'th
/// argument is invalid. Positive error code otherwise.
///
/// @see starneig_GEP_DM_Select
///
/// @todo This interface function is not implemented.
///
starneig_error_t starneig_GEP_DM_Eigenvectors_expert(
    struct starneig_eigenvectors_conf *conf,
    int selected[],
    starneig_distr_matrix_t S,
    starneig_distr_matrix_t T,
    starneig_distr_matrix_t Z,
    starneig_distr_matrix_t X);

///
/// @}
///

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_GEP_DM_H
