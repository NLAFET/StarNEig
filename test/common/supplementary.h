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

#ifndef STARNEIG_TEST_COMMON_SUPPLEMENTARY
#define STARNEIG_TEST_COMMON_SUPPLEMENTARY

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include <stddef.h>
#ifdef STARNEIG_ENABLE_MPI
#include <mpi.h>
#endif

///
/// @brief Supplementary data type enumerator.
///
typedef enum {
    SUPPLEMENTARY_SELECTED,          ///< eigenvalue selection vector
    SUPPLEMENTARY_EIGENVALUES,       ///< eigenvalues
    SUPPLEMENTARY_KNOWN_EIGENVALUES  ///< know eigenvalues
} supplementary_type_t;

///
/// @brief Supplementary data.
///
struct supplementary {
    supplementary_type_t type;  ///< type
    void *ptr;                  ///< pointer
    struct supplementary *next; ///< next supplementary
};

///
/// @brief Frees supplementary data.
///
/// @param[in,out] supp  supplementary data
///
void free_supplementary(struct supplementary *supp);

///
/// @brief Copies supplementary data.
///
/// @param[in,out] supp  supplementary data
///
/// @return copy of the supplementary data
///
struct supplementary * copy_supplementary(struct supplementary const *supp);

///
/// @brief Prints the supplementary data.
///
/// @param[in] supp          supplementary data
///
void print_supplementary(struct supplementary const *supp);

///
/// @brief Loads supplementary data from a file.
///
/// @param[in] begin
///         First entry to be read.
///
/// @param[in] end
///         Last entry to be read + 1.
///
/// @param[in] name
///         Filename.
///
/// @param[in,out] supp
///         Supplementary data.
///
void load_supplementary(
    int begin, int end, char const *name, struct supplementary **supp);

///
/// @brief Stores supplementary data into a file.
///
/// @param[in] name
///         Filename.
///
/// @param[in] supp
///         Supplementary data.
///
void store_supplementary(char const *name, struct supplementary *supp);

#ifdef STARNEIG_ENABLE_MPI

///
/// @brief Broadcasts the supplementary data.
///
/// @param[in]     root          root MPI node
/// @param[in]     communicator  MPI communicator
/// @param[in,out] supp          supplementary data
///
void broadcast_supplementary(
    int root, MPI_Comm communicator, struct supplementary **supp);

#endif

///
/// @brief Initialized a supplementary data of the type SUPPLEMENTARY_SELECTED.
///
/// @param[in]  size      eigenvalue selection vector size
/// @param[out] selected  returns the eigenvalue selection vector
/// @param[out] supp      supplementary data
///
void init_supplementary_selected(
    size_t size, int **selected, struct supplementary **supp);

///
/// @brief Returns the contest of a SUPPLEMENTARY_SELECTED type supplementary
/// data.
///
/// @param[in,out] supp  supplementary data
///
/// @return eigenvalue selection vector
///
int * get_supplementaty_selected(struct supplementary const *supp);

///
/// @brief Initialized a supplementary data of the type
/// SUPPLEMENTARY_EIGENVALUES.
///
/// @param[in]  size      eigenvalue vector size
/// @param[out] real      returns the eigenvalues vector (real parts)
/// @param[out] imag      returns the eigenvalues vector (imaginary part)
/// @param[out] beta      returns the "beta"
/// @param[out] supp      supplementary data
///
void init_supplementary_eigenvalues(
    size_t size, double **real, double **imag, double **beta,
    struct supplementary **supp);

///
/// @brief Returns the contest of a SUPPLEMENTARY_EIGENVALUES type supplementary
/// data.
///
/// @param[in,out] supp  supplementary data
/// @param[out]    real  returns eigenvalues (real parts)
/// @param[out]    imag  returns eigenvalues (imaginary parts)
/// @param[out]    beta  returns the "beta"
///
void get_supplementaty_eigenvalues(struct supplementary const *supp,
    double **real, double **imag, double **beta);

///
/// @brief Initialized a supplementary data of the type
/// SUPPLEMENTARY_KNOWN_EIGENVALUES.
///
/// @param[in]  size      eigenvalue vector size
/// @param[out] real      returns the eigenvalues vector (real parts)
/// @param[out] imag      returns the eigenvalues vector (imaginary part)
/// @param[out] beta      returns the eigenvalues vector (beta part)
/// @param[out] supp      supplementary data
///
void init_supplementary_known_eigenvalues(
    size_t size, double **real, double **imag, double **beta,
    struct supplementary **supp);

///
/// @brief Returns the contest of a SUPPLEMENTARY_KNOWN_EIGENVALUES type
/// supplementary data.
///
/// @param[in,out] supp  supplementary data
/// @param[out]    real  returns eigenvalues (real parts)
/// @param[out]    imag  returns eigenvalues (imaginary parts)
/// @param[out]    beta  returns the eigenvalues vector (beta part)
///
void get_supplementaty_known_eigenvalues(struct supplementary const *supp,
    double **real, double **imag, double **beta);

#endif
