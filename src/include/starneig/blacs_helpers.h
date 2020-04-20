///
/// @file
///
/// @brief This file contains various BLACS helper functions.
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

#ifndef STARNEIG_BLACS_HELPERS_H
#define STARNEIG_BLACS_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

#ifndef STARNEIG_ENABLE_MPI
#error "This header should be included only when STARNEIG_ENABLE_MPI is defined."
#endif

#ifndef STARNEIG_ENABLE_BLACS
#error "This header should be included only when STARNEIG_ENABLE_BLACS is defined."
#endif

#include <starneig/blacs_matrix.h>

///
/// @defgroup starneig_dm_blacs_helpers ScaLAPACK compatibility / BLACS helpers
///
/// @brief Data types and helper functions for BLACS.
///
/// @{
///

///
/// @name Query indeces
/// @{
///

///
/// @brief Data type for blacs_get() function query id.
///
typedef int starneig_blacs_query_id_t;

///
/// @brief Query id for getting the default system context.
///
#define STARNEIG_BLACS_GET_DEFAULT_CONTEXT 0

///
/// @}
///

///
/// @brief Queries process rank information.
///
/// @param[out] my_rank
///         An unique process id (rank).
///
/// @param[out] rank_count
///         The total number of processes (ranks) available.
///
void starneig_blacs_pinfo(int *my_rank, int *rank_count);

///
/// @brief Returns BLACS context's internal defaults.
///
/// @param[in] context
///         The BLACS context.
///
/// @param[in] query
///         The query id.
///
/// @return The internal default value that matches the given query id.
///
int starneig_blacs_get(
    starneig_blacs_context_t context, starneig_blacs_query_id_t query);

///
/// @brief Initializes a BLACS process grid.
///
/// @param[in] system_context
///         The system BLACS context to be used in creating the process grid.
///
/// @param[in] order
///         The process mapping order.
///         "R" : Use row-major natural ordering.
///         "C" : Use column-major natural ordering.
///         ELSE: Use row-major natural ordering.
///
/// @param[in] rows
///         The number of rows in the process grid.
///
/// @param[in] cols
///         The number of columns in the process grid.
///
/// @return A handle to the created BLACS context.
///
starneig_blacs_context_t starneig_blacs_gridinit(
    starneig_blacs_context_t system_context, char *order,
    int rows, int cols);

///
/// @brief Queries BLACS process grid information.
///
/// @param[in] context
///         The BLACS context.
///
/// @param[out] rows
///         The number of rows in the process grid.
///
/// @param[out] cols
///         The number of columns in the process grid.
///
/// @param[out] row
///         The row coordinate of the calling process.
///
/// @param[out] col
///         The column coordinate of the calling process.
///
void starneig_blacs_gridinfo(
    starneig_blacs_context_t context, int *rows, int *cols, int *row, int *col);

///
/// @brief Queries BLACS process grid coordinates.
///
/// @param[in] context
///         The BLACS context.
///
/// @param[in] process
///         The process id (rank).
///
/// @param[out] row
///          The row coordinate of the process.
///
/// @param[out] col
///         The column coordinate of the process.
///
void starneig_blacs_pcoord(
    starneig_blacs_context_t context, int process, int *row, int *col);

///
/// @brief Releases process grid specific resources.
///
/// @param[in] context
///         The BLACS context.
///
void starneig_blacs_gridexit(starneig_blacs_context_t context);

///
/// @brief Releases all contexts and related resources.
///
/// @param[in] cont
///         The continue flag.
///
void starneig_blacs_exit(int cont);

///
/// @brief Computes the number of matrix rows/columns owned by a given process.
///
/// @param[in] n
///         The number of rows/columns in the distributed matrix.
///
/// @param[in] nb
///         The block size.
///
/// @param[in] iproc
///         The coordinate of the process whose local array row or column is to
///         be determined.
///
/// @param[in] isrcproc
///         The coordinate of the process that possesses the first row or column
///         of the distributed matrix.
///
/// @param[in] nprocs
///         The total number processes over which the matrix is distributed.
///
/// @return The number of rows/columns owned by the process.
///
int starneig_blacs_numroc(
    int n, int nb, int iproc, int isrcproc, int nprocs);

///
/// @brief Computes the number of matrix rows/columns owned by a given process.
/// Deprecated.
///
/// @deprecated The starneig_numroc() function has been replaced with the
/// starneig_blacs_numroc() function. This function will be removed in a
/// future release of the library.
///
int starneig_numroc(int n, int nb, int iproc, int isrcproc, int nprocs);

///
/// @brief Initializes a BLACS descriptor.
///
/// @param[out] descr
///        The matrix descriptor.
///
/// @param[in] m
///        The number of rows in the matrix.
///
/// @param[in] n
///        The number of columns in the matrix.
///
/// @param[in] sm
///        The number of rows in a distributed block.
///
/// @param[in] sn
///        The number of columns in a distributed block.
///
/// @param[in] irsrc
///        The process grid row over which the first row is distributed.
///
/// @param[in] icsrc
///        The process grid column over which the first column is distributed.
///
/// @param[in] context
///        The BLACS context.
///
/// @param[in] ld
///        The local array leading dimension.
///
/// @return Zero if the initialization was successful, non-zero otherwise.
///
int starneig_blacs_descinit(
    struct starneig_blacs_descr *descr, int m, int n, int sm, int sn,
    int irsrc, int icsrc, starneig_blacs_context_t context, int ld);

///
/// @brief Initializes a BLACS descriptor. Deprecated.
///
/// @deprecated The starneig_descinit() function has been replaced with the
/// starneig_blacs_descinit() function. This function will be removed in a
/// future release of the library.
///
int starneig_descinit(
    struct starneig_blacs_descr *descr, int m, int n, int sm, int sn,
    int irsrc, int icsrc, starneig_blacs_context_t context, int ld);

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_BLACS_HELPERS_H
