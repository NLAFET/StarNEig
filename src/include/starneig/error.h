///
/// @file
///
/// @brief This file contains the library error codes.
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

#ifndef STARNEIG_ERROR_H
#define STARNEIG_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

///
/// @defgroup starneig_error Error codes
///
/// @brief Interface function return values and error codes.
///
/// @{
///

///
/// @brief Interface function return value data type.
///
typedef int starneig_error_t;

///
/// @brief Success.
///
/// The interface function was executed successfully.
///
#define STARNEIG_SUCCESS                            0

///
/// @brief Reneric error.
///
///The interface function encountered a generic error.
///
#define STARNEIG_GENERIC_ERROR                      1

///
/// @brief Not initialized.
///
/// The library was not initialized when the interface function was called.
///
#define STARNEIG_NOT_INITIALIZED                    2

///
/// @brief Invalid configuration.
///
/// The interface function encountered an invalid configuration argument.
///
#define STARNEIG_INVALID_CONFIGURATION              3

///
/// @brief Invalid argument.
///
/// The interface function encountered an invalid argument.
///
#define STARNEIG_INVALID_ARGUMENTS                  4

///
/// @brief Invalid distributed matrix.
///
/// One or more of the involved distributed matrices have an invalid
/// distribution, invalid dimensions and/or an invalid distributed block size.
///
#define STARNEIG_INVALID_DISTR_MATRIX               5

///
/// @brief Did not converge.
///
/// The interface function encountered a situation where the QR/QZ algorithm did
/// not converge. The matrix (pair) may be partially in (generalized) Schur
/// form.
///
#define STARNEIG_DID_NOT_CONVERGE                   6

///
/// @brief Partial reordering.
///
/// The interface function failed to reorder the (generalized) Schur form. The
/// (generalized) Schur form may be partially reordered.
///
#define STARNEIG_PARTIAL_REORDERING                 7

///
/// @brief Close eigenvalues.
///
/// The interface function encountered a situation where two selected
/// eigenvalues were close to each other. The computed result may be inaccurate.
///
#define STARNEIG_CLOSE_EIGENVALUES                  8

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_ERROR_H
