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

#ifndef STARNEIG_TEST_COMMON_DATA_H
#define STARNEIG_TEST_COMMON_DATA_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include <stddef.h>

///
/// @brief Data type data type.
///
typedef unsigned data_type_t;

///
/// @brief Data type mask data type.
///
typedef unsigned data_mask_t;

///
/// @brief Single precision flag.
///
static const data_type_t PREC_FLOAT  = 0x000;

///
/// @brief Double precision flag.
///
static const data_type_t PREC_DOUBLE = 0x001;

///
/// @brief Real arithmetic flag.
///
static const data_type_t NUM_REAL    = 0x000;

///
/// @brief Complex arithmetic flag.
///
static const data_type_t NUM_COMPLEX = 0x002;

///
/// @brief Floating-point length mask.
///
static const data_mask_t PREC_MASK = 0x001;

///
/// @brief Arithmetic mask.
///
static const data_mask_t NUM_MASK  = 0x002;

///
/// @brief Tests a data type against a given mask and value.
///
/// @param[in] v     data type to be tested
/// @param[in] mask  mask that indicates which bytes should be tested
/// @param[in] test  value to be checked against
///
/// @return non-zero if the values matched, zero otherwise
///
static inline unsigned check_data_type_against_mask(
    data_type_t v, data_mask_t mask, data_type_t test)
{
    return ~(v ^ test) & mask;
}

///
/// @brief Tests a data type gains a given precision and arithmetic.
///
/// @param[in] v      data type to be tested
/// @param[in] prec   floating-point precision
/// @param[in] arith  arithmetic
///
/// @return non-zero if the values matched, zero otherwise
///
static inline unsigned check_data_type_against(
    data_type_t v, data_type_t prec, data_type_t arith)
{
    return check_data_type_against_mask(v, PREC_MASK, prec) &&
        check_data_type_against_mask(v, NUM_MASK, arith);
}

///
/// @brief Returns data type size.
///
/// @param[in] type  data type
///
/// @return data type size
///
static inline size_t data_type_size(data_type_t type)
{
    size_t size;
    if (check_data_type_against_mask(type, PREC_MASK, PREC_DOUBLE))
        size = sizeof(double);
    else
        size = sizeof(float);

    if (check_data_type_against_mask(type, NUM_MASK, NUM_COMPLEX))
        return 2*size;
    else
        return size;
}

#endif
