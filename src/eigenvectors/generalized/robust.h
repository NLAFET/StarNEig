///
/// @file
///
/// @brief Header file
///
/// @author Carl Christian K. Mikkelsen (spock@cs.umu.se), Umeå University
/// @author Angelika Beatrix Schwarz (angies@cs.umu.se), Umeå University
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

#ifndef STARNEIG_EIGVEC_GEN_ROBUST_H_
#define STARNEIG_EIGVEC_GEN_ROBUST_H_

#include <starneig_config.h>
#include <starneig/configuration.h>

// Overflow threshold Omega and 1/Omega
#define log2_Omega 1000

///
/// @brief Set the global overflow threshold Omega used by all robust codes
/// The overflow threshold is an integer power of 2.
///
/// @param[in] k integer
///
void starneig_eigvec_gen_initialize_omega(int k);

// --------------------------------------------------------------------------
// WARNING: Remember to assign values to Omega and OmegaInv.
// --------------------------------------------------------------------------
extern double Omega;
extern double OmegaInv;

///
/// @brief Computes the scaling necessary to prevent overflow in a scalar
/// division y = b/t
///
/// @param[in] b real number bounded by Omega
/// @param[in] t nonzero real number bounded by Omega
///
/// @return scaling alpha, such that (alpha*b)/t is bounded by Omega
///
double starneig_eigvec_gen_protect_division(double b, double t);

///
/// @brief Computes the scaling necessary to prevent overflow in a linear
/// update Y:=Y-T*X
///
/// @param[in] t upper bound of the infinity norm of the matrix T, t <= Omega
/// @param[in] x upper bound of the infinity norm of the matrix X, x <= Omega
/// @param[in] y upper bound of the infinity norm of the matrix Y, y <= Omega
///
/// @return scaling factor alpha, such that the calculation of
/// Y:=(alpha*Y) - T*(alpha*X) cannot exceed Omega
///
double starneig_eigvec_gen_protect_update(double t, double x, double b);

#endif // STARNEIG_EIGVEC_GEN_ROBUST_H_
