///
/// @file
///
/// @brief Overflow protection of basis arithmetic operations
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

#include <starneig_config.h>
#include <starneig/configuration.h>
#include "common.h"
#include "robust.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

// --------------------------------------------------------------------------
// WARNING: Remember to assign values to Omega and OmegaInv.
// --------------------------------------------------------------------------
double Omega;
double OmegaInv;

void starneig_eigvec_gen_initialize_omega(int k)
{
    Omega = pow(2, k); OmegaInv = pow(2, -k);
}

double starneig_eigvec_gen_protect_division(double b, double t)
{
    /* Returns a scaling alpha such that y = (alpha*b)/t can not overflow

          ASSUME: |b|, |t| bounded by Omega
          ENSURE: 1/Omega <= alpha <= 1
    */

    // Return value
    double alpha;

    // Auxiliary variables
    double aux;

    // Initialize the scaling factor
    alpha = 1.0;

    if (fabs(t)<OmegaInv) {
        aux=fabs(t)*Omega;
        if (fabs(b)>aux)
            alpha=aux/fabs(b);
    } else {
        if (fabs(t)<1) {
            aux=fabs(t)*Omega;
            if (fabs(b)>aux)
		alpha=1/fabs(b);
        }
    }
    return alpha;
}

double starneig_eigvec_gen_protect_update(double t, double x, double y)
{
    /* Returns a scaling alpha such that y := (alpha*y) - t*(alpha*x)
          can does not exceed Omega

          ASSUME: 0 <= t, x, y <= Omega
          ENSURE: 0.5*(1/Omega) <= alpha <= 1

    */

    // Return value
    double alpha;

    // Initialize the scaling factor
    alpha = 1.0;

    if (x <= 1) {
        if (t*x > Omega - y)
            alpha = 0.5;
    } else {
        if (t > (Omega - y)/x)
            alpha=0.5*(1/x);
    }
    return alpha;
}
