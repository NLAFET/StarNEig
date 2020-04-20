///
/// @file This file contains general purpose hooks.
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

#ifndef STARNEIG_TESTS_COMMON_HOOKS_H
#define STARNEIG_TESTS_COMMON_HOOKS_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "hook_experiment.h"

extern const struct hook_t hessenberg_test;
extern const struct hook_descr_t default_hessenberg_test_descr;

extern const struct hook_t schur_test;
extern const struct hook_descr_t default_schur_test_descr;

extern const struct hook_t residual_test;
extern const struct hook_descr_t default_residual_test_descr;

extern const struct hook_t print_input_pencil;
extern const struct hook_descr_t default_print_input_pencil_descr;

extern const struct hook_t print_pencil;
extern const struct hook_descr_t default_print_pencil_descr;

extern const struct hook_t eigenvalues_test;
extern const struct hook_descr_t default_eigenvalues_descr;

extern const struct hook_t known_eigenvalues_test;
extern const struct hook_descr_t default_known_eigenvalues_descr;

extern const struct hook_t analysis_test;
extern const struct hook_descr_t default_analysis_descr;

#endif
