///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
///
/// Copyright (c) 2019, Umeå Universitet
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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "parse.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>

char const * read_str(
    char const *name, int argc, char * const *argv, int *argr, char const *def)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            if (argr != NULL)
                argr[i] = argr[i+1] = 1;
            return argv[i+1];
        }
    }
    return def;
}

int read_int(
    char const *name, int argc, char * const *argv, int *argr, int def)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            if (argr != NULL)
                argr[i] = argr[i+1] = 1;
            return atoi(argv[i+1]);
        }
    }
    return def;
}

unsigned read_uint(
    char const *name, int argc, char * const *argv, int *argr,
    unsigned def)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            if (argr != NULL)
                argr[i] = argr[i+1] = 1;
            return labs(atol(argv[i+1]));
        }
    }
    return def;
}

double read_double(char const *name, int argc, char * const *argv,
    int *argr, double def)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            if (argr != NULL)
                argr[i] = argr[i+1] = 1;
            return atof(argv[i+1]);
        }
    }
    return def;
}

int read_opt(char const *name, int argc, char * const *argv, int *argr)
{
    for (int i = 0; i < argc; i++) {
        if (strcmp(name, argv[i]) == 0) {
            if (argr != NULL)
                argr[i] = 1;
            return 1;
        }
    }
    return 0;
}

static int numbers_only(char const *s)
{
    if (s == NULL)
        return 0;

    if (*s != '-' && isdigit(*s) == 0)
        return 0;

    while (*(++s) != 0)
        if (isdigit(*s) == 0)
            return 0;

    return 1;
}

struct multiarg_t read_multiarg(
    char const *name, int argc, char * const *argv, int *argr, ...)
{
    struct multiarg_t ret;
    ret.type = invalid;
    ret.str_value = NULL;

    va_list vl;
    va_start(vl, argr);

    char *val = va_arg(vl, char*);
    char const *in = read_str(name, argc, argv, argr, val);

    while (val != NULL) {
        if (strcmp(val, in) == 0) {
            ret.type = str;
            ret.str_value = in;
            va_end(vl);
            return ret;
        }
        val = va_arg(vl, char*);
    }
    va_end(vl);

    if (numbers_only(in)) {
        ret.type = integer;
        ret.int_value = atoi(in);
    }

    return ret;
}

void print_multiarg(char const *name, int argc, char * const *argv, ...)
{
    va_list vl;
    va_start(vl, argv);

    char *val = va_arg(vl, char*);
    char const *in = read_str(name, argc, argv, NULL, val);

    while (val != NULL) {
        if (strcmp(val, in) == 0) {
            printf(" %s %s", name, val);
            va_end(vl);
            return;
        }
        val = va_arg(vl, char*);
    }
    va_end(vl);

    if (numbers_only(in)) {
        printf(" %s %d", name, atoi(in));
        return;
    }

    printf(" %s <unknown>", name);
}
