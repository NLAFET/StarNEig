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

#ifndef STARNEIG_TESTS_COMMON_PARSE_H
#define STARNEIG_TESTS_COMMON_PARSE_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include <stdio.h>
#include <string.h>

#define READ_FROM_ARGV(                                                     \
    func_name, ret_type, parse_field, array_name, default_entry)            \
ret_type * func_name(                                                       \
    char const *name, int argc, char * const *argv, int *argr)              \
{                                                                           \
    int array_size = sizeof(array_name)/sizeof(array_name[0]);              \
    for (int i = 0; i < argc-1; i++) {                                      \
        if (strcmp(name, argv[i]) == 0) {                                   \
            for (int j = 0; j < array_size; j++) {                          \
                if (strcmp(array_name[j].parse_field, argv[i+1]) == 0) {    \
                    if (argr != NULL)                                       \
                        argr[i] = argr[i+1] = 1;                            \
                    return &array_name[j];                                  \
                }                                                           \
            }                                                               \
            return NULL;                                                    \
        }                                                                   \
    }                                                                       \
    if (0 <= default_entry && default_entry < array_size)                   \
        return &array_name[default_entry];                                  \
    return NULL;                                                            \
}

#define PRINT_AVAIL(                                                        \
    func_name, header, name_field, desc_field, array_name, default_entry)   \
void func_name()                                                            \
{                                                                           \
    int array_size = sizeof(array_name)/sizeof(array_name[0]);              \
    printf(                                                                 \
        "\n"                                                                \
        header "\n");                                                       \
    for (int i = 0; i < array_size; i++) {                                  \
        char const * str = i == default_entry ?                             \
            "    [%s] : %s\n" : "    '%s' : %s\n";                          \
        printf(str, array_name[i].name_field, array_name[i].desc_field);    \
    }                                                                       \
}

#define PRINT_OPT(                                                          \
    func_name, header, name_field, usage_field, array_name)                 \
void func_name()                                                            \
{                                                                           \
    int array_size = sizeof(array_name)/sizeof(array_name[0]);              \
    for (int i = 0; i < array_size; i++) {                                  \
        if (array_name[i].usage_field != NULL) {                            \
            printf("\n%s " header "\n", array_name[i].name_field);          \
            array_name[i].usage_field();                                    \
        }                                                                   \
    }                                                                       \
}

///
/// @brief Multi-value argument can be either an integer of a character string.
///
struct multiarg_t {
    enum {
        MULTIARG_INT,
        MULTIARG_FLOAT,
        MULTIARG_STR,
        MULTIARG_INVALID } type;         ///< Argument type
    union {
        int int_value;                   ///< Integer value
        double double_value;             ///< Floating-point value
        char const *str_value;           ///< Pointer to string value
    };
};

///
/// @brief Reads a string from the command line.
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an array that is used to track which command line
///                      arguments have been processed
/// @param[in] def - default value to be returned when the argument is not found
///
/// @return a pointer to the requested string
///
char const * read_str(char const *name, int argc, char * const *argv,
    int *argr, char const *def);

///
/// @brief Reads an integer from the command line.
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an that is used to track which command line
///                      arguments have been processed
/// @param[in] def - default value to be returned when the argument is not found
///
/// @return the requested integer
///
int read_int(
    char const *name, int argc, char * const *argv, int *argr, int def);

///
/// @brief Reads an unsigned integer from the command line
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an array that is used to track which command line
///                      arguments have been processed
/// @param[in] def - default value to be returned when the argument is not found
///
/// @return the requested unsigned integer
///
unsigned read_uint(char const *name, int argc, char * const *argv,
    int *argr, unsigned def);

///
/// @brief Reads a double precision floating-point number from the command line.
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an that is used to track which command line
///                      arguments have been processed
/// @param[in] def - default value to be returned when the argument is not found
///
/// @return the requested double precision floating-point number
///
double read_double(char const *name, int argc, char * const *argv,
    int *artg, double def);

///
/// @brief Reads a Boolean from the command line.
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an array that is used to track which command line
///                      arguments have been processed
///
/// @return the requested Boolean
///
int read_opt(char const *name, int argc, char * const *argv, int *argr);

///
/// @brief Reads a multi-valued argument from the command line.
///
///  The argument list should end with 0.
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[inout] argr - an array that is used to track which command line
///                      arguments have been processed
/// @param[in] ... - valid character string arguments
///
/// @return the requested multi-valued argument
///
struct multiarg_t read_multiarg(
    char const *name, int argc, char * const *argv, int *argr, ...);

///
/// @brief Prints a multi-valued argument.
///
///  The argument list should end with 0.
///
/// @param[in] name - argument name
/// @param[in] argc - command line argument count
/// @param[in] argv - command line arguments
/// @param[in] ... - valid character string arguments
///
void print_multiarg(char const *name, int argc, char * const *argv, ...);

#endif
