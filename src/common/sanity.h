///
/// @file
///
/// @brief This file contains functions and macros that are used in the sanity
/// checks.
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

#ifndef STARNEIG_COMMON_SANITY
#define STARNEIG_COMMON_SANITY

#include <starneig_config.h>
#include <starneig/configuration.h>

#ifdef STARNEIG_ENABLE_SANITY_CHECKS

#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

///
/// @brief Reports a sanity check error. Aborts the program.
///
/// @param[in] message
///         Message.
///
#define STARNEIG_SANITY_REPORT(message) { \
    fprintf(stderr, "[starneig][sanity] %s:%d: %s\n", \
        __FILE__, __LINE__, message); \
    abort(); \
}

///
/// @brief Reports a sanity check error. Aborts the program.
///
/// @param[in] message
///         printf formatted message.
///
/// @param[in] ...
///         Additional printf compatible arguments.
///
#define STARNEIG_SANITY_REPORT_ARGS(message, ...) { \
    fprintf(stderr, "[starneig][sanity] %s:%d: %s\n", \
        __FILE__, __LINE__, message, __VA_ARGS__); \
    abort(); \
}

///
/// @brief Checks a conditional statement and reports a sanity check error if
/// the condition is not satisfied. Aborts the program.
///
/// @param[in] cond
///         The conditional statement.
///
/// @param[in] message
///         Message.
///
#define STARNEIG_SANITY_CHECK(cond, message) { \
    if (!(cond)) { \
        fprintf(stderr, "[starneig][sanity] %s:%d: %s\n", \
            __FILE__, __LINE__, message); \
        abort(); \
    } \
}

///
/// @brief Checks a conditional statement and reports a sanity check error if
/// the condition is not satisfied. Aborts the program.
///
/// @param[in] cond
///         The conditional statement.
///
/// @param[in] message
///         printf formatted message.
///
/// @param[in] ...
///         Additional printf compatible arguments.
///
#define STARNEIG_SANITY_CHECK_ARGS(cond, message, ...) { \
    if (!(cond)) { \
        fprintf(stderr, "[starneig][sanity] %s:%d: %s\n", \
            __FILE__, __LINE__, message, __VA_ARGS__); \
        abort(); \
    } \
}

static inline void starneig_sanity_check_inf(
    int rbegin, int rend, int cbegin, int cend, int ldA, double const *A,
    char const *mat, char const *file, int line)
{
    if (A == NULL)
        return;

    for (int i = cbegin; i < cend; i++) {
        for (int j = rbegin; j < rend; j++) {
            if (isinf(A[i*ldA+j])) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix %s has an infinite "
                    "element.\n", file, line, mat);
                abort();
            }
            if (isnan(A[i*ldA+j])) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix %s has a NaN element.\n",
                    file, line, mat);
                abort();
            }
        }
    }
}

#define STARNEIG_SANITY_CHECK_INF(rbegin, rend, cbegin, cend, ldA, A, mat) \
    starneig_sanity_check_inf(rbegin, rend, cbegin, cend, ldA, A, mat, \
        __FILE__, __LINE__)

static inline void starneig_sanity_check_multiplicities(
    int n, double *real, double *imag, char const *file, int line)
{
    for (int i = 0; i < n; i++) {
        double lambda_re = real[i];
        double lambda_im = imag[i];

        if (lambda_im == 0.0) { // real eigenvalue
            for (int j = i+1; j < n; j++) {
                if (real[j] == lambda_re && imag[j] == 0.0) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Multiple eigenvalues at "
                        "S(%d,%d) and S(%d,%d).\n", file, line, i, i, j, j);
                    abort();
                }
            }
        }
        else { // complex eigenvalue
            for (int j = i+1; j < n; j++) {
                if (real[j] == lambda_re && imag[j] == lambda_im) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Multiple eigenvalues at "
                        "S(%d:%d,%d:%d) and S(%d:%d,%d:%d).\n",
                        file, line, i, i+1, i, i+1, j, j+1, j, j+1);
                    abort();
                }
            }
        }
    }
}

///
/// @brief A sanity check that checks for multiple eigenvalues.
///
/// @param[in] n
///         The length of the vectors real and imag.
///
/// @param[in] real
///         A vector that contains the real parts of the eigenvalues.
///
/// @param[in] imag
///         A vector that contains the imaginary parts of the eigenvalues.
///
#define STARNEIG_SANITY_CHECK_MULTIPLICITIES(n, real, imag) \
    starneig_sanity_check_multiplicities(n, real, imag, __FILE__, __LINE__)

static inline void starneig_sanity_check_orthogonality(
    int n, int ldQ, double const *Q, char const *mat, char const *file,
    int line)
{
    if (Q == NULL)
        return;

    starneig_sanity_check_inf(0, n, 0, n, ldQ, Q, mat, file, line);

    size_t ldT;
    double *T = starneig_alloc_matrix(n, n, sizeof(double), &ldT);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0,
        Q, ldQ, Q, ldQ, 0.0, T, ldT);

    double dot = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dot += squ(T[i*ldT+j] - (i == j ? 1.0 : 0.0));

    double norm = ((long long)1<<52) * sqrt(dot)/sqrt(n);

    if (10000 < norm || isnan(norm)) {
        fprintf(stderr,
            "[starneig][sanity] %s:%d: Matrix %s is not orthogonal.\n",
            file, line, mat);
        fprintf(stderr,
            "[starneig][sanity] %s:%d: |%s %s^T - I| / |I| = %.0f u.\n",
            file, line, mat, mat, norm);
        abort();
    }

    starneig_free_matrix(T);
}

///
/// @brief A sanity check that makes sure a matrix Q is orthogonal.
///
/// @param[in] n
///         The order of the matrix Q.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] Q
///         The matrix Q.
///
/// @param[in] mat
///         A string that identifies the matrix Q.
///
#define STARNEIG_SANITY_CHECK_ORTHOGONALITY(n, ldQ, Q, mat) \
    starneig_sanity_check_orthogonality(n, ldQ, Q, mat, __FILE__, __LINE__)

struct starneig_sanity_check_args {
    double *A; size_t ldA;
    double *B; size_t ldB;
};

static inline struct starneig_sanity_check_args *
starneig_sanity_check_residuals_begin(
    int n, int ldQ, int ldZ, int ldA, int ldB, double const *Q, double const *Z,
    double const *A, double const *B, char const *file, int line)
{
    starneig_sanity_check_inf(0, n, 0, n, ldA, A, "A", file, line);
    starneig_sanity_check_inf(0, n, 0, n, ldB, B, "B", file, line);
    starneig_sanity_check_orthogonality(n, ldQ, Q, "Q", file, line);
    starneig_sanity_check_orthogonality(n, ldZ, Z, "Z", file, line);

    if (Z == NULL) {
        Z = Q; ldZ = ldQ;
    }

    struct starneig_sanity_check_args *ret =
        malloc(sizeof(struct starneig_sanity_check_args));

    size_t ldT;
    double *T = starneig_alloc_matrix(n, n, sizeof(double), &ldT);

    ret->A = starneig_alloc_matrix(n, n, sizeof(double), &ret->ldA);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
        Q, ldQ, A, ldA, 0.0, T, ldT);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0,
        T, ldT, Z, ldZ, 0.0, ret->A, ret->ldA);

    ret->B = NULL; ret->ldB = 0;
    if (B != NULL) {
        ret->B = starneig_alloc_matrix(n, n, sizeof(double), &ret->ldB);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
            Q, ldQ, B, ldB, 0.0, T, ldT);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0,
            T, ldT, Z, ldZ, 0.0, ret->B, ret->ldB);
    }

    starneig_free_matrix(T);

    starneig_sanity_check_inf(0, n, 0, n, ret->ldA, ret->A, "A", file, line);
    starneig_sanity_check_inf(0, n, 0, n, ret->ldB, ret->B, "B", file, line);

    return ret;
}

///
/// @brief The first part of a sanity check that makes sure a matrix pencil
/// Q (A,B) X^T and an updated matrix pencil ~Q (~A,~B) ~X^T are equivalent.
///
/// @param[in] name
///         An unique identifier.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Z.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] Q
///         The matrix Q.
///
/// @param[in] Z
///         The matrix Z.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B.
///
#define STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN( \
    name, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B) \
    struct starneig_sanity_check_args *name = \
    starneig_sanity_check_residuals_begin( \
        n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, __FILE__, __LINE__)

static inline void starneig_sanity_check_residuals_end(
    int n, int ldQ, int ldZ, int ldA, int ldB, double const *Q, double const *Z,
    double const *A, double const *B,
    struct starneig_sanity_check_args const *args, char const *file, int line)
{
    struct starneig_sanity_check_args *ret =
        starneig_sanity_check_residuals_begin(
            n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, file, line);

    int failure = 0;

    {
        double dot = 0.0;
        for(int i = 0; i < n; ++i )
            for(int j = 0; j < n; ++j )
                dot += squ(ret->A[ret->ldA*i+j] - args->A[args->ldA*i+j]);

        double a_dot = 0.0;
        for(int i = 0; i < n; ++i )
            for(int j = 0; j < n; ++j )
                a_dot += squ(args->A[args->ldA*i+j]);

        double norm = ((long long)1<<52) * sqrt(dot)/sqrt(a_dot);

        if (10000 < norm || isnan(norm)) {
            fprintf(stderr,
                "[starneig][sanity] %s:%d: Residual check failed for the "
                "matrix A.\n", file, line);
            fprintf(stderr,
                "[starneig][sanity] %s:%d: The norm was %.0f u.\n",
                file, line, norm);
            failure++;
        }
    }

    if (B != NULL) {
        double dot = 0.0;
        for(int i = 0; i < n; ++i )
            for(int j = 0; j < n; ++j )
                dot += squ(ret->B[ret->ldB*i+j] - args->B[args->ldB*i+j]);

        double a_dot = 0.0;
        for(int i = 0; i < n; ++i )
            for(int j = 0; j < n; ++j )
                a_dot += squ(args->B[args->ldB*i+j]);

        double norm = ((long long)1<<52) * sqrt(dot)/sqrt(a_dot);

        if (10000 < norm || isnan(norm)) {
            fprintf(stderr,
                "[starneig][sanity] %s:%d: Residual check failed for the "
                "matrix B.\n", file, line);
            fprintf(stderr,
                "[starneig][sanity] %s:%d: The norm was %.0f u.\n",
                file, line, norm);
            failure++;
        }
    }

    if (failure)
        abort();

    starneig_free_matrix(ret->A);
    starneig_free_matrix(ret->B);
    free(ret);
}

///
/// @brief The second part of a sanity check that makes sure a matrix pencil
/// Q (A,B) X^T and an updated matrix pencil ~Q (~A,~B) ~X^T are equivalent.
///
/// @param[in] name
///         An unique identifier.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of the matrix Q.
///
/// @param[in] ldZ
///         The leading dimension of the matrix Z.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] Q
///         The matrix Q.
///
/// @param[in] Z
///         The matrix Z.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B.
///
#define STARNEIG_SANITY_CHECK_RESIDUALS_END( \
    name, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B) \
    starneig_sanity_check_residuals_end( \
        n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, name, __FILE__, __LINE__); \
    if (name != NULL) { \
        starneig_free_matrix(name->A); \
        starneig_free_matrix(name->B); \
        free(name); \
        name = NULL; \
    }

///
/// @brief Skips the second part of a sanity check that makes sure a matrix
/// pencil Q (A,B) X^T and an updated matrix pencil ~Q (~A,~B) ~X^T are
/// equivalent.
///
/// @param[in] name
///         An unique identifier.
///
#define STARNEIG_SANITY_CHECK_RESIDUALS_SKIP(name) \
    if (name != NULL) { \
        starneig_free_matrix(name->A); \
        starneig_free_matrix(name->B); \
        free(name); \
        name = NULL; \
    }

static inline void starneig_sanity_check_bulges(
    int begin, int shifts, int n, int ldA, int ldB,
    double const *A, double const *B, char const *file, int line)
{
    starneig_sanity_check_inf(
        0, n, begin, begin+3*(shifts/2)+1, ldA, A, "A", file, line);
    starneig_sanity_check_inf(
        0, n, begin, begin+3*(shifts/2)+1, ldB, B, "B", file, line);

    if (n < begin+3*(shifts/2)+1) {
        fprintf(stderr,
            "[starneig][sanity] %s:%d: Matrix A has an invalid bulge.\n",
            file, line);
        abort();
    }

    double const *_A = A+begin*ldA+begin;
    double const *_B = B != NULL ? B+begin*ldB+begin : NULL;
    int _n = n - begin;
    for (int i = 0; i < shifts/2; i++) {
        for (int j = 3*i+4; j < _n; j++) {
            if (_A[3*i*ldA+j] != 0.0 || _A[(3*i+1)*ldA+j] != 0.0 ||
            _A[(3*i+2)*ldA+j] != 0.0) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix A has an invalid "
                    "bulge.\n", file, line);
                abort();
            }
        }
        for (int j = 3*i+3; _B != NULL && j < _n; j++) {
            if (_B[3*i*ldB+j] != 0.0 || _B[(3*i+1)*ldB+j] != 0.0 ||
            _B[(3*i+2)*ldB+j] != 0.0) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix B has invalid bulge .\n",
                    file, line);
                abort();
            }
        }
        if (_B != NULL &&
        (_B[3*i*ldB+3*i+1] != 0.0 || _B[3*i*ldB+3*i+2] != 0.0)) {
            fprintf(stderr,
                "[starneig][sanity] %s:%d: Matrix B has invalid bulge.\n",
                file, line);
            abort();
        }
    }
}

///
/// @brief A sanity check that makes sure that matrix pencil (A,B) contains the
/// correct number of bulges.
///
/// @param[in] begin
///         The column that should contain the first bulge.
///
/// @param[in] shifts
///         The number of shifts (shifts/2 bulges).
///
/// @param[in] n
///         The order of the matrices A and B.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B.
///
#define STARNEIG_SANITY_CHECK_BULGES(begin, shifts, n, ldA, ldB, A, B) \
    starneig_sanity_check_bulges( \
        begin, shifts, n, ldA, ldB, A, B, __FILE__, __LINE__)

static inline void starneig_sanity_check_schur(
    int begin, int end, int n, int ldA, int ldB,
    double const *A, double const *B, char const *file, int line)
{
    starneig_sanity_check_inf(0, n, begin, end, ldA, A, "A", file, line);
    starneig_sanity_check_inf(0, n, begin, end, ldB, B, "B", file, line);

    const double safmin = dlamch("S");

    int two_by_two = 0;
    for (int i = begin; i < end; i++) {
        if (i+1 < end && A[i*ldA+i+1] != 0.0) {
            if (two_by_two) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix A is not in Schur "
                    "form.\n", file, line);
                abort();
            }

            if (B != NULL) {
                if (B[(i+1)*ldB+i] != 0.0 || B[i*ldB+i+1] != 0.0) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Matrix pencil (A,B) "
                        "contains a non-normalized 2-by-2 block.\n",
                        file, line);
                    abort();
                }

                if (B[i*ldB+i] == 0.0 || B[(i+1)*ldB+i+1] == 0.0) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Matrix B is singular.",
                        file, line);
                    abort();
                }

                double s1, s2, wr1, wr2, wi;

                extern void dlag2_(double const *, int const *, double const *,
                    int const *, double const *, double *, double *,
                    double *, double *, double *);

                dlag2_(&A[i*ldA+i], &ldA, &B[i*ldB+i], &ldB, &safmin,
                    &s1, &s2, &wr1, &wr2, &wi);

                if (wi == 0.0) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Matrix pencil (A,B) "
                        "contains a fake 2-by-2 block.\n", file, line);
                    abort();
                }
            }
            else {
                if (A[i*ldA+i] != A[(i+1)*ldA+i+1]) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Matrix A contains a "
                        "non-normalized 2-by-2 block.\n",
                        file, line);
                    abort();
                }

                double a[] = {
                    A[i*ldA+i], A[i*ldA+i+1], A[(i+1)*ldA+i], A[(i+1)*ldA+i+1]
                };

                double rt1r, rt1i, rt2r, rt2i, cs, ss;

                extern void dlanv2_(
                    double *, double *, double *, double *, double *,
                    double *, double *, double *, double *, double *);

                dlanv2_(&a[0], &a[2], &a[1], &a[3],
                    &rt1r, &rt1i, &rt2r, &rt2i, &cs, &ss);

                if (rt1i == 0.0 || rt2i == 0.0) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Matrix A contains a fake "
                        "2-by-2 block.\n", file, line);
                    abort();
                }
            }

            two_by_two = 1;
        }
        else {
            two_by_two = 0;
        }

        for (int j = i+2; j < n; j++) {
            if (A[i*ldA+j] != 0.0) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix A is not in Schur "
                    "form.\n", file, line);
                abort();
            }
        }
    }

    if (B != NULL) {
        for (int i = begin; i < end; i++) {
            for (int j = i+2; j < n; j++) {
                if (B[i*ldB+j] != 0.0) {
                    fprintf(stderr,
                        "[starneig][sanity] %s:%d: Matrix B is not in upper "
                        "Hessenberg form.\n", file, line);
                    abort();
                }
            }
        }
    }
}

///
/// @brief A sanity check that makes sure that matrix pencil (A,B) is in Schur
/// form.
///
/// @param[in] begin
///         The first column to check.
///
/// @param[in] end
///         The last column to check + 1;
///
/// @param[in] n
///         The order of the matrices A and B.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B.
///
#define STARNEIG_SANITY_CHECK_SCHUR(begin, end, n, ldA, ldB, A, B) \
    starneig_sanity_check_schur( \
        begin, end, n, ldA, ldB, A, B, __FILE__, __LINE__)

static inline void starneig_sanity_check_hessenberg(
    int begin, int end, int n, int ldA, int ldB,
    double const *A, double const *B, char const *file, int line)
{
    starneig_sanity_check_inf(0, n, begin, end, ldA, A, "A", file, line);
    starneig_sanity_check_inf(0, n, begin, end, ldB, B, "B", file, line);

    for (int i = begin; i < end; i++) {
        for (int j = i+2; j < n; j++) {
            if (A[i*ldA+j] != 0.0) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix A is not in upper "
                    "Hessenberg form.\n", file, line);
                abort();
            }
        }
    }
    for (int i = begin; B != NULL && i < end; i++) {
        for (int j = i+1; j < n; j++) {
            if (B[i*ldB+j] != 0.0) {
                fprintf(stderr,
                    "[starneig][sanity] %s:%d: Matrix B is not in upper "
                    "triangular form.\n", file, line);
                abort();
            }
        }
    }
}

///
/// @brief A sanity check that makes sure that matrix pencil (A,B) is in
/// Hessenberg-triangular form.
///
/// @param[in] begin
///         The first column to check.
///
/// @param[in] end
///         The last column to check + 1;
///
/// @param[in] n
///         The order of the matrices A and B.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         The matrix A.
///
/// @param[in] B
///         The matrix B.
///
#define STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B) \
    starneig_sanity_check_hessenberg( \
        begin, end, n, ldA, ldB, A, B, __FILE__, __LINE__)

#else

#define STARNEIG_SANITY_REPORT(message) {}
#define STARNEIG_SANITY_REPORT_ARGS(message, ...) {}
#define STARNEIG_SANITY_CHECK(cond, message) {}
#define STARNEIG_SANITY_CHECK_ARGS(cond, message, ...) {}
#define STARNEIG_SANITY_CHECK_INF(rbegin, rend, cbegin, cend, ldA, A, mat) {}
#define STARNEIG_SANITY_CHECK_MULTIPLICITIES(n, real, imag) {}
#define STARNEIG_SANITY_CHECK_ORTHOGONALITY(n, ldQ, Q, mat) {}
#define STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN( \
    name, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B) {}
#define STARNEIG_SANITY_CHECK_RESIDUALS_END( \
    name, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B) {}
#define STARNEIG_SANITY_CHECK_RESIDUALS_SKIP(name) {}
#define STARNEIG_SANITY_CHECK_BULGES(begin, shifts, n, ldA, ldB, A, B) {}
#define STARNEIG_SANITY_CHECK_SCHUR(begin, end, n, ldA, ldB, A, B) {}
#define STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B) {}

#endif // STARNEIG_ENABLE_SANITY_CHECKS

#endif // STARNEIG_COMMON_SANITY
