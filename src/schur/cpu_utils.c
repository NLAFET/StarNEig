///
/// @file
///
/// @brief This file contains code that is used in the Schur reduction CPU
/// codelets.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
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
#include "cpu_utils.h"
#include "../common/common.h"
#include "../common/sanity.h"
#include "../common/math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define _A(i,j) A[(j)*ldA+(i)]
#define _B(i,j) B[(j)*ldB+(i)]
#define _Q(i,j) Q[(j)*ldQ+(i)]
#define _Z(i,j) Z[(j)*ldZ+(i)]

#define _A_offset(i,j) (A != NULL ? &_A(i,j) : NULL)
#define _B_offset(i,j) (B != NULL ? &_B(i,j) : NULL)
#define _Q_offset(i,j) (Q != NULL ? &_Q(i,j) : NULL)
#define _Z_offset(i,j) (Z != NULL ? &_Z(i,j) : NULL)

///
/// @brief Applies a 2 X 2 rotation to a matrix A from the left.
///
///  +-------++-------------------+
///  | c   s ||###################|
///  |-s   c ||###################|
///  +-------++-------------------+
///
/// @param[in] c
///         The diagonal element of the rotation matrix.
///
/// @param[in] s
///         The off-diagonal element of the rotation matrix.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in,out]
///         The matrix A.
///
inline static void lmul2rot(double c, double s, int n, int ldA, double *A)
{
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        double q1 = _A(0,i);
        double q2 = _A(1,i);
        _A(0,i)   =  c * q1 + s * q2;
        _A(1,i)   = -s * q1 + c * q2;
    }
}

///
/// @brief Applies a 2 X 2 rotation to a matrix A from the right.
///
///  +--+
///  |##|
///  |##|
///  |##|
///  |##|+-------+
///  |##|| c  -s |
///  |##|| s   c |
///  |##|+-------+
///  |##|
///  |##|
///  |##|
///  +--+
///
/// @param[in] c
///         The diagonal element of the rotation matrix.
///
/// @param[in] s
///         The off-diagonal element of the rotation matrix.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in,out]
///         The matrix A.
///
inline static void rmul2rot(double c, double s, int n, int ldA, double *A)
{
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        double q1 = _A(i,0);
        double q2 = _A(i,1);
        _A(i,0)   =  c * q1 + s * q2;
        _A(i,1)   = -s * q1 + c * q2;
    }
}

///
/// @brief Applies a 2 X 2 "reversed" rotation to a matrix A from the right.
///
///  +--+
///  |##|
///  |##|
///  |##|
///  |##|+-------+
///  |##|| -s  c |
///  |##||  c  s |
///  |##|+-------+
///  |##|
///  |##|
///  |##|
///  +--+
///
/// @param[in] c
///         The off-diagonal element of the rotation matrix.
///
/// @param[in] s
///         The diagonal element of the rotation matrix.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in,out]
///         The matrix A.
///
inline static void rmul2rrot(double c, double s, int n, int ldA, double *A)
{
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        double q1 = _A(i,0);
        double q2 = _A(i,1);
        _A(i,0)   = -s * q1 + c * q2;
        _A(i,1)   =  c * q1 + s * q2;
    }
}

///
/// @brief Applies a 2 X 2 reflector to a matrix A from the left.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in] v
///         The reflector in the form I - [ v[0]; v[1] ] * [ v[0], v[1] ]^T.
///
/// @param[in,out]
///         The matrix A.
///
inline static void lmul2ref(
    int n, int ldA, double const * restrict v, double * restrict A)
{
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        double s = v[0] * _A(0,i) + v[1] * _A(1,i);
        _A(0,i) -= s * v[0];
        _A(1,i) -= s * v[1];
    }
}

///
/// @brief Applied a 2 X 2 reflector to a matrix A from the right.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in] v
///         The reflector in the form I - [ v[0]; v[1] ] * [ v[0], v[1] ]^T.
///
/// @param[in,out]
///         The matrix A.
///
inline static void rmul2ref(
    int m, int ldA, double const * restrict v, double * restrict A)
{
    #pragma GCC ivdep
    for (int i = 0; i < m; i++) {
        double s = v[0] * _A(i,0) + v[1] * _A(i,1);
        _A(i,0) -= s * v[0];
        _A(i,1) -= s * v[1];
    }
}

///
/// @brief Applied a 3 X 3 reflector to a matrix A from the left.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in] v
///         The reflector in the form
///         I - [ v[0]; v[1]; v[2] ] * [ v[0], v[1]; v[2] ]^T.
///
/// @param[in,out]
///         The matrix A.
///
inline static void lmul3ref(
    int n, int ldA, double const * restrict v, double * restrict A)
{
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        double s = v[0] * _A(0,i) + v[1] * _A(1,i) + v[2] * _A(2,i);
        _A(0,i) -= s * v[0];
        _A(1,i) -= s * v[1];
        _A(2,i) -= s * v[2];
    }
}

///
/// @brief Applied a 3 X 3 reflector to a matrix A from the right.
///
/// @param[in] n
///         The order of matrix A.
///
/// @param[in] ldA
///         The leading dimension of matrix A.
///
/// @param[in] v
///         The reflector in the form
///         I - [ v[0]; v[1]; v[2] ] * [ v[0], v[1]; v[2] ]^T.
///
/// @param[in,out]
///         The matrix A.
///
inline static void rmul3ref(
    int m, int ldA, double const * restrict v, double * restrict A)
{
    #pragma GCC ivdep
    for (int i = 0; i < m; i++) {
        double s = v[0] * _A(i,0) + v[1] * _A(i,1) + v[2] * _A(i,2);
        _A(i,0) -= s * v[0];
        _A(i,1) -= s * v[1];
        _A(i,2) -= s * v[2];
    }
}

///
/// @brief Creates a rotation G = [c, s; -s, c] such that G * [x1; x2] is zero
/// in its second component.
///
/// @param[in] x1
///         The first component of the input vector.
///
/// @param[in] x2
///         The second component of the input vector.
///
/// @param[out] c
///         The diagonal element of the rotation matrix.
///
/// @param[out] s
///         The off-diagonal element of the rotation matrix.
///
/// @return The first component of the rotated vector.
///
static double create_rotation(double x1, double x2, double *c, double *s)
{
    extern void dlartg_(
        double const *, double const *, double*, double*, double*);
    double r;
    dlartg_(&x1, &x2, c, s, &r);
    return r;
}

///
/// @brief Moves an infinite eigenvalue upwards.
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] from
///         The original location of the infinite eigenvalue.
///
/// @param[in] to
///         The desired location of the infinite eigenvalue.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B.
///         On exit, the matrix ~B.
///
/// @param[in] deflate
///         If non-zero, then the infinite eigenvalue is deflated.
///
inline static void push_inf_up(
    int from, int to, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B, int deflate)
{
    if (B == NULL)
        return;

    for (int i = from; to < i; i--) {
        double r, c, s;

        //
        // x x x x x   x x x x x       x x X X x   x x X X x
        // x x x x x     x x x x       x x X X x     x X X x
        //   x x x x       x x x  ==>    x X X x       0 r x
        //     x x x         0 x           X X x         0 x  <-- i
        //       x x           x           # X x           x
        //

        r = create_rotation(_B(i-1, i-1), _B(i-1, i), &c, &s);

        rmul2rrot(c, s, MIN(n, i+2), ldA, _A_offset(0, i-1));
        rmul2rrot(c, s, i, ldB, _B_offset(0, i-1));
        rmul2rrot(c, s, n, ldZ, _Z_offset(0, i-1));

        _B(i-1, i) = r;
        _B(i-1, i-1) = 0.0;

        //
        // x x x x x   x x x x x       x x x x x   x x x x x
        // x x x x x     x x x x       x x x x x     x x x x
        //   x x x x       0 x x  ==>    x x x x       0 x x
        //     x x x         0 x           r X X         0 X  <-- i
        //     x x x           x           0 X X           X
        //

        if (i+1 < n) {
            r = create_rotation(_A(i, i-1), _A(i+1, i-1), &c, &s);
            lmul2rot(c, s, n-i, ldA, _A_offset(i, i));
            lmul2rot(c, s, n-i-1, ldB, _B_offset(i, i+1));
            rmul2rot(c, s, n, ldQ, _Q_offset(0, i));

            _A(i, i-1) = r;
            _A(i+1, i-1) = 0.0;
        }
    }

    if (deflate) {

        //
        // x x x x x   x x x x x       x x x x x   x x x x x
        // x x x x x     x x x x       x x x x x     x x x x
        //     x x x       0 x x  ==>      r X X       0 X X  <-- to
        //     x x x         x x           0 X X         X X
        //       x x           x             x x           x
        //

        double r, c, s;
        r = create_rotation(_A(to, to), _A(to+1, to), &c, &s);
        lmul2rot(c, s, n-to-1, ldA, _A_offset(to, to+1));
        lmul2rot(c, s, n-to-1, ldB, _B_offset(to, to+1));
        rmul2rot(c, s, n, ldQ, _Q_offset(0, to));

        _A(to, to) = r;
        _A(to+1, to) = 0.0;
    }
}

int starneig_deflate_inf_top(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B)
{
    if (B == NULL)
        return begin;

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    while (begin+1 < end && _B(begin,begin) == 0.0 && _A(begin+1,begin) == 0.0)
        begin++;

    int i = begin;
    while (i < end) {
        if (_B(i,i) == 0.0) {
            push_inf_up(i, begin++, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 1);
            i = begin;
            continue;
        }
        i++;
    }

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return begin;
}

///
/// @brief Moves an infinite eigenvalue downwards.
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] from
///         The original location of the infinite eigenvalue.
///
/// @param[in] to
///         The desired location of the infinite eigenvalue.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B.
///         On exit, the matrix ~B.
///
/// @param[in] deflate
///         If non-zero, then the infinite eigenvalue is deflated.
///
inline static void push_inf_down(
    int from, int to, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B, int deflate)
{
    if (B == NULL)
        return;

    for (int i = from; i < to; i++) {
        double r, c, s;

        //
        // x x x x x   x x x x x       x x x x x   x x x x x
        // x x x x x     0 x x x       X X X X X     0 r X X  <-- i
        //   x x x x       x x x  ==>  # X X X X       0 X X
        //     x x x         x x           x x x         x x
        //       x x           x             x x           x
        //

        r = create_rotation(_B(i, i+1), _B(i+1, i+1), &c, &s);
        lmul2rot(c, s, MIN(n, n-i+1), ldA, _A_offset(i, MAX(0, i-1)));
        lmul2rot(c, s, n-i-1, ldB, _B_offset(i, i+1));
        rmul2rot(c, s, n, ldQ, _Q_offset(0, i));

        _B(i, i+1) = r;
        _B(i+1, i+1) = 0.0;

        //
        // x x x x x   x x x x x       X X x x x   X X x x x
        // x x x x x     0 x x x       X X x x x     0 x x x  <-- i
        // x x x x x       0 x x  ==>  0 r x x x       0 x x
        //     x x x         x x           x x x         x x
        //       x x           x             x x           x
        //

        if (0 <= i-1) {
            r = create_rotation(_A(i+1, i-1), _A(i+1, i), &c, &s);
            rmul2rrot(c, s, i+1, ldA, _A_offset(0, i-1));
            rmul2rrot(c, s, i, ldB, _B_offset(0, i-1));
            rmul2rrot(c, s, n, ldZ, _Z_offset(0, i-1));

            _A(i+1, i) = r;
            _A(i+1, i-1) = 0.0;
        }
    }

    if (deflate) {

        //
        // x x x x x   x x x x x       x X X x x   x X X x x
        // x x x x x     x x x x       x X X x x     X X x x
        //   x x x x       0 x x  ==>    0 r x x       0 x x  <-- to
        //       x x         x x             x x         x x
        //       x x           x             x x           x
        //

        double r, c, s;
        r = create_rotation(_A(to, to-1), _A(to, to), &c, &s);
        rmul2rrot(c, s, to, ldA, _A_offset(0, to-1));
        rmul2rrot(c, s, to, ldB, _B_offset(0, to-1));
        rmul2rrot(c, s, n, ldZ, _Z_offset(0, to-1));

        _A(to, to) = r;
        _A(to, to-1) = 0.0;
    }
}

int starneig_deflate_inf_bottom(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B)
{
    STARNEIG_ASSERT_MSG(0, "starneig_deflate_inf_bottom might be broken!")

    if (B == NULL)
        return end;

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    // TODO: This might be broken!
    while (begin < end && _B(end-1,end-1) == 0.0 &&
    (end-1 == 0 || _A(end-1,end-2) == 0.0))
        end--;

    int i = end-1;
    while (begin <= i) {
        if (_B(i,i) == 0.0) {
            push_inf_down(i, --end, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 1);
            i = end-1;
            continue;
        }
        i--;
    }

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return end;
}

int starneig_push_inf_top(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B, int deflate)
{
    if (B == NULL)
        return begin;

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    int top = begin;
    if (!deflate)
        while (top < end && _B(top,top) == 0.0) top++;

    for (int i = top; i < end; i++) {
        if (_B(i,i) == 0.0) {
            push_inf_up(i++, top++, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, deflate);
            if (!deflate)
                top++;
        }
    }

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return top-begin;
}

///
/// @brief "Flips" an infinite eigenvalue over a bulge.
///
///    A               B                 A               B
///    x x x x x x     x x x x x x       x x x x x x     x x x x x x
///    x x x x x x       x x x x x       x x x x x x       x x x x x
///      x x x x x         x x x x  ==>    x x x x x         0 x x x
///      x x x x x         x x x x         x x x x x           0 x x
///      x x x x x             0 x         x x x x x             0 x  <-- i
///            x x               x           x x x x               x
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] i
///         The original location of the infinite eigenvalue.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B.
///         On exit, the matrix ~B.
///
inline static void push_inf_bulge(
    int i, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B)
{
    if (B == NULL)
        return;

    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    double r, c, s;

    //
    // x x x x x   x x x x x       x X X x x   x X X x x
    // x x x x x     x x x x       x X X x x     X X x x
    // x x x x x     x x x x  ==>  x X X x x     0 X x x
    // x x x x x         0 x       x X X x x         0 x  <- i
    //       x x           x             x x           x
    //

    r = create_rotation(_B(i-1, i-2), _B(i-1, i-1), &c, &s);

    rmul2rrot(c, s, MIN(n, i+1), ldA, _A_offset(0, i-2));
    rmul2rrot(c, s, i-1, ldB, _B_offset(0, i-2));
    rmul2rrot(c, s, n, ldZ, _Z_offset(0, i-2));

    _B(i-1, i-1) = r;
    _B(i-1, i-2) = 0.0;

    //
    // x x x x x   x x x x x       x x X X x   x x X X x
    // x x x x x     x x x x       x x X X x     x X X x
    // x x x x x       x x x  ==>  x x X X x       0 X x
    // x x x x x         0 x       x x X X x         0 x  <- i
    //       x x           x           # X x           x
    //

    r = create_rotation(_B(i-1, i-1), _B(i-1, i), &c, &s);

    rmul2rrot(c, s, MIN(n, i+2), ldA, _A_offset(0, i-1));
    rmul2rrot(c, s, i, ldB, _B_offset(0, i-1));
    rmul2rrot(c, s, n, ldZ, _Z_offset(0, i-1));

    _B(i-1, i) = r;
    _B(i-1, i-1) = 0.0;

    //
    // x x x x x   x x x x x       x X X x x   x X X x x
    // x x x x x     x x x x       x X X x x     0 X x x
    // x x x x x       0 x x  ==>  x X X x x       0 x x
    // x x x x x         0 x       x X X x x         0 x  <- i
    //     x x x           x         X X x x           x
    //

    r = create_rotation(_B(i-2, i-2), _B(i-2, i-1), &c, &s);

    rmul2rrot(c, s, MIN(n, i+2), ldA, _A_offset(0, i-2));
    rmul2rrot(c, s, i, ldB, _B_offset(0, i-2));
    rmul2rrot(c, s, n, ldZ, _Z_offset(0, i-2));

    _B(i-2, i-1) = r;
    _B(i-2, i-2) = 0.0;

    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);
}

///
/// @brief Normalizes a 2-by-2 block.
///
///  Takes a matrix pencil Q (A,B) Z^T as an input and produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] i
///         The location of the 2-by-2 block.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in,out] real
///         Eigenvalues (real parts). If NULL, then the parameter is ignored.
///
/// @param[in,out] imag
///          Eigenvalues (imaginary parts). If NULL, then the parameter is
///         ignored.
///
/// @param[in,out] beta
///          Eigenvalues (scaling factors). If NULL, then the parameter is
///         ignored.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
static void process_2x2_block(
    int i, int n, int ldQ, int ldZ, int ldA, int ldB,
    double *real, double *imag, double *beta, double *Q, double *Z,
    double *A, double *B)
{
    extern void dlanv2_(double *, double *, double *, double *, double *,
        double *, double *, double *, double *, double *);

    extern void dlagv2_(double *, const int *, double *, const int *, double *,
        double *, double *, double *, double *, double *, double *);

    double csl, snl, csr, snr;

    if (B != NULL) {
        double real__[2], imag__[2], beta__[2];
        double *_real = real == NULL ? real__ : real+i;
        double *_imag = imag == NULL ? imag__ : imag+i;
        double *_beta = beta == NULL ? beta__ : beta+i;

        dlagv2_(_A_offset(i,i), &ldA, _B_offset(i,i), &ldB,
            _real, _imag, _beta, &csl, &snl, &csr, &snr);
    }
    else {
        double real__[2], imag__[2];
        double *_real = real == NULL ? real__ : real+i;
        double *_imag = imag == NULL ? imag__ : imag+i;

        dlanv2_(&_A(i,i), &_A(i,i+1), &_A(i+1,i), &_A(i+1,i+1),
            _real, _imag, _real+1, _imag+1, &csl, &snl);
        csr = csl;
        snr = snl;
    }

    // update A
    lmul2rot(csl, snl, n-i-2, ldA, _A_offset(i,i+2));
    rmul2rot(csr, snr, i, ldA, _A_offset(0,i));

    // update B
    if (B != NULL) {
        lmul2rot(csl, snl, n-i-2, ldB, _B_offset(i,i+2));
        rmul2rot(csr, snr, i, ldB, _B_offset(0,i));
    }

    // update Q
    rmul2rot(csl, snl, n, ldQ, _Q_offset(0,i));

    // update Z
    if (Z != NULL && Z != Q)
        rmul2rot(csr, snr, n, ldZ, _Z_offset(0,i));
}

///
/// @brief Computes the first column of the matrix product
/// (A * B^-1 - l1 * I) * (A * B^-1 - l2 * I).
///
///  It is assumet that either
///     Im(l1) == Im(l2) == 0
///  or
///     Re(l1) == R1(l2) and Im(l1) == -Im(l2).
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] A
///         3 X 3 matrix A.
///
/// @param[in] B
///         3 X 3 matrix B. If NULL, then it is assumet that B = I.
///
/// @param[in] real
///         The real parts of l1 and l2.
///
/// @param[in] imag
///         The imaginary parts of l1 and l2.
///
inline static void create_bulge(
    int ldA, int ldB, double const *A, double const *B,
    double const *real, double const *imag, double *v)
{
    // LAPACK routine that is used to generate a bulge
    extern void dlaqr1_(int const *, double const *, int const *,
        double const *, double const *, double const *, double const *,
        double *);

    if (B != NULL) {
        if (_B(0,0) == 0.0 || _B(1,1) == 0.0) {
            v[0] = v[1] = v[2] = 0.0;
            return;
        }

        // z1 = A * B^-1 * e1
        double z1[2] = {
            _A(0,0) / _B(0,0),
            _A(1,0) / _B(0,0)
        };

        // t = B^-1 * z1
        double t[2] = {
            (z1[0]-_B(0,1)*z1[1]/_B(1,1))/_B(0,0),
            z1[1]/_B(1,1)
        };

        // z2 = A * t = A * B^-1 * z1 = (A * B^-1)^2 * e1
        double z2[3] = {
            _A(0,0) * t[0] + _A(0,1) * t[1],
            _A(1,0) * t[0] + _A(1,1) * t[1],
                             _A(2,1) * t[1]
        };

        v[0] = z2[0] - (real[0]+real[1]) * z1[0] +
            (real[0]*real[0]+imag[0]*imag[0]);
        v[1] = z2[1] - (real[0]+real[1]) * z1[1];
        v[2] = z2[2];
    }
    else {
        dlaqr1_((int[]){3}, A, &ldA, &real[0], &imag[0], &real[1], &imag[1], v);
    }

    STARNEIG_SANITY_CHECK(
        !isinf(v[0]) && !isnan(v[0]), "v[0] is not a real number.");
    STARNEIG_SANITY_CHECK(
        !isinf(v[1]) && !isnan(v[0]), "v[1] is not a real number.");
    STARNEIG_SANITY_CHECK(
        !isinf(v[2]) && !isnan(v[0]), "v[2] is not a real number.");
}

///
/// @brief Generates a Householder reflector H such that
/// H * A * e_1 = out:alpha * e_1.
///
///  The reflector is stored in a form
///  I - [ out:v[0]; out:v[1]; out:v[2]; ... ] *
///     [ out:v[0]; out:v[1]; out:v[2]; ... ]^T.
///
/// @param[in] n
///         The order of the matrices H and A.
///
/// @param[in] A
///         The first column of the matrix A.
///
/// @param[out] v
///         Returns the reflector in the documented format.
///
/// @return The first row of H * A * e_1.
///
inline static double create_left_reflector(int n, double const *A, double * v)
{
    // LAPACK routine that generates a real elementary reflector H
    extern void dlarfg_(int const *, double *, double *, int const *, double *);

    double alpha = A[0];
    for (int i = 1; i < n; i++)
        v[i] = A[i];

    double tau;
    dlarfg_(&n, &alpha, v+1, (int[]){1}, &tau);

    double stau = v[0] = sqrt(tau);
    for (int i = 1; i < n; i++)
        v[i] *= stau;

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
    STARNEIG_SANITY_CHECK(
        !isinf(alpha) && !isnan(alpha), "Alpha is not a real number.");
    for (int i = 0; i < n; i++)
        STARNEIG_SANITY_CHECK(!isinf(v[i]) && !isnan(v[i]),
            "Vector v contains a non-real number.");
#endif

    return alpha;
}

///
/// @brief Generates a Householder reflector H such that
/// B * H^T * e_1 = (return value) * e_1.
///
///  The reflector is stored in a form
///  I - * [ out:v[0]; out:v[1]; out:v[2]; ... ] *
///     [ out:v[0]; out:v[1]; out:v[2]; ... ]^T.
///
/// @param[in] n
///         The order of the matrices B.
///
/// @param[in] ldB
///         The leading dimension of the matrix B.
///
/// @param[in] B
///         The matrix B.
///
/// @param[out] v
///         Returns the reflector in the documented format.
///
/// @return The first row of B * H^T * e_1.
///
inline static double create_right_reflector(
    int n, int ldB, double const *B, double *v)
{
    extern void dgerq2_(int const *, int const *, double *, int const *,
        double *, double *, int *);

    extern void dormr2_(char const *, char const *, int const *, int const *,
        int const *, double const *, int const *, double const *, double *,
        int const *, double*, int *);

    // lB <- B
    double lB[n*n];
    for (int i = 0; i < n; i++)
        memcpy(lB+i*n, &_B(0,i), n*sizeof(double));

    int info, one = 1;
    double tau[n], work[n];

    // form lB = R * Q
    dgerq2_(&n, &n, lB, &n, tau, work, &info);

    if (info != 0)
        starneig_fatal_error(
            "An unrecoverable internal error occured while calling dgerq2.");

    // v <- e_1
    v[0] = 1.0;
    for (int i = 1; i < n; i++)
        v[i] = 0.0;

    // v <- Q^T v = Q^T * e_1
    dormr2_("L", "T", &n, &one, &n, lB, &n, tau, v, &n, work, &info);

    if (info != 0)
        starneig_fatal_error(
            "An unrecoverable internal error occured while calling dormr2.");

    STARNEIG_SANITY_CHECK(
        !isinf(lB[0]) && !isnan(lB[0]), "lB[0] is not a real number.");

    // generate reflector H v = e1 and return lB(0,0) / tau
    return lB[0]/create_left_reflector(n, v, v);
}

///
/// @brief If possible, sets a sub-diagonal entry of a matrix to zero.
///
/// @param[in] threshold
///         The threshold.
///
/// @param[in] j
///         Column.
///
/// @param[in] n
///         The order of the matrix A.
///
/// @param[in] ldA
///         The leading dimension of the matrix A.
///
/// @param[in,out] A
///         The matrix A.
///
static void vigilant_deflation_check(
    double threshold, int j, int n, int ldA, double *A)
{
    if (0.0 < threshold) {
        if (_A(j+1,j) != 0.0 && fabs(_A(j+1,j)) < threshold) {
            starneig_verbose("A vigilant deflation occured.");
            _A(j+1,j) = 0.0;
        }
    }
    else {
        const double ulp = dlamch("Precision");
        const double safmin = dlamch("Safe minimum");
        double smlnum = safmin*(n/ulp);

        if (0 <= j && j+1 < n && _A(j+1,j) != 0.0) {
            double tst1 = fabs(_A(j,j)) + fabs(_A(j+1,j+1));
            if (tst1 == 0.0) {
                if (0 <= j-1)
                    tst1 += fabs(_A(j,j-1));
                if (0 <= j-2)
                    tst1 += fabs(_A(j,j-2));
                if (0 <= j-3)
                    tst1 += fabs(_A(j,j-3));
                if (j+2 < n)
                    tst1 += fabs(_A(j+2,j+1));
                if (j+3 < n)
                    tst1 += fabs(_A(j+3,j+1));
                if (j+4 < n)
                    tst1 += fabs(_A(j+4,j+1));
            }

            if (fabs(_A(j+1,j)) <= MAX(smlnum, ulp*tst1)) {
                double h12 = MAX(fabs(_A(j+1,j)), fabs(_A(j,j+1)));
                double h21 = MIN(fabs(_A(j+1,j)), fabs(_A(j,j+1)));
                double h11 = MAX(fabs(_A(j+1,j+1)), fabs(_A(j,j) - _A(j+1,j+1)));
                double h22 = MIN(fabs(_A(j+1,j+1)), fabs(_A(j,j) - _A(j+1,j+1)));
                double scl = h11 + h12;
                double tst2 = h22*(h11/scl);
                if (tst2 == 0.0 || h21*(h12/scl) <= MAX(smlnum, ulp*tst2)) {
                    starneig_verbose("A vigilant deflation occured.");
                    _A(j+1,j) = 0.0;
                }
            }
        }
    }
}

///
/// @brief Small bulge chasing kernel.
///
///  Chases a set of bulges across a matrix pencil Q (A,B) Z^T. Produces an
///  updated matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] mode
///         The bulge chasing mode.
///
/// @param[in] shifts
///         The number of shifts to use.
///
/// @param[in] n
///         The order of matrices Q, Z, A and B.
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
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] real
///         Shifts (real parts).
///
/// @param[in] imag
///         Shifts (imaginary parts).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
static void process_small_window(
    bulge_chasing_mode_t mode, int shifts, int n,
    int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double const * restrict real, double const * restrict imag,
    double * restrict Q, double * restrict Z, double * restrict A,
    double * restrict B)
{
#ifdef STARNEIG_ENABLE_SANITY_CHECKS
    STARNEIG_SANITY_CHECK(2 <= shifts, "Too few shifts.");
    STARNEIG_SANITY_CHECK(shifts % 2 == 0, "Odd number of shifts.");
    STARNEIG_SANITY_CHECK(mode == BULGE_CHASING_MODE_FULL ||
        3*(shifts/2)+1 <= n, "All bulges do not fit into the window.");

    for (int i = 0; i < shifts; i++)
        STARNEIG_SANITY_CHECK(
            real[i] != 0.0 || imag[i] != 0, "Some shifts are zero.");

    for (int i = 0; i < shifts; i++)
        STARNEIG_SANITY_CHECK(!isinf(real[i]) && !isinf(imag[i]),
            "Some shifts are infinite.");

    for (int i = 0; i < shifts; i++)
        STARNEIG_SANITY_CHECK(!isnan(real[i]) && !isnan(imag[i]),
            "Some shifts are NaNs.");

    for (int i = 0; i < shifts/2; i++)
        STARNEIG_SANITY_CHECK(
            imag[2*i] == -imag[2*i+1], "The shifts are not ordered correctly.");

    if (mode == BULGE_CHASING_MODE_CHASE ||
    mode == BULGE_CHASING_MODE_FINALIZE) {
        STARNEIG_SANITY_CHECK_BULGES(0, shifts, n, ldA, ldB, A, B);
        STARNEIG_SANITY_CHECK_HESSENBERG(3*(shifts/2), n, n, ldA, ldB, A, B);
    }
    else {
        STARNEIG_SANITY_CHECK_HESSENBERG(0, n, n, ldA, ldB, A, B);
    }
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);
#endif

    int introduce =
        mode == BULGE_CHASING_MODE_FULL || mode == BULGE_CHASING_MODE_INTRODUCE;
    int finalize =
        mode == BULGE_CHASING_MODE_FULL || mode == BULGE_CHASING_MODE_FINALIZE;

    int left; // the leftmost bulge starts from column `left`
    if (introduce)
        //
        // L                L.v
        //  +-------------   +-------------
        //  |x x x x x x x   |x x x x x x x
        //  |x x x x x x x   |x x x x x x x
        //  |  x x x x x x   |x x x x x x x
        //  |    x x x x x   |x x x x x x x
        //  |      x x x x   |      x x x x
        //  |        x x x   |        x x x
        //  |          x x   |          x x
        //
        left = 2-3*(shifts/2);
    else
        //
        //   L                L.v
        //  +-------------   +-------------
        //  |x x x x x x x   |x x x x x x x
        //  |x x x x x x x   |x x x x x x x
        //  |x x x x x x x   |  x x x x x x
        //  |x x x x x x x   |  x x x x x x
        //  |      x x x x   |  x x x x x x
        //  |      x x x x   |        x x x
        //  |      x x x x   |        x x x
        //
        left = 0;

    int right; // the leftmost bulge stops at column `right`
    if (finalize)
        //
        //     ..v R              ..v
        // x x x x x x|     x x x x x x|
        // x x x x x x|       x x x x x|
        // x x x x x x|       x x x x x|
        //       x x x|       x x x x x|
        //       x x x|             x x|
        // -----------+     -----------+
        //
        right = n-2;
    else
        //
        // v R                ..v
        // x x x x x x x x|   x x x x x x x x|
        // x x x x x x x x|   x x x x x x x x|
        // x x x x x x x x|     x x x x x x x|
        // x x x x x x x x|     x x x x x x x|
        //       x x x x x|     x x x x x x x|
        //       x x x x x|           x x x x|
        //       x x x x x|           x x x x|
        //             x x|           x x x x|
        // ---------------+   ---------------+
        //
        right = n-1 - 3*(shifts/2);

    //
    // deflate infinite eigenvalues
    //

    if (B != NULL) {
        if (introduce) {
            // we can savely deflate infinite eigenvalues from the top
            starneig_push_inf_top(
                0, n-1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 1);
        }
        else {
            // locate the part of the window that is in Hessenberg-triangular
            // form
            int i = -1;
            while (i+4 < n && _A(i+3,i+1) == 0.0 && _A(i+4,i+1) == 0.0 &&
            _B(i+2,i+1) == 0.0 && _B(i+3,i+2) == 0.0)
                i++;

            if (0 < i) {
                // if infinite eigenvalues can be deflated, ...
                if (_A(1,0) == 0.0) {
                    // deflate them
                    starneig_push_inf_top(
                        1, i, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 1);
                }
                // otherwise, ...
                else {
                    // gather them to the upper left corner
                    int j = 0;
                    while (j < i && _B(j,j) == 0.0)
                        j++;
                    starneig_push_inf_top(
                        j, i, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 0);
                }
            }
        }
    }

    //
    // chase all bulges across the window
    //

    int begin = left;
    while (begin < right) {

        //
        // push all bulges forward one column
        //
        for (int i = shifts/2-1; 0 <= i; i--) {
            int j = begin + 3*i;

            if (introduce && j < -1)
                break;       // the bulge does not yet fit inside the window
            if (n-2 <= j)
                continue;    // the bulge has left the window

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
            int zeros[n];
            memset(zeros, 0, sizeof(zeros));
            for (int k = 0; k < n-1; k++) {
                if (k+1 < n && 0 <= k-2 && _A(k+1,k-2) != 0.0)
                    continue;
                if (k+1 < n && 0 <= k-1 && _A(k+1,k-1) != 0.0)
                    continue;
                if (k+1 < n &&             _A(k+1,k)   != 0.0)
                    continue;
                if (k+2 < n && 0 <= k-1 && _A(k+2,k-1) != 0.0)
                    continue;
                if (k+2 < n &&             _A(k+2,k)   != 0.0)
                    continue;
                if (k+3 < n &&             _A(k+3,k)   != 0.0)
                    continue;
                zeros[k] = 1;
            }
#endif

            //
            // try to deflate the first two columns
            //

            if (j == -1) {

                // this is "the column left of the first column" and the matrix
                // pencil is in Hessenberg-triangular form

                vigilant_deflation_check(thres_a, 1, n, ldA, A);
                if (_A(2,1) == 0.0) {
                    //
                    //  A              B
                    // +-----------    +-----------
                    // |x x x x x x    |x x x x x x
                    // |? x x x x x    |  x x x x x
                    // |  0 x x x x    |    x x x x
                    // |    x x x x    |      x x x
                    // |      x x x    |        x x
                    // |        x x    |          x
                    //
                    // skip two 1x1 or a 2x2 block in the top left corner
                    process_2x2_block(
                        0, n, ldQ, ldZ, ldA, ldB, NULL, NULL, NULL, Q, Z, A, B);
                    goto skip_column;
                }
                else {
                    vigilant_deflation_check(thres_a, 0, n, ldA, A);
                    if (_A(1,0) == 0.0)
                        //
                        //  A              B
                        // +-----------    +-----------
                        // |x x x x x x    |x x x x x x
                        // |0 x x x x x    |  x x x x x
                        // |  x x x x x    |    x x x x
                        // |    x x x x    |      x x x
                        // |      x x x    |        x x
                        // |        x x    |          x
                        //
                        // skip a 1x1 block in the top left corner
                        goto skip_column;
                }
            }
            else if (_A(j+2,j) == 0.0 && (n <= j+3 || _A(j+3,j) == 0.0)) {

                //
                //  A              B
                // +-----------    +-----------
                // |x x x x x x    |x x x x x x
                // |x j x x x x    |  j x x x x
                // |  ? x x x x    |    x x x x
                // |  0 x x x x    |    ? x x x
                // |  0 x x x x    |        x x
                // |        x x    |          x
                //

                vigilant_deflation_check(thres_a, j, n, ldA, A);
                if (B != NULL)
                    vigilant_deflation_check(thres_b, j+1, n, ldB, B);
            }

            if (j == n-3 || (j+3 < n && ((j < 0 || _A(j+3,j) == 0.0) &&
            _A(j+3,j+1) == 0.0 && _A(j+3,j+2) == 0.0))) {

                //
                // A           B           A               B
                // x x x x     x x x x     x x x x x x     x x x x x x
                // x j x x       j x x     x j x x x x       j x x x x
                //   x x x         x x       x x x x x         x x x x
                //   x x x         ? x       x x x x x         ? x x x
                //                           0 0 0 x x             x x
                //                                 x x               x
                //
                // push a 2x2 bulge forward
                //

                double lV[3];
                double lAlpha = create_left_reflector(2, _A_offset(j+1,j), lV);

                _A(j+1,j) = lAlpha;
                _A(j+2,j) = 0.0;

                lmul2ref(n-j-1, ldA, lV, _A_offset(j+1,j+1));
                if (B != NULL)
                    lmul2ref(n-j-1, ldB, lV, _B_offset(j+1,j+1));
                rmul2ref(n, ldQ, lV, _Q_offset(0,j+1));

                vigilant_deflation_check(thres_a, j, n, ldA, A);

                if (B != NULL) {
                    double rV[3];
                    double rAlpha = create_right_reflector(
                        2, ldB, _B_offset(j+1,j+1), rV);
                    rmul2ref(j+3, ldA, rV, _A_offset(0,j+1));
                    rmul2ref(j+3, ldB, rV, _B_offset(0,j+1));
                    _B(j+1,j+1) = rAlpha;
                    _B(j+2,j+1) = 0.0;
                    rmul2ref(n, ldZ, rV, _Z_offset(0,j+1));
                }
                else {
                    rmul2ref(j+3, ldA, lV, _A_offset(0,j+1));
                }
            }
            else if (j < 0 || (_A(j+1,j) == 0.0 && _A(j+2,j) == 0.0 &&
            _A(j+3,j) == 0.0 && _A(j+3,j+1) == 0.0)) {

                //
                // A               B
                // x x x x x x     x x x x x x
                // x j x x x x       j x x x x
                //   0 x x x x         x x x x
                //   0 x x x x         ? x x x
                //   0 0 x x x             x x
                //         x x               x
                //

                // try to deflate the second column of B
                if (B != NULL)
                    vigilant_deflation_check(thres_b, j+1, n, ldB, B);

                if (B == NULL || (_B(j+1,j+1) != 0.0 && _B(j+2,j+2) != 0.0 &&
                _B(j+3,j+3) != 0.0 && _B(j+2,j+1) == 0.0)) {

                    //
                    // A               B
                    // x x x x x x     x x x x x x
                    // x j x x x x       j x x x x
                    //     x x x x         # x x x
                    //     x x x x         0 # x x
                    //       x x x             # x
                    //         x x               x
                    //

                    //
                    // introduce a new bulge
                    //

                    double lV[3];
                    create_bulge(
                        ldA, ldB, _A_offset(j+1,j+1), _B_offset(j+1,j+1),
                        real+2*i, imag+2*i, lV);
                    create_left_reflector(3, lV, lV);

                    lmul3ref(n-j-1, ldA, lV, _A_offset(j+1,j+1));
                    if (B != NULL)
                        lmul3ref(n-j-1, ldB, lV, _B_offset(j+1,j+1));
                    rmul3ref(n, ldQ, lV, _Q_offset(0,j+1));

                    if (B != NULL) {
                        double rV[3];
                        double rAlpha = create_right_reflector(
                            3, ldB, _B_offset(j+1,j+1), rV);
                        rmul3ref(MIN(n, j+5), ldA, rV, _A_offset(0,j+1));
                        rmul3ref(j+4, ldB, rV, _B_offset(0,j+1));
                        _B(j+1,j+1) = rAlpha;
                        _B(j+2,j+1) = 0.0;
                        _B(j+3,j+1) = 0.0;
                        rmul3ref(n, ldZ, rV, _Z_offset(0,j+1));
                    }
                    else {
                        rmul3ref(MIN(n, j+5), ldA, lV, _A_offset(0,j+1));
                    }
                }
            }
            else if (B != NULL && _B(j+3,j+3) == 0.0) {

                //
                // A               B
                // x x x x x x     x x x x x x
                // x j x x x x       j x x x x
                //   x x x x x         x x x x
                //   x x x x x         x x x x
                //   x x x x x             0 x
                //         x x               x
                //

                push_inf_bulge(j+3, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

                //
                // A               B
                // x x x x x x     x x x x x x
                // x j x x x x       j x x x x
                //   x x x x x         0 x x x
                //   x x x x x           0 x x
                //   x x x x x             0 x
                //     x x x x               x
                //

                double lV[3];
                double lAlpha = create_left_reflector(3, _A_offset(j+1,j), lV);

                lmul3ref(n-j-1, ldA, lV, _A_offset(j+1,j+1));
                lmul3ref(n-j-2, ldB, lV, _B_offset(j+1,j+2));
                rmul3ref(n, ldQ, lV, _Q_offset(0,j+1));

                _A(j+1,j) = lAlpha;
                _A(j+2,j) = 0.0;
                _A(j+3,j) = 0.0;

                //
                // A               B
                // x x x x x x     x x x x x x
                // x j x x x x       j x x x x
                //   x x x x x         0 x x x
                //     x x x x           x x x
                //     x x x x           x x x
                //     x x x x               x
                //

                STARNEIG_SANITY_CHECK(
                    _B(j+1,j+1) == 0.0, "An infinite eigenvalue was lost.");

                //
                // vigilant deflation check
                //

                vigilant_deflation_check(thres_a, j, n, ldA, A);

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
                zeros[j] = zeros[j] || _A(j+1,j) == 0.0;
#endif
            }
            else {

                //
                // push a 3x3 pulge forward
                //

                double lV[3];
                double lAlpha = create_left_reflector(3, _A_offset(j+1,j), lV);

                double rAlpha, rV[3];
                if (B != NULL) {
                    double lB[3*3] = {
                        _B(j+1,j+1), _B(j+2,j+1), _B(j+3,j+1),
                        _B(j+1,j+2), _B(j+2,j+2), _B(j+3,j+2),
                        _B(j+1,j+3), _B(j+2,j+3), _B(j+3,j+3)
                    };

                    lmul3ref(3, 3, lV, lB);
                    rAlpha = create_right_reflector(3, 3, lB, rV);
                }
                else {
                    rAlpha = lAlpha;
                    rV[0] = lV[0];
                    rV[1] = lV[1];
                    rV[2] = lV[2];
                }

                if (rV[2] == 0.0 && _A(j+3,j+1) == 0.0 && (B == NULL ||
                (_B(j+1,j+1) != 0.0 && _B(j+2,j+2) != 0.0 &&
                _B(j+2,j+1) == 0.0))) {

                    //
                    // The bulge is about to collapse:
                    //
                    // A               B
                    // x x x x x x     x x x x x x
                    // x j x x x x       j x x x x    ? == 0 if B == I
                    //   x x x x x         # x x x
                    //   x x x x x         0 # x x
                    //   ? 0 x x x             x x
                    //         x x               x
                    //
                    // See what happens if we re-create the bulge in the next
                    // column.
                    //

                    // compute the reflector that would re-create the bulge
                    double vt[3];
                    create_bulge(
                        ldA, ldB, _A_offset(j+1,j+1), _B_offset(j+1,j+1),
                        real+2*i, imag+2*i, vt);
                    create_left_reflector(3, vt, vt);

                    //
                    // compute what the two entries (#) in the first column
                    // would be if we re-created the bulge by applying the
                    // new reflector from the left
                    //
                    // x x x x x x
                    // x_j_x_x_x_x
                    //   X X X X X
                    //   # X X X X
                    // __#_X_X_X_X
                    //         x x
                    //
                    double s0 = vt[0] * _A(j+1,j) + vt[1] * _A(j+2,j) +
                        vt[2] * _A(j+3,j);
                    double a20 = _A(j+2,j) - s0 * vt[1];
                    double a30 = _A(j+3,j) - s0 * vt[2];

                    //
                    // check whether the two entries can be safely ignored
                    //
                    int can_be_ignored;
                    if (0.0 < thres_a) {
                        // compare the two entries agains the threshold
                        can_be_ignored =
                            fabs(a20) < thres_a && fabs(a30) < thres_a;
                    }
                    else {
                        //
                        // use the sum of the absolute values of diagonal
                        // entries as a threshold
                        //
                        // x x x x x x
                        // x_#_x_x_x_x
                        //   X # X X X
                        //   X X # X X
                        // __X_X_X_#_X
                        //         x x
                        //

                        double s1 = vt[0] * _A(j+1,j+1) + vt[1] * _A(j+2,j+1);
                        double s2 = vt[0] * _A(j+1,j+2) + vt[1] * _A(j+2,j+2) +
                            vt[2] * _A(j+3,j+2);
                        double s3 = vt[0] * (_A(j+1,j+3) + vt[1] * _A(j+2,j+3) +
                            vt[2] * _A(j+3,j+3));
                        double a11 = _A(j+1,j+1) - s1 * vt[0];
                        double a22 = _A(j+2,j+2) - s2 * vt[1];
                        double a33 = _A(j+3,j+3) - s3 * vt[2];

                        double ulp = dlamch("Precision");
                        double eps = ulp *
                            (fabs(_A(j,j)) + fabs(a11) + fabs(a22) + fabs(a33));

                        can_be_ignored = fabs(a20) + fabs(a30) < eps;
                    }

                    if (can_be_ignored) {
                        // The two entires can be ignored. The original
                        // reflector is replaced with the new one.
                        _A(j+1,j) -= s0 * vt[0];
                        _A(j+2,j)  = 0.0;
                        _A(j+3,j)  = 0.0;
                        lV[0] = vt[0];
                        lV[1] = vt[1];
                        lV[2] = vt[2];

                        if (B != NULL) {
                            double lB[3*3] = {
                                _B(j+1,j+1), _B(j+2,j+1), _B(j+3,j+1),
                                _B(j+1,j+2), _B(j+2,j+2), _B(j+3,j+2),
                                _B(j+1,j+3), _B(j+2,j+3), _B(j+3,j+3)
                            };

                            lmul3ref(3, 3, lV, lB);
                            rAlpha = create_right_reflector(3, 3, lB, rV);
                        }
                        else {
                            rAlpha = lAlpha;
                            rV[0] = lV[0];
                            rV[1] = lV[1];
                            rV[2] = lV[2];
                        }
                    }
                    else {
                        // The two entries cannot be ignored and they would
                        // destroy the upper Hessenberg form. Use the old
                        // reflector.
                        _A(j+1,j) = lAlpha;
                        _A(j+2,j) = 0.0;
                        _A(j+3,j) = 0.0;
                    }

                }
                else {
                    //
                    // the bulge is not going to collapse
                    //
                    // x x x x x x
                    // x j x x x x
                    //   x x x x x
                    //   x x x x x
                    //   x x x x x
                    //         x x
                    //
                    _A(j+1,j) = lAlpha;
                    _A(j+2,j) = 0.0;
                    _A(j+3,j) = 0.0;
                }

                lmul3ref(n-j-1, ldA, lV, _A_offset(j+1,j+1));
                if (B != NULL)
                    lmul3ref(n-j-1, ldB, lV, _B_offset(j+1,j+1));
                rmul3ref(n, ldQ, lV, _Q_offset(0,j+1));

                if (B != NULL) {
                    rmul3ref(MIN(n, j+5), ldA, rV, _A_offset(0,j+1));
                    rmul3ref(j+4, ldB, rV, _B_offset(0,j+1));
                    _B(j+1,j+1) = rAlpha;
                    _B(j+2,j+1) = 0.0;
                    _B(j+3,j+1) = 0.0;
                    rmul3ref(n, ldZ, rV, _Z_offset(0,j+1));
                }
                else {
                    rmul3ref(MIN(n, j+5), ldA, lV, _A_offset(0,j+1));
                }
            }

skip_column:

            //
            // vigilant deflation check
            //

            if (j != -1) {
                vigilant_deflation_check(thres_a, j, n, ldA, A);

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
                zeros[j] = zeros[j] || _A(j+1,j) == 0.0;
#endif
            }

            //
            // infinite eigenvalue check
            //

            if (B != NULL && fabs(_B(j+1,j+1)) < thres_inf)
                _B(j+1,j+1) = 0.0;

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
            for (int k = 0; k < n-1; k++) {
                if (zeros[k]) {
                    STARNEIG_SANITY_CHECK(
                        n <= k+1 || k-2 < 0 || _A(k+1,k-2) == 0.0,
                        "A zero sub-diagonal entry was lost.");
                    STARNEIG_SANITY_CHECK(
                        n <= k+1 || k-1 < 0 || _A(k+1,k-1) == 0.0,
                        "A zero sub-diagonal entry was lost.");
                    STARNEIG_SANITY_CHECK(
                        n <= k+1 ||            _A(k+1,k)   == 0.0,
                        "A zero sub-diagonal entry was lost.");
                    STARNEIG_SANITY_CHECK(
                        n <= k+2 || k-1 < 0 || _A(k+2,k-1) == 0.0,
                        "A zero sub-diagonal entry was lost.");
                    STARNEIG_SANITY_CHECK(
                        n <= k+2 ||            _A(k+2,k)   == 0.0,
                        "A zero sub-diagonal entry was lost.");
                    STARNEIG_SANITY_CHECK(
                        n <= k+3 ||            _A(k+3,k)   == 0.0,
                        "A zero sub-diagonal entry was lost.");
                }
            }
#endif
        }

        begin++;
    }

    if (mode == BULGE_CHASING_MODE_CHASE ||
    mode == BULGE_CHASING_MODE_INTRODUCE) {
        STARNEIG_SANITY_CHECK_HESSENBERG(
            0, n-3*(shifts/2)-1, n, ldA, ldB, A, B);
        STARNEIG_SANITY_CHECK_BULGES(
            n-3*(shifts/2)-1, shifts, n, ldA, ldB, A, B);
    }
    else {
        STARNEIG_SANITY_CHECK_HESSENBERG(0, n, n, ldA, ldB, A, B);
    }

    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);
}

///
/// @brief Computes the amount of worspace required by perform_push_bulges().
///
/// @see perform_push_bulges()
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @return The amount of required workspace.
///
static size_t get_push_bulges_workspace(
    int n, int ldQ, int ldZ, int ldA, int ldB)
{
    const int batch = 16;
    int window_size = MIN(n, 3*batch+2);

    size_t lwork = 0;
    lwork += window_size * divceil(window_size, 8)*8;
    if (0 < ldB)
        lwork += window_size * divceil(window_size, 8)*8;
    lwork += MAX(n * divceil(window_size, 8)*8, window_size * divceil(n, 8)*8);

    return lwork;
}

///
/// @brief Bulge chasing kernel.
///
///  Chases a set of bulges across a matrix pencil Q (A,B) Z^T. Produces an
///  updated matrix pencil ~Q (~A,~B) ~Z^T. The chasing is performed between
///  columns `begin` and `end`.
///
/// @param[in] mode
///         The bulge chasing mode.
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] shifts
///         Number of shifts to use.
///
/// @param[in] n
///         The order of matrices Q, Z, A and B.
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
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] real
///         Shifts (real parts).
///
/// @param[in] imag
///         Shifts (imaginary parts).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @param[out] work
///         The workspace buffer.
///
static void perform_push_bulges(
    bulge_chasing_mode_t mode, int begin, int end, int shifts, int n,
    int ldQ, int ldZ, int ldA, int ldB, size_t lwork,
    double thres_a, double thres_b, double thres_inf,
    double const *real, double const *imag,
    double *Q, double *Z, double *A, double *B, double *work)
{
    // the shifts are processed in batches
    const int batch = 16;
    int window_size = MIN(n, 3*batch+2);

#ifdef STARNEIG_ENABLE_SANITY_CHECKS
    STARNEIG_SANITY_CHECK(2 <= shifts, "Too few shifts.");
    STARNEIG_SANITY_CHECK(shifts % 2 == 0, "Odd number of shifts.");
    STARNEIG_SANITY_CHECK(mode == BULGE_CHASING_MODE_FULL ||
        3*(shifts/2)+1 <= n, "All bulges do not fit into the window.");

    for (int i = 0; i < shifts; i++)
        STARNEIG_SANITY_CHECK(
            real[i] != 0.0 || imag[i] != 0, "Some shifts are zero.");

    for (int i = 0; i < shifts; i++)
        STARNEIG_SANITY_CHECK(!isinf(real[i]) && !isinf(imag[i]),
            "Some shifts are infinite.");

    for (int i = 0; i < shifts; i++)
        STARNEIG_SANITY_CHECK(!isnan(real[i]) && !isnan(imag[i]),
            "Some shifts are NaNs.");

    for (int i = 0; i < shifts/2; i++)
        STARNEIG_SANITY_CHECK(
            imag[2*i] == -imag[2*i+1], "The shifts are not ordered correctly.");

    if (mode == BULGE_CHASING_MODE_CHASE ||
    mode == BULGE_CHASING_MODE_FINALIZE) {
        STARNEIG_SANITY_CHECK_BULGES(0, shifts, n, ldA, ldB, A, B);
        STARNEIG_SANITY_CHECK_HESSENBERG(3*(shifts/2), n, n, ldA, ldB, A, B);
    }
    else {
        STARNEIG_SANITY_CHECK_HESSENBERG(0, n, n, ldA, ldB, A, B);
    }

    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    STARNEIG_SANITY_CHECK(
        get_push_bulges_workspace(n, ldQ, ldZ, ldA, ldB) <= lwork,
        "Invalid workspace size.");
#endif

    int introduce =
        mode == BULGE_CHASING_MODE_FULL || mode == BULGE_CHASING_MODE_INTRODUCE;
    int finalize =
        mode == BULGE_CHASING_MODE_FULL || mode == BULGE_CHASING_MODE_FINALIZE;

    //
    // deflate infinite eigenvalues
    //

    if (B != NULL) {
        if (introduce) {
            // we can savely deflate infinite eigenvalues from the top
            starneig_push_inf_top(
                begin, 0, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 1);
        }
        else {
            // locate the part of the window that is in Hessenberg-triangular
            // form
            int i = begin-1;
            while (i+4 < n && _A(i+3,i+1) == 0.0 && _A(i+4,i+1) == 0.0 &&
            _B(i+2,i+1) == 0.0 && _B(i+3,i+2) == 0.0)
                i++;

            if (begin < i) {
                // if infinite eigenvalues can be deflated, ...
                if (_A(begin+1,begin) == 0.0) {
                    // deflate them;
                    starneig_push_inf_top(
                        begin+1, i, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 1);
                }
                // otherwise, ...
                else {
                    // gather them to the upper left corner
                    int j = begin;
                    while (j < i && _B(j,j) == 0.0)
                        j++;
                    starneig_push_inf_top(
                        j, i, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B, 0);
                }
            }
        }
    }

    int top, bottom, bjump, ejump;
    switch (mode) {
        case BULGE_CHASING_MODE_INTRODUCE:
            top = begin;
            bottom = end;
            bjump = 0;
            ejump = -3*(batch/2);
            break;
        case BULGE_CHASING_MODE_CHASE:
            top = MAX(begin, begin+3*(shifts/2)-3*(batch/2));
            bottom = end;
            bjump = -3*(batch/2);
            ejump = -3*(batch/2);
            break;
        case BULGE_CHASING_MODE_FINALIZE:
            top = MAX(begin, begin+3*(shifts/2)-3*(batch/2));
            bottom = end;
            bjump = -3*(batch/2);
            ejump = 0;
            break;
        default:
            top = begin;
            bottom = end;
            bjump = 0;
            ejump = 0;
            break;
    }

    int ldlQ = divceil(window_size, 8)*8;
    double *lQ = work;
    work += window_size*ldlQ;

    int ldlZ = ldlQ;
    double *lZ = lQ;
    if (B != NULL) {
        ldlZ = divceil(window_size, 8)*8;
        lZ = work;
        work += window_size*ldlZ;
    }

    int ldhT = divceil(window_size, 8)*8;
    double *hT = work;

    int ldvT = divceil(n, 8)*8;
    double *vT = work;

    // divide the shifts to batches
    for (int i = 0; i < shifts; i += batch) {
        int wbegin = top;
        int wend = MIN(bottom, wbegin+window_size);

        // push a batch across the matrix
        while (wbegin < bottom) {

            // infer the bulge chasing mode
            bulge_chasing_mode_t window_mode;
            if (wbegin == top && wend == bottom)
                window_mode = mode;
            else if (wbegin == top && introduce)
                window_mode = BULGE_CHASING_MODE_INTRODUCE;
            else if (wend == bottom && finalize)
                window_mode = BULGE_CHASING_MODE_FINALIZE;
            else
                window_mode = BULGE_CHASING_MODE_CHASE;

            starneig_init_local_q(wend-wbegin, ldlQ, lQ);
            if (lZ != lQ)
                starneig_init_local_q(wend-wbegin, ldlZ, lZ);

            // push the batch across a small diagonal window
            process_small_window(
                window_mode, MIN(batch, shifts-i), wend-wbegin,
                ldlQ, ldlZ, ldA, ldB, thres_a, thres_b, thres_inf,
                real+i, imag+i, lQ, lZ,
                _A_offset(wbegin,wbegin), _B_offset(wbegin,wbegin));

            starneig_small_gemm_updates(
                wbegin, wend, n, ldlQ, ldlZ, ldQ, ldZ, ldA, ldB, ldhT, ldvT,
                lQ, lZ, Q, Z, A, B, hT, vT);

            if (wend == bottom)
                break;

            wbegin = wend - 3*(MIN(batch, shifts-i)/2) - 1;
            wend = MIN(bottom, wbegin+window_size);
        }

        top = MAX(0, MIN(n, top+bjump));
        bottom = MAX(0, MIN(n, bottom+ejump));
    }

    if (mode == BULGE_CHASING_MODE_CHASE ||
    mode == BULGE_CHASING_MODE_INTRODUCE) {
        STARNEIG_SANITY_CHECK_HESSENBERG(
            0, n-3*(shifts/2)-1, n, ldA, ldB, A, B);
        STARNEIG_SANITY_CHECK_BULGES(
            n-3*(shifts/2)-1, shifts, n, ldA, ldB, A, B);
    }
    else {
        STARNEIG_SANITY_CHECK_HESSENBERG(0, n, n, ldA, ldB, A, B);
    }
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);
}

///
/// @brief Computes the amount of worspace required by
/// perform_lapack_schur_reduction().
///
/// @see perform_lapack_schur_reduction()
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @return The amount of required workspace.
///
static size_t get_lapack_schur_reduction_workspace(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB)
{
    extern void dhseqr_(
        char const *, char const *, int const *, int const *, int const *,
        double *, int const *, double *, double *, double *, int const *,
        double *, int const *, int *);

    extern void dhgeqz_(
        char const *, char const *, char const *, int const *, int const *,
        int const *, double *, int const *, double *, int const *, double *,
        double *, double *, double *, int const *, double *, int const *,
        double *, int const *, int *);

    if (end-begin <= 2)
        return 0;

    double *A = NULL, *B = NULL, *Q = NULL, *Z = NULL;
    double dlwork, *real = NULL, *imag = NULL, *beta = NULL;
    int info, _lwork = -1, ilo = begin+1, ihi = end;

    if (0 < ldB)
        dhgeqz_("S", "V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, real, imag, beta, Q, &ldQ, Z, &ldZ,
            &dlwork, &_lwork, &info);
    else
        dhseqr_("S", "V", &n, &ilo, &ihi,
            A, &ldA, real, imag, Q, &ldQ, &dlwork, &_lwork, &info);

    if (info < 0)
        starneig_fatal_error(
            "An unrecoverable internal error occured while calling dhgeqz "
            "or dhseqr.");

    return (size_t) dlwork;
}

///
/// @brief Performs a Schur reduction using LAPACK subroutines.
///
///  Reduces a matrix pencil Q (A,B) Z^T to Schur form. Produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T. Only two columns between `begin` and `end`
///  are processed.
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] shifts
///         Number of shifts to use.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in] lwork
///         Size of the workspace buffer.
///
/// @param[out] real
///         Returns the eigenvalues (real parts).
///
/// @param[out] imag
///          Returns the eigenvalues (imaginary parts).
///
/// @param[out] beta
///          Returns the eigenvalues (scaling factors).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @param[out] work
///         The workspace buffer.
///
/// @return The first column that has been reduced to Schur form.
///
static int perform_lapack_schur_reduction(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double *real, double *imag, double *beta, double *Q, double *Z, double *A,
    double *B, double *work)
{
    extern void dhseqr_(
        char const *, char const *, int const *, int const *, int const *,
        double *, int const *, double *, double *, double *, int const *,
        double *, int const *, int *);

    extern void dhgeqz_(
        char const *, char const *, char const *, int const *, int const *,
        int const *, double *, int const *, double *, int const *, double *,
        double *, double *, double *, int const *, double *, int const *,
        double *, int const *, int *);

    if (end-begin < 2)
        return begin;

    STARNEIG_SANITY_CHECK(
        get_lapack_schur_reduction_workspace(
            begin, end, n, ldQ, ldZ, ldA, ldB) <= lwork,
        "Invalid workspace size.");

    STARNEIG_SANITY_CHECK(
        begin == 0 || _A(begin,begin-1) == 0.0, "Invalid matrix.");
    STARNEIG_SANITY_CHECK(end == n, "Invalid matrix (LAPACK bug?).");
    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    int info, ilo = begin+1, ihi = end;

    if (end-begin == 2) {
        process_2x2_block(
            begin, n, ldQ, ldZ, ldA, ldB, real, imag, beta, Q, Z, A, B);
        info = 0;
    }
    else if (B != NULL) {
        dhgeqz_("S", "V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, real, imag, beta, Q, &ldQ, Z, &ldZ,
            work, &lwork, &info);
    }
    else {
        dhseqr_("S", "V", &n, &ilo, &ihi,
            A, &ldA, real, imag, Q, &ldQ, work, &lwork, &info);
    }

    if (info < 0)
        starneig_fatal_error(
            "An unrecoverable internal error occured while calling dhgeqz "
            "or dhseqr.");

    int bottom = info == 0 ? begin : info;

    STARNEIG_SANITY_CHECK_HESSENBERG(begin, bottom, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_SCHUR(bottom, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return bottom;
}

///
/// @brief Computes the amount of worspace required by
/// perform_small_schur_reduction().
///
/// @see perform_small_schur_reduction()
///
/// @param[in] window_size
///         The window size (end-begin).
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @return The amount of required workspace.
///
static size_t get_small_schur_reduction_workspace(
    int window_size, int n, int ldQ, int ldZ, int ldA, int ldB)
{
    if (window_size == n)
        return get_lapack_schur_reduction_workspace(
            0, n, n, ldQ, ldZ, ldA, ldB);

    int lwork = 0, ld = divceil(window_size, 8)*8;

    if (0 < ldB) {
        lwork += 4*window_size * ld;
        lwork += get_lapack_schur_reduction_workspace(
            0, window_size, window_size, ld, ld, ld, ld);
    }
    else {
        lwork += 2*window_size * ld;
        lwork += get_lapack_schur_reduction_workspace(
            0, window_size, window_size, ld, ld, ld, 0);
    }

    lwork += MAX(n*ld, window_size*divceil(n, 8)*8);

    return lwork;
}

///
/// @brief Performs a Schur reduction.
///
///  Reduces a matrix pencil Q (A,B) Z^T to Schur form. Produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T. Only two columns between `begin` and `end`
///  are processed.
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] shifts
///         Number of shifts to use.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in] lwork
///         Size of the workspace buffer.
///
/// @param[out] real
///         Returns the eigenvalues (real parts).
///
/// @param[out] imag
///          Returns the eigenvalues (imaginary parts).
///
/// @param[out] beta
///          Returns the eigenvalues (scaling factors).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @param[out] work
///         The workspace buffer.
///
/// @return The first column that has been reduced to Schur form.
///
static int perform_small_schur_reduction(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double *real, double *imag, double *beta, double *Q, double *Z, double *A,
    double *B, double *work)
{
    if (begin == 0 && end == n)
        return perform_lapack_schur_reduction(0, n, n, ldQ, ldZ, ldA, ldB,
            lwork, real, imag, beta, Q, Z, A, B, work);

    int window_size = end-begin;

    STARNEIG_SANITY_CHECK(
        get_small_schur_reduction_workspace(
            window_size, n, ldQ, ldZ, ldA, ldB) <= lwork,
        "Invalid workspace size.");

    STARNEIG_SANITY_CHECK(
        begin == 0 || _A(begin,begin-1) == 0.0, "Invalid matrix.");
    STARNEIG_SANITY_CHECK(
        end == n || _A(end,end-1) == 0.0, "Invalid matrix.");
    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    // prepare workspace

    double *__A = NULL, *__B = NULL, *__Q = NULL, *__Z = NULL,
        *__hT = NULL, *__vT = NULL;
    double *__work = work;
    int __lwork = lwork;
    int __ld = divceil(window_size, 8)*8;
    int __ldhT = divceil(window_size, 8)*8;
    int __ldvT = divceil(n, 8)*8;

    #define add_work(__X, __X_size) \
        __X = __work; __work += __X_size; __lwork -= __X_size

    if (B != NULL) {
        add_work(__A, window_size*__ld);
        add_work(__B, window_size*__ld);
        add_work(__Q, window_size*__ld);
        add_work(__Z, window_size*__ld);
    }
    else {
        add_work(__A, window_size*__ld);
        add_work(__Q, window_size*__ld);
        __Z = __Q;
    }
    add_work(__hT, 0);
    add_work(__vT, MAX(n*__ldhT, window_size*__ldvT));

    #undef add_work

    // copy a diagonal window to workspace buffers

    starneig_copy_matrix(window_size, window_size, ldA, __ld, sizeof(double),
        _A_offset(begin, begin), __A);
    starneig_init_local_q(window_size, __ld, __Q);
    if (B != NULL) {
        starneig_copy_matrix(window_size, window_size, ldB, __ld,
            sizeof(double), _B_offset(begin, begin), __B);
        starneig_init_local_q(window_size, __ld, __Z);
    }

    // reduce the copy of the window to Schur form

    int __bottom = perform_lapack_schur_reduction(0, window_size, window_size,
        __ld, __ld, __ld, __ld, __lwork, real+begin, imag+begin, beta+begin,
        __Q, __Z, __A, __B, __work);
    int bottom = begin + __bottom;

    // copy the reduced copy back to the matrix

    starneig_copy_matrix(window_size, window_size, __ld, ldA, sizeof(double),
        __A, _A_offset(begin, begin));
    if (B != NULL)
        starneig_copy_matrix(
            window_size, window_size, __ld, ldB, sizeof(double),
            __B, _B_offset(begin, begin));

    // apply off-diagonal updates

    starneig_small_gemm_updates(begin, end, n, __ld, __ld, ldQ, ldZ, ldA, ldB,
        __ldhT, __ldvT, __Q, __Z, Q, Z, A, B, __hT, __vT);

    STARNEIG_SANITY_CHECK_SCHUR(bottom, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return bottom;
}

///
/// @brief Computes the amount of worspace required by
/// perform_hessenberg_reduction().
///
/// @see perform_hessenberg_reduction()
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @return The amount of required workspace.
///
static size_t get_hessenberg_reduction_workspace(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB)
{
    extern void dgehrd_(
        int const *, int const *, int const *, double *, int const *, double *,
        double *, int const *, int *);

    extern void dormhr_(
        char const *, char const *, int const *, int const *, int const *,
        int const *, double const *, int const *, double const *, double *,
        int const *, double *,  int const *, int *);

    if (0 < ldB)
        return 0;

    double *A = NULL, *Q = NULL;
    int info, lwork = 0, ilo = begin+1, ihi = end;

    {
        int _lwork = -1;
        double dlwork;

        dgehrd_(&n, &ilo, &ihi, A, &ldA, NULL, &dlwork, &_lwork, &info);

        if (info < 0)
            starneig_fatal_error(
                "An unrecoverable internal error occured while calling "
                "dgehrd.");

        lwork = MAX(lwork, dlwork);
    }

    {
        int _lwork = -1;
        double dlwork;

        dormhr_("Right", "No transpose", &n, &n,
            &ilo, &ihi, A, &ldA, NULL, Q, &ldQ, &dlwork, &_lwork, &info);

        if (info < 0)
            starneig_fatal_error(
                "An unrecoverable internal error occured while calling "
                "dormhr.");

        lwork = MAX(lwork, dlwork);
    }

    // tau
    lwork += n;

    return lwork;
}

///
/// @brief Performs Hessenberg reduction.
///
///  Reduces a matrix pencil Q (A,B) Z^T to upper Hessenberg form. Produces an
///  updated matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in] lwork
///         Size of the workspace buffer.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @param[out] work
///         The workspace buffer.
///
/// @return The first column that has been reduced to Hessenberg form.
///
static int perform_hessenberg_reduction(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double *Q, double *Z, double *A, double *B, double *work)
{
    extern void dgehrd_(
        int const *, int const *, int const *, double *, int const *, double *,
        double *, int const *, int *);

    extern void dormhr_(
        char const *, char const *, int const *, int const *, int const *,
        int const *, double const *, int const *, double const *, double *,
        int const *, double *,  int const *, int *);

    extern void dgghrd_(
        char const *, char const *, int const *, int const *, int const *,
        double *, int const *, double *, int const *, double *, int const *,
        double *, int const *, int *);

    STARNEIG_SANITY_CHECK(
        get_hessenberg_reduction_workspace(
            begin, end, n, ldQ, ldZ, ldA, ldB) <= lwork,
        "Invalid workspace size.");

    STARNEIG_SANITY_CHECK_HESSENBERG(0, begin, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_HESSENBERG(end, n, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    int info, ilo = begin+1, ihi = end;

    if (B != NULL) {
        dgghrd_("V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, Q, &ldQ, Z, &ldZ, &info);
    }
    else {
        double *tau = work;
        double *_work = work + n;
        int _lwork = lwork - n;

        dgehrd_(&n, &ilo, &ihi, A, &ldA, tau, _work, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        dormhr_("Right", "No transpose", &n, &n,
            &ilo, &ihi, A, &ldA, tau, Q, &ldQ, _work, &_lwork, &info);
        if (info != 0)
            goto cleanup;

        for (int i = begin; i < end; i++)
            for (int j = i+2; j < end; j++)
                _A(j,i) = 0.0;
    }

cleanup:

    if (info < 0)
        starneig_fatal_error(
            "An unrecoverable internal error occured while calling dgghrd, "
            "dgehrd or dormhr.");

    STARNEIG_SANITY_CHECK_HESSENBERG(0, n, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return info;
}

static size_t get_schur_reduction_workspace(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB);

static int perform_schur_reduction(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double thres_a, double thres_b, double thres_inf, double *work,
    double *real, double *imag, double *beta, double *Q,
    double *Z, double *A, double *B);

///
/// @brief Computes the amount of worspace required by
/// perform_aggressively_deflate().
///
/// @see perform_aggressively_deflate()
///
/// @param[in] n
///         Order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @return The amount of required workspace.
///
static size_t get_aggressively_deflate_workspace(
    int n, int ldQ, int ldZ, int ldA, int ldB)
{
    size_t lwork = 0;

    if (0 < ldB)
        lwork += 4*n+16;    // dtgexc_
    else
        lwork += n;         // dtrexc_

    lwork = MAX(lwork,
        get_schur_reduction_workspace(1, n, n, ldQ, ldZ, ldA, ldB));
    lwork = MAX(lwork,
        get_hessenberg_reduction_workspace(1, n, n, ldQ, ldZ, ldA, ldB));

    lwork += 3*n;           // eigenvalues

    return lwork;
}

///
/// @brief Performs an aggressive early deflation.
///
///  Performs AED on a matrix pencil  Q (A,B) Z^T. May produce an updated matrix
///  pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in] lwork
///         Size of the workspace buffer.
///
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[out] real
///         Returns the real parts of the computed shifts.
///
/// @param[out] imag
///         Returns the imaginary parts of the computed shifts.
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q if 0 < converged.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z if 0 < converged.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A if 0 < converged.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B if 0 < converged.
///
/// @param[out] work
///         The workspace buffer.
///
/// @param[out] unconverged
///         Returns the number of unconverged eigenvalues / shifts.
///
/// @param[out] converged
///         Returns the number of computed shifts.
///
static void perform_aggressively_deflate(
    int n,  int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double thres_a, double thres_b, double thres_inf,
    double * restrict real, double * restrict imag, double * restrict Q,
    double * restrict Z, double * restrict A, double * restrict B,
    double * restrict work, int * restrict unconverged,
    int * restrict converged)
{
    STARNEIG_SANITY_CHECK(
        get_aggressively_deflate_workspace(n, ldQ, ldZ, ldA, ldB) <= lwork,
        "Invalid workspace size.");

    STARNEIG_SANITY_CHECK_HESSENBERG(0, n, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    *unconverged = 0;
    *converged = 0;
    int info = 0;

    //
    // decouple the AED window from the rest of the matrix
    //

    double sub = _A(1,0);
    _A(1,0) = 0.0;

    //
    // reduce the AED window to Schur form
    //

    int roof = perform_schur_reduction(1, n, n, ldQ, ldZ, ldA, ldB, lwork-3*n,
        thres_a, thres_b, thres_inf, work, work+n, work+2*n, Q, Z, A, B,
        work+3*n);
    if (1 < roof)
        starneig_verbose(
            "Failed to reduce the whole AED window to Schur form.");

    STARNEIG_SANITY_CHECK_HESSENBERG(1, roof, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_SCHUR(roof, n, n, ldA, ldB, A, B);

    //
    // attempt to deflate eigenvalues
    //

    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_2, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    int top = roof;
    {
        //
        // norm stable deflation condition
        //

        if (0.0 < thres_a) {

            int i = n-1;
            while (top <= i) {

                // if we are dealing with a 2-by-2 block, ...
                if (top <= i-1 && _A(i,i-1) != 0.0) {

                    // and the 2-by-2 block is deflatable, ...
                    if (fabs(sub*_Q(1,i-1)) < thres_a &&
                    fabs(sub*_Q(1,i)) < thres_a) {
                        // decrease the AED window
                        i -= 2;
                    }
                    // otherwise, ...
                    else {
                        // move the 2-by-2 block out of the way
                        top = starneig_move_block(
                            i, top, n, ldQ, ldZ, ldA, ldB, lwork,
                            Q, Z, A, B, work);
                        top += 2;
                    }
                }
                // otherwise, ...
                else {
                    // if the 1-by-1 block is deflatable, ...
                    if (fabs(sub*_Q(1,i)) < thres_a) {
                        // decrease the AED window
                        i--;
                    }
                    // otherwise, ...
                    else {
                        // move the 1-by-1 block out of the way
                        top = starneig_move_block(
                            i, top, n, ldQ, ldZ, ldA, ldB, lwork,
                            Q, Z, A, B, work);
                        top++;
                    }
                }
            }
        }

        //
        // LAPACK-style deflation condition
        //

        else {
            const double safmin = dlamch("Safe minimum");
            const double ulp = dlamch("Precision");
            double smlnum = safmin*(n/ulp);

            int i = n-1;
            while (top <= i) {

                // if we are dealing with a 2-by-2 block, ...
                if (top <= i-1 && _A(i,i-1) != 0.0) {
                    double foo = fabs(_A(i,i)) +
                        sqrt(fabs(_A(i,i-1))) * sqrt(fabs(_A(i-1,i)));
                    if (foo == 0.0)
                        foo = fabs(sub);

                    // and the 2-by-2 block is deflatable, ...
                    if (MAX(fabs(sub*_Q(1,i-1)), fabs(sub*_Q(1,i))) <
                    MAX(smlnum, ulp*foo)) {
                        // decrease the AED window
                        i -= 2;
                    }
                    // otherwise, ...
                    else {
                        // move the 2-by-2 block out of the way
                        top = starneig_move_block(
                            i, top, n, ldQ, ldZ, ldA, ldB, lwork,
                            Q, Z, A, B, work);
                        top += 2;
                    }
                }
                // otherwise, ...
                else {
                    double foo = fabs(_A(i,i));
                    if (foo == 0.0)
                        foo = fabs(sub);

                    // if the 1-by-1 block is deflatable, ...
                    if (fabs(sub*_Q(1,i)) < MAX(smlnum, ulp*foo)) {
                        // decrease the AED window
                        i--;
                    }
                    // otherwise, ...
                    else {
                        // move the 1-by-1 block out of the way
                        top = starneig_move_block(
                            i, top, n, ldQ, ldZ, ldA, ldB, lwork,
                            Q, Z, A, B, work);
                        top++;
                    }
                }
            }
        }
    }

    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_2, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    //
    // extract shifts
    //

    int shifts;
    if (2 <= top-roof) {
        shifts = starneig_extract_shifts(
            top-roof, ldA, ldB, _A_offset(roof,roof), _B_offset(roof, roof),
            real, imag);
    }
    else {
        // extract something
        shifts = starneig_extract_shifts(
            top-1, ldA, ldB, _A_offset(1,1), _B_offset(1, 1), real, imag);
    }

    *unconverged = shifts;
    *converged = n-top;

    STARNEIG_SANITY_CHECK_HESSENBERG(0, roof, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_SCHUR(roof, n, n, ldA, ldB, A, B);

    if (*converged == 0)
        goto cleanup;

    //
    // embed the spike
    //

    for (int i = 1; i < top; i++)
        _A(i,0) = sub*_Z(0,0)*_Q(1,i); // _Z(0.0) can be something else that 1.0

    //
    // reduce non-deflated upper half to upper Hessenberg form
    //

    info = perform_hessenberg_reduction(
        0, top, n, ldQ, ldZ, ldA, ldB, lwork, Q, Z, A, B, work);
    if (info != 0)
        goto cleanup;

    STARNEIG_SANITY_CHECK_HESSENBERG(0, top, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_SCHUR(top, n, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

cleanup:

    STARNEIG_SANITY_CHECK_RESIDUALS_SKIP(SANITY_1);

    if (info != 0)
        starneig_verbose("Something went wrong with AED.");
}

///
/// @brief Computes the amount of worspace required by
/// perform_schur_reduction().
///
/// @see perform_schur_reduction()
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @return The amount of required workspace.
///
static size_t get_schur_reduction_workspace(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB)
{
    const int small_limit = 64;
    const int shift_count = divceil(0.06*n, 2)*2;
    const int aed_window_size = MAX(shift_count+2, 0.08*n);

    if (ldB == 0 || end-begin <= small_limit)
        return get_small_schur_reduction_workspace(
            end-begin, n, ldQ, ldZ, ldA, ldB);

    // compute AED workspace size

    size_t aed_lwork = 0;

    int ld = divceil(aed_window_size, 8)*8;
    if (0 < ldB) {
        aed_lwork += get_aggressively_deflate_workspace(
            aed_window_size, ld, ld, ld, ld);
        aed_lwork += 4*aed_window_size*ld;
    }
    else {
        aed_lwork += get_aggressively_deflate_workspace(
            aed_window_size, ld, ld, ld, 0);
        aed_lwork += 2*aed_window_size*ld;
    }

    aed_lwork +=
        MAX(n*divceil(aed_window_size, 8)*8, aed_window_size*divceil(n, 8)*8);

    return MAX(aed_lwork, MAX(
        get_push_bulges_workspace(n, ldQ, ldZ, ldA, ldB),
        get_small_schur_reduction_workspace(
            small_limit, n, ldQ, ldZ, ldA, ldB)));
}

///
/// @brief Performs Schur reduction.
///
///  Reduces a matrix pencil Q (A,B) Z^T to Schur form. Produces an updated
///  matrix pencil ~Q (~A,~B) ~Z^T.
///
/// @param[in] begin
///         The first row/column to be processed.
///
/// @param[in] end
///         The last row/column to be processed + 1.
///
/// @param[in] n
///         The order of the matrices A, B, Q and Z.
///
/// @param[in] ldQ
///         The leading dimension of Q.
///
/// @param[in] ldZ
///         The leading dimension of Z.
///
/// @param[in] ldA
///         The leading dimension of A.
///
/// @param[in] ldB
///         The leading dimension of B.
///
/// @param[in] lwork
///         Size of the workspace buffer.
///
/// @param[in] thres_a
///         Those off-diagonal entries of the matrix A that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_b
///         Those off-diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[in] thres_inf
///         Those diagonal entries of the matrix B that are smaller in
///         magnitudes than this threshold may be set to zero.
///
/// @param[out] real
///         Returns the eigenvalues (real parts).
///
/// @param[out] imag
///         Returns the eigenvalues (imaginary parts).
///
/// @param[out] beta
///         Returns the eigenvalues (scaling factors).
///
/// @param[in,out] Q
///         On entry, the matrix Q.
///         On exit, the matrix ~Q.
///
/// @param[in,out] Z
///         On entry, the matrix Z. If NULL, then it is assumed that Z = Q.
///         On exit, the matrix ~Z.
///
/// @param[in,out] A
///         On entry, the matrix A.
///         On exit, the matrix ~A.
///
/// @param[in,out] B
///         On entry, the matrix B. If NULL, then it is assumed that B = I.
///         On exit, the matrix ~B.
///
/// @param[out] work
///         The workspace buffer.
///
/// @return The first column that has been reduced to Schur form.
///
static int perform_schur_reduction(
    int begin, int end, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double thres_a, double thres_b, double thres_inf,
    double *real, double *imag, double *beta, double *Q, double *Z, double *A,
    double *B, double *work)
{
    const int max_iter = 300;
    const int small_limit = 64;
    const int shift_count = divceil(0.06*n, 2)*2;
    const int aed_window_size = MAX(shift_count+2, 0.08*n);

    if (B == NULL || end-begin <= small_limit)
        return perform_small_schur_reduction(
            begin, end, n, ldQ, ldZ, ldA, ldB, lwork, real, imag, beta,
            Q, Z, A, B, work);

    STARNEIG_SANITY_CHECK(
        get_schur_reduction_workspace(
            begin, end, n, ldQ, ldZ, ldA, ldB) <= lwork,
        "Invalid workspace size.");

    STARNEIG_SANITY_CHECK(
        begin == 0 || _A(begin,begin-1) == 0.0, "Invalid matrix.");
    STARNEIG_SANITY_CHECK(
        end == n || _A(end,end-1) == 0.0, "Invalid matrix.");
    STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    // prepare workspace for AED

    double *__A = NULL, *__B = NULL, *__Q = NULL, *__Z = NULL,
        *__hT = NULL, *__vT = NULL, *__work = NULL;
    int __lwork = 0, __ld = 0, __ldhT = 0, __ldvT = 0;
    if (small_limit < end-begin) {
        __ld = divceil(aed_window_size, 8)*8;
        __ldhT = divceil(aed_window_size, 8)*8;
        __ldvT = divceil(n, 8)*8;

        __work = work;
        __lwork = lwork;

        #define add_work(__X, __X_size) \
            __X = __work; __work += __X_size; __lwork -= __X_size

        if (B != NULL) {
            add_work(__A, aed_window_size*__ld);
            add_work(__B, aed_window_size*__ld);
            add_work(__Q, aed_window_size*__ld);
            add_work(__Z, aed_window_size*__ld);
        }
        else {
            add_work(__A, aed_window_size*__ld);
            add_work(__Q, aed_window_size*__ld);
            __Z = __Q;
        }
        add_work(__hT, 0);
        add_work(__vT, MAX(n*__ldhT, aed_window_size*__ldvT));

        #undef add_work
    }

    int top = begin;
    int bottom = end;

    // deflate infinite eigenvalues

    top =
        starneig_deflate_inf_top(
            top, bottom, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    // main loop

    for (int i = 0; i < max_iter && small_limit < bottom-top; i++) {
        int aed_begin = MAX(top, bottom - aed_window_size);

        // copy AED window and initialize local transformation matrices

        starneig_copy_matrix(bottom-aed_begin, bottom-aed_begin, ldA, __ld,
            sizeof(double), _A_offset(aed_begin, aed_begin), __A);
        starneig_init_local_q(bottom-aed_begin, __ld, __Q);
        if (B != NULL) {
            starneig_copy_matrix(bottom-aed_begin, bottom-aed_begin, ldB, __ld,
                sizeof(double), _B_offset(aed_begin, aed_begin), __B);
            starneig_init_local_q(bottom-aed_begin, __ld, __Z);
        }

        // perform AED

        int unconverged, converged;
        perform_aggressively_deflate(
            bottom-aed_begin, __ld, __ld, __ld, __ld, __lwork,
            thres_a, thres_b, thres_inf, real+aed_begin, imag+aed_begin,
            __Q, __Z, __A, __B, __work, &unconverged, &converged);

        // if AED managed to deflate eigenvalues, apply it

        if (0 < converged) {

            STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
                SANITY_2, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

            // copy AED window back
            starneig_copy_matrix(bottom-aed_begin, bottom-aed_begin, __ld, ldA,
                sizeof(double), __A, _A_offset(aed_begin, aed_begin));
            if (B != NULL)
                starneig_copy_matrix(
                    bottom-aed_begin, bottom-aed_begin, __ld, ldB,
                    sizeof(double), __B, _B_offset(aed_begin, aed_begin));

            starneig_small_gemm_updates(aed_begin, bottom, n,
                __ld, __ld, ldQ, ldZ, ldA, ldB, __ldhT, __ldvT,
                __Q, __Z, Q, Z, A, B, __hT, __vT);

            STARNEIG_SANITY_CHECK_HESSENBERG(begin, end, n, ldA, ldB, A, B);
            STARNEIG_SANITY_CHECK_RESIDUALS_END(
                SANITY_2, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

            starneig_extract_eigenvalues(converged, ldA, ldB,
                _A_offset(bottom-converged, bottom-converged),
                _B_offset(bottom-converged, bottom-converged),
                real+bottom-converged, imag+bottom-converged,
                beta+bottom-converged);

            bottom -= converged;
        }

        // if there is not enough shifts, repeat AED

        if (unconverged < shift_count && 0 < converged)
            continue;

        // push bulges

        perform_push_bulges(
            BULGE_CHASING_MODE_FULL, top, bottom, shift_count, n,
            ldQ, ldZ, ldA, ldB, lwork, thres_a, thres_b, thres_inf,
            real+aed_begin, imag+aed_begin, Q, Z, A, B, work);

        // deflate infinite eigenvalues

        top = starneig_deflate_inf_top(
            top, bottom, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

        // vigilant deflation check

        for (int j = bottom-1; top < j; j--) {
            if (_A(j,j-1) == 0.0) {
                // 1x1 block
                if (bottom-j < 2) {
                    bottom = j;
                }
                // 2x2 block
                else if (bottom-j == 2) {
                    process_2x2_block(
                        j, n, ldQ, ldZ, ldA, ldB, real, imag, beta, Q, Z, A, B);
                    bottom = j;
                }
                else {
                    int _bottom = perform_schur_reduction(
                        j, bottom, n, ldQ, ldZ, ldA, ldB, lwork,
                        thres_a, thres_b, thres_inf, real, imag, beta,
                        Q, Z, A, B, work);
                    if (_bottom == j)
                        bottom = _bottom;
                }
            }
        }
    }

    // reduce the remaining active region

    if (0 < bottom-top && bottom-top <= small_limit)
        bottom = perform_small_schur_reduction(
            top, bottom, n, ldQ, ldZ, ldA, ldB, lwork,
            real, imag, beta, Q, Z, A, B, work);

    STARNEIG_SANITY_CHECK_HESSENBERG(top, bottom, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_SCHUR(bottom, end, n, ldA, ldB, A, B);
    STARNEIG_SANITY_CHECK_RESIDUALS_END(
        SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    if (top == bottom)
        return begin;
    else
        return bottom;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int starneig_move_block(
    int from, int to, int n, int ldQ, int ldZ, int ldA, int ldB, int lwork,
    double *Q, double *Z, double *A, double *B, double *work)
{
    extern void dtrexc_(
        char const *, int const *, double *, int const *, double *, int const *,
        int *, int *, double *, int *);

    extern void dtgexc_(
        int const *, int const *, int const *, double *, int const *,
        double *, int const *, double *, int const *, double *, int const *,
        int const *, int const *, double *, int const *, int *);

    STARNEIG_SANITY_CHECK(
        (B != NULL && 4*n+16 <= lwork) || (B == NULL && n <= lwork),
        "Invalid workspace size.");

    STARNEIG_SANITY_CHECK_SCHUR(to, from, n, ldA, ldB, A, B);
    //STARNEIG_SANITY_CHECK_RESIDUALS_BEGIN(
    //    SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    int info, one = 1, _from = from+1, _to = to+1;

    if (B != NULL)
        dtgexc_(&one, &one, &n, A, &ldA, B, &ldB, Q, &ldQ, Z, &ldZ, &_from,
            &_to, work, &lwork, &info);
    else
        dtrexc_("V", &n, A, &ldA, Q, &ldQ, &_from, &_to, work, &info);

    if (info < 0)
        starneig_fatal_error(
            "An unrecoverable internal error occured while calling dtrexc "
            "or dtgexc.");

    STARNEIG_SANITY_CHECK_SCHUR(to, from, n, ldA, ldB, A, B);
    //STARNEIG_SANITY_CHECK_RESIDUALS_END(
    //    SANITY_1, n, ldQ, ldZ, ldA, ldB, Q, Z, A, B);

    return _to-1;
}

void starneig_push_bulges(
    bulge_chasing_mode_t mode, int shifts, int n,
    int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double const *real, double const *imag,
    double *Q, double *Z, double *A, double *B)
{
    size_t lwork = get_push_bulges_workspace(n, ldQ, ldZ, ldA, ldB);
    double *work = NULL;
    if (0 < lwork)
        work = malloc(lwork*sizeof(double));

    perform_push_bulges(
        mode, 0, n, shifts, n, ldQ, ldZ, ldA, ldB, lwork,
        thres_a, thres_b, thres_inf, real, imag, Q, Z, A, B, work);

    free(work);
}

void starneig_aggressively_deflate(
    int n, int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double *real, double *imag, double *Q, double *Z, double *A, double *B,
    int *unconverged, int *converged)
{
    size_t lwork = get_aggressively_deflate_workspace(n, ldQ, ldZ, ldA, ldB);
    double *work = NULL;
    if (0 < lwork)
        work = malloc(lwork*sizeof(double));

    perform_aggressively_deflate(
        n, ldQ, ldZ, ldA, ldB, lwork, thres_a, thres_b, thres_inf,
        real, imag, Q, Z, A, B, work, unconverged, converged);

    free(work);
}

int starneig_schur_reduction(
    int n, int ldQ, int ldZ, int ldA, int ldB,
    double thres_a, double thres_b, double thres_inf,
    double *real, double *imag, double *beta,
    double *Q, double *Z, double *A, double *B)
{
    size_t lwork = get_schur_reduction_workspace(0, n, n, ldQ, ldZ, ldA, ldB);
    double *work = NULL;
    if (0 < lwork)
        work = malloc(lwork*sizeof(double));

    int bottom = perform_schur_reduction(
        0, n, n, ldQ, ldZ, ldA, ldB, lwork, thres_a, thres_b, thres_inf,
        real, imag, beta, Q, Z, A, B, work);

    free(work);

    return bottom;
}

int starneig_hessenberg_reduction(
    int n, int ldQ, int ldZ, int ldA, int ldB,
    double *Q, double *Z, double *A, double *B)
{
    size_t lwork = get_hessenberg_reduction_workspace(
        0, n, n, ldQ, ldZ, ldA, ldB);
    double *work = NULL;
    if (0 < lwork)
        work = malloc(lwork*sizeof(double));

    int bottom = perform_hessenberg_reduction(
        0, n, n, ldQ, ldZ, ldA, ldB, lwork, Q, Z, A, B, work);

    free(work);

    return bottom;
}

void starneig_extract_eigenvalues(int n, int ldA, int ldB,
    double const *A, double const *B, double *real, double *imag, double *beta)
{
    for (int i = 0; i < n; i++) {
        if (i+1 < n && _A(i+1,i) != 0.0) {
            starneig_compute_complex_eigenvalue(
                ldA, ldB, _A_offset(i,i), _B_offset(i,i),
                &real[i], &imag[i], &real[i+1], &imag[i+1],
                beta ? &beta[i] : NULL, beta ? &beta[i+1] : NULL);
            i++;
        }
        else {
            if (B != NULL) {
                if (beta != NULL) {
                    real[i] = _A(i,i);
                    beta[i] = _B(i,i);
                }
                else {
                    real[i] = _A(i,i)/_B(i,i);
                }
            }
            else {
                real[i] = _A(i,i);
            }
            imag[i] = 0.0;
        }
    }
}

int starneig_extract_shifts(int n, int ldA, int ldB,
    double const *A, double const *B, double *real, double *imag)
{
    starneig_extract_eigenvalues(n, ldA, ldB, A, B, real, imag, NULL);

    // move all zero and infinite eigenvalues to the end
    int end = n;
    {
        int i = end-1;
        while (0 <= i) {
            if ((real[i] == 0.0 && imag[i] == 0.0) ||
            (isinf(real[i]) || isinf(imag[i])) ||
            (isnan(real[i]) || isnan(imag[i]))) {
                {
                    double swap = real[end-1];
                    real[end-1] = real[i];
                    real[i] = swap;
                }
                {
                    double swap = imag[end-1];
                    imag[end-1] = imag[i];
                    imag[i] = swap;
                }
                end--;
            }
            i--;
        }
    }

    // order smallest eigenvalues to the beginning of the buffer
    int ordered = 0;
    while(!ordered) {
        ordered = 1;
        for (int i = 0; i+1 < end; i++) {
            double norm1 = fabs(real[i]) + fabs(imag[i]);
            double norm2 = fabs(real[i+1]) + fabs(imag[i+1]);
            if (norm2 < norm1 || norm1 == 0.0) {
                {
                    double swap = real[i];
                    real[i] = real[i+1];
                    real[i+1] = swap;
                }
                {
                    double swap = imag[i];
                    imag[i] = imag[i+1];
                    imag[i+1] = swap;
                }
                ordered = 0;
            }
        }
    }

    // shuffle shifts into pairs of real shifts and pairs of complex conjugate
    // shifts
    for (int i = 0; i+2 < end; i += 2) {
        if (imag[i] != -imag[i+1]) {
            {
                double swap = real[i];
                real[i] = real[i+1];
                real[i+1] = real[i+2];
                real[i+2] = swap;
            }
            {
                double swap = imag[i];
                imag[i] = imag[i+1];
                imag[i+1] = imag[i+2];
                imag[i+2] = swap;
            }
        }
    }

    return end;
}
