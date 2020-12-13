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

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "solvers.h"
#include "../common/common.h"
#include "../common/parse.h"
#include "../common/threads.h"
#include "../common/local_pencil.h"
#include <starneig/starneig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef STARNEIG_ENABLE_MPI
#include "../common/starneig_pencil.h"
#endif


static hook_solver_state_t lapack_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return (hook_solver_state_t) env->data;
}

static int lapack_finalize(hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int lapack_run(hook_solver_state_t state)
{
    pencil_t data = (pencil_t) state;

    extern void dhseqr_(
        char const *,       // job
        char const *,       // compz
        int const *,        // order of the matrices H and Q
        int const *,        // ilo
        int const *,        // ihi
        double *,           // matrix H
        int const *,        // matrix H leading dimension
        double *,           // wr (real part)
        double *,           // wi (imaginary part)
        double *,           // matrix Q
        int const *,        // matrix Q leading dimension
        double *,           // work
        int const *,        // lwork
        int *);             // info

    extern void dhgeqz_(
        char const *,       // job
        char const *,       // compq
        char const *,       // compz
        int const *,        // order of the matrices H, T, Q and Z
        int const *,        // ilo
        int const *,        // ihi
        double *,           // matrix H
        int const *,        // matrix H leading dimension
        double *,           // matrix T
        int const *,        // matrix T leading dimension
        double *,           // alphar (real part)
        double *,           // alphai (imaginary part)
        double *,           // beta
        double *,           // matrix Q
        int const *,        // matrix Q leading dimension
        double *,           // matrix Z
        int const *,        // matrix Z leading dimension
        double *,           // work
        int const *,        // lwork
        int *);             // info

    int n = LOCAL_MATRIX_N(data->mat_a);
    double *A = LOCAL_MATRIX_PTR(data->mat_a);
    int ldA = LOCAL_MATRIX_LD(data->mat_a);
    double *Q = LOCAL_MATRIX_PTR(data->mat_q);
    int ldQ = LOCAL_MATRIX_LD(data->mat_q);

    double *B = NULL;
    int ldB = 0;
    double *Z = NULL;
    int ldZ = 0;

    if (data->mat_b != NULL) {
        B = LOCAL_MATRIX_PTR(data->mat_b);
        ldB = LOCAL_MATRIX_LD(data->mat_b);
        Z = LOCAL_MATRIX_PTR(data->mat_z);
        ldZ = LOCAL_MATRIX_LD(data->mat_z);
    }

    int info, lwork = -1, ilo = 1, ihi = n;
    double dlwork;

    double *work = NULL;
    double *wr = NULL;
    double *wi = NULL;
    double *beta = NULL;

    // request optimal work space size
    if (B != NULL)
        dhgeqz_("S", "V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, wr, wi, beta, Q, &ldQ, Z, &ldZ,
            &dlwork, &lwork, &info);
    else
        dhseqr_("S", "V", &n, &ilo, &ihi,
            A, &ldA, wr, wi, Q, &ldQ, &dlwork, &lwork, &info);

    if (info != 0)
        goto finalize;

    lwork = dlwork;
    work = malloc(lwork*sizeof(double));

    init_supplementary_eigenvalues(n, &wr, &wi, &beta, &data->supp);

    threads_set_mode(THREADS_MODE_LAPACK);

    // reduce
    if (B != NULL)
        dhgeqz_("S", "V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, wr, wi, beta, Q, &ldQ, Z, &ldZ,
            work, &lwork, &info);
    else
        dhseqr_("S", "V", &n, &ilo, &ihi,
            A, &ldA, wr, wi, Q, &ldQ, work, &lwork, &info);

    threads_set_mode(THREADS_MODE_DEFAULT);

finalize:

    free(work);
    return info;
}

struct hook_solver schur_lapack_solver = {
    .name = "lapack",
    .desc = "dhseqr/dhgeqz subroutine from LAPACK",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .prepare = &lapack_prepare,
    .finalize = &lapack_finalize,
    .run = &lapack_run
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef PDLAHQR_FOUND

static hook_solver_state_t pdlahqr_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return env->data;
}

static int pdlahqr_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int pdlahqr_has_valid_descr(
    int matrix_size, int section_size, starneig_blacs_descr_t const *descr)
{
    if (descr->m != matrix_size || descr->n != matrix_size)
        return 0;
    if (descr->sm != section_size || descr->sn != section_size)
        return 0;
    return 1;
}

static int pdlahqr_run(hook_solver_state_t state)
{
    extern void pdlahqr_(
        int const *,                // wantT
        int const *,                // wantZ
        int const *,                // n
        int const *,                // ilo
        int const *,                // ihi
        double *,                   // A
        starneig_blacs_descr_t const *, // descA
        double *,                   // wr
        double *,                   // wi
        int const *,                // iloz
        int const *,                // ihiz
        double *,                   // Z
        starneig_blacs_descr_t const *, // descZ
        double *,                   // work
        int const *,                // lwork
        int *,                      // iwork
        int const *,                // liwork
        int *);                     // info

    pencil_t pencil = ((pencil_t) state);

    if (pencil->mat_a == NULL) {
        fprintf(stderr, "Missing matrix A.\n");
        return -1;
    }

    if (pencil->mat_b != NULL) {
        fprintf(stderr, "Solver does not support generalized cases.\n");
        return -1;
    }

    int n = STARNEIG_MATRIX_N(pencil->mat_a);
    int sn = STARNEIG_MATRIX_BN(pencil->mat_a);

    starneig_distr_t distr = STARNEIG_MATRIX_DISTR(pencil->mat_a);
    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    starneig_blacs_descr_t desc_a, desc_q;
    double *local_a, *local_q;
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_a, context, &desc_a, (void **)&local_a);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_q, context, &desc_q, (void **)&local_q);

    if (!pdlahqr_has_valid_descr(n, sn, &desc_a)) {
        fprintf(stderr, "Matrix A has invalid dimensions.\n");
        return -1;
    }

    if (pencil->mat_q != NULL && !pdlahqr_has_valid_descr(n, sn, &desc_q)) {
        fprintf(stderr, "Matrix Q has invalid dimension.\n");
        return -1;
    }

    int wantT = 1;
    int wantZ = pencil->mat_q != NULL;
    int ilo = 1, ihi = n, iloz = 1, ihiz = n;

    double *wr, *wi, *beta;
    init_supplementary_eigenvalues(n, &wr, &wi, &beta, &pencil->supp);

    int lwork = 3*n +
        MAX(
            2*MAX(desc_q.lld, desc_a.lld) + 2*n,
            7*divceil(n, sn)
        );
    double *work = malloc(lwork*sizeof(double));

    int liwork = n;
    int *iwork = malloc(liwork*sizeof(int));

    int info;

    threads_set_mode(THREADS_MODE_SCALAPACK);

    pdlahqr_(&wantT, &wantZ, &n, &ilo, &ihi, local_a, &desc_a,
        wr, wi, &iloz, &ihiz, local_q, &desc_q, work, &lwork,
        iwork, &liwork, &info);

    threads_set_mode(THREADS_MODE_DEFAULT);

    starneig_blacs_gridexit(context);

    free(work);
    free(iwork);

    return info;
}

const struct hook_solver schur_pdlahqr_solver = {
    .name = "pdlahqr",
    .desc = "pdlahqr subroutine from scaLAPACK (old)",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_BLACS, 0 },
    .prepare = &pdlahqr_prepare,
    .finalize = &pdlahqr_finalize,
    .run = &pdlahqr_run
};

#endif // PDLAHQR_FOUND

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef PDHSEQR_FOUND

static hook_solver_state_t pdhseqr_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return env->data;
}

static int pdhseqr_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int pdhseqr_has_valid_descr(
    int matrix_size, int section_size, starneig_blacs_descr_t const *descr)
{
    if (descr->m != matrix_size || descr->n != matrix_size)
        return 0;
    if (descr->sm != section_size || descr->sn != section_size)
        return 0;
    return 1;
}

static int pdhseqr_run(hook_solver_state_t state)
{
    extern void pdhseqr_(
        char const *,               // job
        char const *,               // compZ
        int const *,                // n
        int const *,                // ilo
        int const *,                // ihi
        double *,                   // A
        starneig_blacs_descr_t const *, // descA
        double *,                   // wr
        double *,                   // wi
        double *,                   // Z
        starneig_blacs_descr_t const *, // descZ
        double *,                   // work
        int const *,                // lwork
        int *,                      // iwork
        int const *,                // liwork
        int *);                     // info

    pencil_t pencil = ((pencil_t) state);

    if (pencil->mat_a == NULL) {
        fprintf(stderr, "Missing matrix A.\n");
        return -1;
    }

    if (pencil->mat_b != NULL) {
        fprintf(stderr, "Solver does not support generalized cases.\n");
        return -1;
    }

    int n = STARNEIG_MATRIX_N(pencil->mat_a);
    int sn = STARNEIG_MATRIX_BN(pencil->mat_a);

    starneig_distr_t distr = STARNEIG_MATRIX_DISTR(pencil->mat_a);
    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    starneig_blacs_descr_t desc_a, desc_q;
    double *local_a, *local_q;
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_a, context, &desc_a, (void **)&local_a);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_q, context, &desc_q, (void **)&local_q);

    if (!pdhseqr_has_valid_descr(n, sn, &desc_a)) {
        fprintf(stderr, "Matrix A has invalid dimensions.\n");
        return -1;
    }

    if (pencil->mat_q != NULL && !pdhseqr_has_valid_descr(n, sn, &desc_q)) {
        fprintf(stderr, "Matrix Q has invalid dimension.\n");
        return -1;
    }

    // large section/block sizes may lead to weird behavior (wrong results,
    // segmentation faults, incorrect workspace sizes, etc)
    int sn_limit = 80;
    if (sn_limit < sn)
        fprintf(stderr,
            "Warning: pdhseqr may fail when the section/block size is larger "
            "than %d.\n", sn_limit);

    int ilo = 1, ihi = n, info;

    double *wr = NULL;
    double *wi = NULL;
    double *beta = NULL;

    int lwork = 4000000, liwork = 1000000;
    double *work = malloc(lwork*sizeof(double));
    int *iwork = malloc(liwork*sizeof(int));

    pdhseqr_("S", pencil->mat_q != NULL ? "V" : "N", &n, &ilo, &ihi,
        local_a, &desc_a, wr, wi, local_q, &desc_q,
        work, (const int[]){-1}, iwork, (const int[]){-1}, &info);

    if (info)
        goto cleanup;

    init_supplementary_eigenvalues(n, &wr, &wi, &beta, &pencil->supp);

    if (lwork < 1.15*work[0]) {
        lwork = 1.15*work[0];
        free(work);
        work = malloc(lwork*sizeof(double));
    }

    if (liwork < 1.15*iwork[0]) {
        liwork = 1.15*iwork[0];
        free(iwork);
        iwork = malloc(liwork*sizeof(int));
    }

    threads_set_mode(THREADS_MODE_SCALAPACK);

    pdhseqr_("S", pencil->mat_q != NULL ? "V" : "N", &n, &ilo, &ihi,
        local_a, &desc_a, wr, wi, local_q, &desc_q,
        work, &lwork, iwork, &liwork, &info);

    threads_set_mode(THREADS_MODE_DEFAULT);

cleanup:

    free(work);
    free(iwork);

    return info;
}

const struct hook_solver schur_pdhseqr_solver = {
    .name = "pdhseqr",
    .desc = "pdhseqr subroutine from scaLAPACK (new)",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_BLACS, 0 },
    .prepare = &pdhseqr_prepare,
    .finalize = &pdhseqr_finalize,
    .run = &pdhseqr_run
};


#endif // PDHSEQR_FOUND

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef CUSTOM_PDHSEQR

static hook_solver_state_t custom_pdhseqr_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return env->data;
}

static int custom_pdhseqr_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int custom_pdhseqr_has_valid_descr(
    int matrix_size, int section_size, starneig_blacs_descr_t const *descr)
{
    if (descr->m != matrix_size || descr->n != matrix_size)
        return 0;
    if (descr->sm != section_size || descr->sn != section_size)
        return 0;
    return 1;
}

static int custom_pdhseqr_run(hook_solver_state_t state)
{
    extern void pdhseqr__(
        char const *,               // job
        char const *,               // compZ
        int const *,                // n
        int const *,                // ilo
        int const *,                // ihi
        double *,                   // A
        starneig_blacs_descr_t const *, // descA
        double *,                   // wr
        double *,                   // wi
        double *,                   // Z
        starneig_blacs_descr_t const *, // descZ
        double *,                   // work
        int const *,                // lwork
        int *,                      // iwork
        int const *,                // liwork
        int *);                     // info

    pencil_t pencil = ((pencil_t) state);

    if (pencil->mat_a == NULL) {
        fprintf(stderr, "Missing matrix A.\n");
        return -1;
    }

    if (pencil->mat_b != NULL) {
        fprintf(stderr, "Solver does not support generalized cases.\n");
        return -1;
    }

    int n = STARNEIG_MATRIX_N(pencil->mat_a);
    int sn = STARNEIG_MATRIX_BN(pencil->mat_a);

    starneig_distr_t distr = STARNEIG_MATRIX_DISTR(pencil->mat_a);
    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    starneig_blacs_descr_t desc_a, desc_q;
    double *local_a, *local_q;
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_a, context, &desc_a, (void **)&local_a);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_q, context, &desc_q, (void **)&local_q);

    if (!custom_pdhseqr_has_valid_descr(n, sn, &desc_a)) {
        fprintf(stderr, "Matrix A has invalid dimensions.\n");
        return -1;
    }

    if (pencil->mat_q != NULL &&
    !custom_pdhseqr_has_valid_descr(n, sn, &desc_q)) {
        fprintf(stderr, "Matrix Q has invalid dimension.\n");
        return -1;
    }

    // large section/block sizes may lead to weird behavior (wrong results,
    // segmentation faults, incorrect workspace sizes, etc)
    int sn_limit = 80;
    if (sn_limit < sn)
        fprintf(stderr,
            "Warning: pdhseqr may fail when the section/block size is larger "
            "than %d.\n", sn_limit);

    int ilo = 1, ihi = n, info;

    double *wr = NULL;
    double *wi = NULL;
    double *beta = NULL;

    int lwork = 4000000, liwork = 1000000;
    double *work = malloc(lwork*sizeof(double));
    int *iwork = malloc(liwork*sizeof(int));

    pdhseqr__("S", pencil->mat_q != NULL ? "V" : "N", &n, &ilo, &ihi,
        local_a, &desc_a, wr, wi, local_q, &desc_q,
        work, (const int[]){-1}, iwork, (const int[]){-1}, &info);

    if (info)
        goto cleanup;

    init_supplementary_eigenvalues(n, &wr, &wi, &beta, &pencil->supp);

    if (lwork < 3.15*work[0]) {
        lwork = 3.15*work[0];
        free(work);
        work = malloc(lwork*sizeof(double));
    }

    if (liwork < 3.15*iwork[0]) {
        liwork = 3.15*iwork[0];
        free(iwork);
        iwork = malloc(liwork*sizeof(int));
    }

    threads_set_mode(THREADS_MODE_SCALAPACK);

    pdhseqr__("S", pencil->mat_q != NULL ? "V" : "N", &n, &ilo, &ihi,
        local_a, &desc_a, wr, wi, local_q, &desc_q,
        work, &lwork, iwork, &liwork, &info);

    threads_set_mode(THREADS_MODE_DEFAULT);

cleanup:

    free(work);
    free(iwork);

    return info;
}

const struct hook_solver schur_custom_pdhseqr_solver = {
    .name = "custom_pdhseqr",
    .desc = "custom pdhseqr subroutine",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_BLACS, 0 },
    .prepare = &custom_pdhseqr_prepare,
    .finalize = &custom_pdhseqr_finalize,
    .run = &custom_pdhseqr_run
};


#endif // CUSTOM_PDHSEQR

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef PDHGEQZ_FOUND

static hook_solver_state_t pdhgeqz_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return env->data;
}

static int pdhgeqz_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int pdhgeqz_has_valid_descr(
    int matrix_size, int section_size, starneig_blacs_descr_t const *descr)
{
    if (descr->m != matrix_size || descr->n != matrix_size)
        return 0;
    if (descr->sm != section_size || descr->sn != section_size)
        return 0;
    return 1;
}

static int pdhgeqz_run(hook_solver_state_t state)
{
    extern double pdlamch_(starneig_blacs_context_t const *, char const *);
    extern void kkqzconf_(double const *);

    extern void pdhgeqz_(
        char const *,               // job
        char const *,               // compQ
        char const *,               // compZ
        int const *,                // n
        int const *,                // ilo
        int const *,                // ihi
        double *,                   // H
        starneig_blacs_descr_t const *, // descH
        double *,                   // T
        starneig_blacs_descr_t const *, // descT
        double *,                   // alphar
        double *,                   // alphai
        double *,                   // beta
        double *,                   // Q
        starneig_blacs_descr_t const *, // descQ
        double *,                   // Z
        starneig_blacs_descr_t const *, // descZ
        double *,                   // work
        int const *,                // lwork
        int *,                      // iwork
        int const *,                // liwork
        int *);                     // info

    pencil_t pencil = ((pencil_t) state);

    if (pencil->mat_a == NULL) {
        fprintf(stderr, "Missing matrix A.\n");
        return -1;
    }

    if (pencil->mat_b == NULL) {
        fprintf(stderr, "Missing matrix B.\n");
        return -1;
    }

    int n = STARNEIG_MATRIX_N(pencil->mat_a);
    int sn = STARNEIG_MATRIX_BN(pencil->mat_a);

    starneig_distr_t distr = STARNEIG_MATRIX_DISTR(pencil->mat_a);
    starneig_blacs_context_t context = starneig_distr_to_blacs_context(distr);

    starneig_blacs_descr_t desc_a, desc_b, desc_q, desc_z;
    double *local_a, *local_b, *local_q, *local_z;
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_a, context, &desc_a, (void **)&local_a);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_b, context, &desc_b, (void **)&local_b);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_q, context, &desc_q, (void **)&local_q);
    STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
        pencil->mat_z, context, &desc_z, (void **)&local_z);

    if (!pdhgeqz_has_valid_descr(n, sn, &desc_a)) {
        fprintf(stderr, "Matrix A has invalid dimensions.\n");
        return -1;
    }

    if (!pdhgeqz_has_valid_descr(n, sn, &desc_b)) {
        fprintf(stderr, "Matrix B has invalid dimensions.\n");
        return -1;
    }

    if (pencil->mat_q != NULL && !pdhgeqz_has_valid_descr(n, sn, &desc_q)) {
        fprintf(stderr, "Matrix Q has invalid dimension.\n");
        return -1;
    }

    if (pencil->mat_z != NULL && !pdhgeqz_has_valid_descr(n, sn, &desc_z)) {
        fprintf(stderr, "Matrix Z has invalid dimension.\n");
        return -1;
    }

    int ilo = 1, ihi = n, info;

    double *wr = NULL;
    double *wi = NULL;
    double *beta = NULL;

    double eps = pdlamch_(&context, "PRECISION");
    kkqzconf_(&eps);

    int lwork, liwork;
    double *work = NULL;
    int *iwork = NULL;
    {
        double _work;
        int _iwork;
        pdhgeqz_("S",
            pencil->mat_q != NULL ? "V" : "N",
            pencil->mat_z != NULL ? "V" : "N",
            &n, &ilo, &ihi, local_a, &desc_a, local_b, &desc_b, wr, wi, beta,
            local_q, &desc_q, local_z, &desc_z,
            &_work, (const int[]){-1}, &_iwork, (const int[]){-1}, &info);
        lwork = _work;
        liwork = _iwork;
    }

    if (info)
        goto cleanup;

    init_supplementary_eigenvalues(n, &wr, &wi, &beta, &pencil->supp);

    work = malloc(lwork*sizeof(double));
    iwork = malloc(liwork*sizeof(int));

    threads_set_mode(THREADS_MODE_SCALAPACK);

    pdhgeqz_("S",
        pencil->mat_q != NULL ? "V" : "N",
        pencil->mat_z != NULL ? "V" : "N",
        &n, &ilo, &ihi, local_a, &desc_a, local_b, &desc_b, wr, wi, beta,
        local_q, &desc_q, local_z, &desc_z,
        work, &lwork, iwork, &liwork, &info);

    threads_set_mode(THREADS_MODE_DEFAULT);

cleanup:

    free(work);
    free(iwork);

    return info;
}

const struct hook_solver schur_pdhgeqz_solver = {
    .name = "pdhgeqz",
    .desc = "pdhgeqz subroutine",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_BLACS, 0 },
    .prepare = &pdhgeqz_prepare,
    .finalize = &pdhgeqz_finalize,
    .run = &pdhgeqz_run
};


#endif // PDHGEQZ_FOUND

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct starpu_state {
    int argc;
    char * const *argv;
    struct hook_data_env *env;
};

static void starpu_print_usage(int argc, char * const *argv)
{
    printf(
        "  --cores [default,(num)} -- Number of CPU cores\n"
        "  --gpus [default,(num)} -- Number of GPUS\n"
        "  --iteration-limit [default,(num)] -- Iteration limit\n"
        "  --tile-size [default,(num)] -- Tile size\n"
        "  --small-limit [default,(num)] -- Sequential QR switching "
        " point\n"
        "  --aed-window-size [default,(num)] -- AED window size\n"
        "  --aed-nibble [default,(1-99)] -- Nibble point point\n"
        "  --aed-parallel-soft-limit [default,(num)] -- Soft sequential"
        " AED switching point\n"
        "  --aed-parallel-hard-limit [default,(num)] -- Hard sequential"
        " AED switching point\n"
        "  --shift-count [default,(num)] -- Shift count\n"
        "  --window-size [default,rounded,(num)] -- Window size\n"
        "  --shifts-per-window [default,(num)] -- Shifts per window\n"
        "  --update-width [default,(num)] -- Update task width\n"
        "  --update-height [default,(num)] -- Update task width\n"
        "  --left-threshold [default,norm,lapack,(num)] -- Left-hand"
        " side deflation threshold\n"
        "  --right-threshold [default,norm,lapack,(num)] -- Right-hand"
        " side deflation threshold\n"
        "  --inf-threshold [default,norm,(num)] -- Infinite eigenvalue"
        " threshold\n"
    );
}

static int starpu_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    struct multiarg_t iteration_limit = read_multiarg(
        "--iteration-limit", argc, argv, argr, "default", NULL);
    if (iteration_limit.type == MULTIARG_INVALID ||
    (iteration_limit.type == MULTIARG_INT && iteration_limit.int_value < 1)) {
        fprintf(stderr, "Invalid iteration limit.\n");
        return -1;
    }

    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, argr, "default", NULL);
    if (tile_size.type == MULTIARG_INVALID ||
    (tile_size.type == MULTIARG_INT && tile_size.int_value < 1)) {
        fprintf(stderr, "Invalid tile size.\n");
        return -1;
    }

    struct multiarg_t small_limit = read_multiarg(
        "--small-limit", argc, argv, argr, "default", NULL);
    if (small_limit.type == MULTIARG_INVALID ||
    (small_limit.type == MULTIARG_INT && small_limit.int_value < 1)) {
        fprintf(stderr, "Invalid sequential QR switch point.\n");
        return -1;
    }

    struct multiarg_t aed_window_size = read_multiarg(
        "--aed-window-size", argc, argv, argr, "default", NULL);
    if (aed_window_size.type == MULTIARG_INVALID ||
    (aed_window_size.type == MULTIARG_INT && aed_window_size.int_value < 5)) {
        fprintf(stderr, "Invalid AED window size.\n");
        return -1;
    }

    struct multiarg_t shift_count = read_multiarg(
        "--shift-count", argc, argv, argr, "default", NULL);
    if (shift_count.type == MULTIARG_INVALID ||
    (shift_count.type == MULTIARG_INT && shift_count.int_value < 2)) {
        fprintf(stderr, "Invalid AED shift count.\n");
        return -1;
    }

    struct multiarg_t aed_nibble = read_multiarg(
        "--aed-nibble", argc, argv, argr, "default", NULL);
    if (aed_nibble .type == MULTIARG_INVALID ||
    (aed_nibble .type == MULTIARG_INT &&
    (aed_nibble.int_value < 1 || 99 < aed_nibble.int_value))) {
        fprintf(stderr, "Invalid aed_nibble point.\n");
        return -1;
    }

    struct multiarg_t aed_parallel_soft__limit = read_multiarg(
        "--aed-parallel-soft-limit", argc, argv, argr, "default", NULL);
    if (aed_parallel_soft__limit.type == MULTIARG_INVALID ||
    (aed_parallel_soft__limit.type == MULTIARG_INT &&
    aed_parallel_soft__limit.int_value < 1)) {
        fprintf(stderr, "Invalid soft sequential AED switching point.\n");
        return -1;
    }

    struct multiarg_t aed_parallel_hard_limit = read_multiarg(
        "--aed-parallel-hard-limit", argc, argv, argr, "default", NULL);
    if (aed_parallel_hard_limit.type == MULTIARG_INVALID ||
    (aed_parallel_hard_limit.type == MULTIARG_INT &&
    aed_parallel_hard_limit.int_value < 1)) {
        fprintf(stderr, "Invalid hard sequential AED switching point.\n");
        return -1;
    }

    struct multiarg_t window_size = read_multiarg(
        "--window-size", argc, argv, argr, "default", "rounded", NULL);
    if (window_size.type == MULTIARG_INVALID ||
    (window_size.type == MULTIARG_INT && window_size.int_value < 5)) {
        fprintf(stderr, "Invalid window size.\n");
        return -1;
    }

    struct multiarg_t shifts_per_window = read_multiarg(
        "--shifts-per-window", argc, argv, argr, "default", NULL);
    if (shifts_per_window.type == MULTIARG_INVALID ||
    (shifts_per_window.type == MULTIARG_INT &&
    shifts_per_window.int_value < 2)) {
        fprintf(stderr, "Invalid number of shifts per window.\n");
        return -1;
    }

    struct multiarg_t update_width = read_multiarg(
        "--update-width", argc, argv, argr, "default", NULL);
    if (update_width.type == MULTIARG_INVALID ||
        (update_width.type == MULTIARG_INT && update_width.int_value < 1)) {
        fprintf(stderr, "Invalid update task width.\n");
        return -1;
    }

    struct multiarg_t update_height = read_multiarg(
        "--update-height", argc, argv, argr, "default", NULL);
    if (update_height.type == MULTIARG_INVALID ||
        (update_height.type == MULTIARG_INT && update_height.int_value < 1)) {
        fprintf(stderr, "Invalid update task height.\n");
        return -1;
    }

    struct multiarg_t left_threshold = read_multiarg(
        "--left-threshold", argc, argv, argr, "default", "norm", "lapack",
        NULL);
    if (left_threshold.type == MULTIARG_INVALID ||
    (left_threshold.type != MULTIARG_STR &&
    left_threshold.double_value <= 0.0)) {
        fprintf(stderr, "Invalid left threshold.\n");
        return -1;
    }

    struct multiarg_t right_threshold = read_multiarg(
        "--right-threshold", argc, argv, argr, "default", "norm", "lapack",
        NULL);
    if (right_threshold.type == MULTIARG_INVALID ||
    (right_threshold.type != MULTIARG_STR &&
    right_threshold.double_value <= 0.0)) {
        fprintf(stderr, "Invalid right threshold.\n");
        return -1;
    }

    struct multiarg_t inf_threshold = read_multiarg(
        "--inf-threshold", argc, argv, argr, "default", "norm", NULL);
    if (inf_threshold.type == MULTIARG_INVALID ||
    (inf_threshold.type != MULTIARG_STR &&
    inf_threshold.double_value <= 0.0)) {
        fprintf(stderr, "Invalid infinity threshold.\n");
        return -1;
    }

    return 0;
}

static void starpu_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
    print_multiarg("--iteration-limit", argc, argv, "default", NULL);
    print_multiarg("--tile-size", argc, argv, "default", NULL);
    print_multiarg("--small-limit", argc, argv, "default", NULL);
    print_multiarg("--aed-window-size", argc, argv, "default", NULL);
    print_multiarg("--aed-nibble", argc, argv, "default", NULL);
    print_multiarg("--aed-parallel-soft-limit", argc, argv, "default", NULL);
    print_multiarg("--aed-parallel-hard-limit", argc, argv, "default", NULL);
    print_multiarg("--shift-count", argc, argv, "default", NULL);
    print_multiarg("--window-size", argc, argv, "default", "rounded", NULL);
    print_multiarg("--shifts-per-window", argc, argv, "default", NULL);
    print_multiarg("--update-width", argc, argv, "default", NULL);
    print_multiarg("--update-height", argc, argv, "default", NULL);
    print_multiarg("--left-threshold", argc, argv,
        "default", "norm", "lapack", NULL);
    print_multiarg("--right-threshold", argc, argv,
        "default", "norm", "lapack", NULL);
    print_multiarg("--inf-threshold", argc, argv,
        "default", "norm", NULL);
}

static hook_solver_state_t starpu_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    struct starpu_state *state = malloc(sizeof(struct starpu_state));

    state->argc = argc;
    state->argv = argv;
    state->env = env;

    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, NULL, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, NULL, "default", NULL);

    int cores = STARNEIG_USE_ALL;
    if (arg_cores.type == MULTIARG_INT)
        cores = arg_cores.int_value;

    int gpus = STARNEIG_USE_ALL;
    if (arg_gpus.type == MULTIARG_INT)
        gpus = arg_gpus.int_value;

#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS)
        starneig_node_init(cores, gpus, STARNEIG_FAST_DM);
    else
#endif
        starneig_node_init(
            cores, gpus, STARNEIG_HINT_SM | STARNEIG_AWAKE_WORKERS);

    return state;
}

static int starpu_finalize(hook_solver_state_t state, struct hook_data_env *env)
{
    if (state == NULL)
        return 0;

    starneig_node_finalize();

    free(state);
    return 0;
}

static int starpu_run(hook_solver_state_t state)
{
    int argc = ((struct starpu_state *) state)->argc;
    char * const *argv = ((struct starpu_state *) state)->argv;
    struct hook_data_env *env = ((struct starpu_state *) state)->env;

    struct starneig_schur_conf conf;
    starneig_schur_init_conf(&conf);

    struct multiarg_t iteration_limit = read_multiarg(
        "--iteration-limit", argc, argv, NULL, "default", NULL);
    if (iteration_limit.type == MULTIARG_INT)
        conf.iteration_limit = iteration_limit.int_value;

    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, NULL, "default", NULL);
    if (tile_size.type == MULTIARG_INT)
        conf.tile_size = tile_size.int_value;

    struct multiarg_t small_limit = read_multiarg(
        "--small-limit", argc, argv, NULL, "default", NULL);
    if (small_limit.type == MULTIARG_INT)
        conf.small_limit = small_limit.int_value;

    struct multiarg_t aed_window_size = read_multiarg(
        "--aed-window-size", argc, argv, NULL, "default", NULL);
    if (aed_window_size.type == MULTIARG_INT)
        conf.aed_window_size = aed_window_size.int_value;

    struct multiarg_t aed_nibble = read_multiarg(
        "--aed-nibble", argc, argv, NULL, "default", NULL);
    if (aed_nibble.type == MULTIARG_INT)
        conf.aed_nibble = aed_nibble.int_value;

    struct multiarg_t aed_parallel_soft_limit = read_multiarg(
        "--aed-parallel-soft-limit", argc, argv, NULL, "default", NULL);
    if (aed_parallel_soft_limit.type == MULTIARG_INT)
        conf.aed_parallel_soft_limit = aed_parallel_soft_limit.int_value;

    struct multiarg_t aed_parallel_hard_limit = read_multiarg(
        "--aed-parallel-hard-limit", argc, argv, NULL, "default", NULL);
    if (aed_parallel_hard_limit.type == MULTIARG_INT)
        conf.aed_parallel_hard_limit = aed_parallel_hard_limit.int_value;

    struct multiarg_t shift_count = read_multiarg(
        "--shift-count", argc, argv, NULL, "default", NULL);
    if (shift_count.type == MULTIARG_INT)
        conf.shift_count = shift_count.int_value;

    struct multiarg_t window_size = read_multiarg(
        "--window-size", argc, argv, NULL, "default", "rounded", NULL);
    if (window_size.type == MULTIARG_STR &&
    strcmp("rounded", window_size.str_value) == 0)
        conf.window_size = STARNEIG_SCHUR_ROUNDED_WINDOW_SIZE;
    if (window_size.type == MULTIARG_INT)
        conf.window_size = window_size.int_value;

    struct multiarg_t shifts_per_window = read_multiarg(
        "--shifts-per-window", argc, argv, NULL, "default", NULL);
    if (shifts_per_window.type == MULTIARG_INT)
        conf.shifts_per_window = shifts_per_window.int_value;

    struct multiarg_t update_width = read_multiarg(
        "--update-width", argc, argv, NULL, "default", NULL);
    if (update_width.type == MULTIARG_INT)
        conf.update_width = update_width.int_value;

    struct multiarg_t update_height = read_multiarg(
        "--update-height", argc, argv, NULL, "default", NULL);
    if (update_height.type == MULTIARG_INT)
        conf.update_height = update_height.int_value;

    struct multiarg_t left_threshold = read_multiarg(
        "--left-threshold", argc, argv, NULL, "default", "norm", "lapack",
        NULL);
    if (left_threshold.type == MULTIARG_STR) {
        if (strcmp("norm", left_threshold.str_value) == 0)
            conf.left_threshold = STARNEIG_SCHUR_NORM_STABLE_THRESHOLD;
        if (strcmp("lapack", left_threshold.str_value) == 0)
            conf.left_threshold = STARNEIG_SCHUR_LAPACK_THRESHOLD;
    }
    if (left_threshold.type == MULTIARG_FLOAT)
        conf.left_threshold = left_threshold.double_value;

    struct multiarg_t right_threshold = read_multiarg(
        "--right-threshold", argc, argv, NULL, "default", "norm", "lapack",
        NULL);
    if (right_threshold.type == MULTIARG_STR) {
        if (strcmp("norm", right_threshold.str_value) == 0)
            conf.right_threshold = STARNEIG_SCHUR_NORM_STABLE_THRESHOLD;
        if (strcmp("lapack", right_threshold.str_value) == 0)
            conf.right_threshold = STARNEIG_SCHUR_LAPACK_THRESHOLD;
    }
    if (right_threshold.type == MULTIARG_FLOAT)
        conf.right_threshold = right_threshold.double_value;

    struct multiarg_t inf_threshold = read_multiarg(
        "--inf-threshold", argc, argv, NULL, "default", "norm", NULL);
    if (inf_threshold.type == MULTIARG_STR) {
        if (strcmp("norm", inf_threshold.str_value) == 0)
            conf.inf_threshold = STARNEIG_SCHUR_NORM_STABLE_THRESHOLD;
    }
    if (inf_threshold.type == MULTIARG_FLOAT)
        conf.inf_threshold = inf_threshold.double_value;

    int ret = 0;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil_t pencil = (pencil_t) env->data;

        int n = LOCAL_MATRIX_N(pencil->mat_a);
        double *real, *imag, *beta;
        init_supplementary_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

        if (pencil->mat_b != NULL)
            ret = starneig_GEP_SM_Schur_expert(
                &conf, LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                real, imag, beta);
        else
            ret = starneig_SEP_SM_Schur_expert(
                &conf, LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                real, imag);
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        pencil_t pencil = (pencil_t) env->data;

        int n = STARNEIG_MATRIX_N(pencil->mat_a);
        double *real, *imag, *beta;
        init_supplementary_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

        if (pencil->mat_b != NULL)
            ret = starneig_GEP_DM_Schur_expert(
                &conf,
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
                real, imag, beta);
        else
            ret = starneig_SEP_DM_Schur_expert(
                &conf,
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                real, imag);
    }
#endif

    return ret;
}

struct hook_solver schur_starpu_solver = {
    .name = "starneig",
    .desc = "StarPU based subroutine",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &starpu_print_usage,
    .print_args = &starpu_print_args,
    .check_args = &starpu_check_args,
    .prepare = &starpu_prepare,
    .finalize = &starpu_finalize,
    .run = &starpu_run
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void starpu_simple_print_usage(int argc, char * const *argv)
{
    printf(
        "  --cores [default,(num)} -- Number of CPU cores\n"
        "  --gpus [default,(num)} -- Number of GPUS\n"
    );
}

static void starpu_simple_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
}

static int starpu_simple_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    return 0;
}

static hook_solver_state_t starpu_simple_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, NULL, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, NULL, "default", NULL);

    int cores = STARNEIG_USE_ALL;
    if (arg_cores.type == MULTIARG_INT)
        cores = arg_cores.int_value;

    int gpus = STARNEIG_USE_ALL;
    if (arg_gpus.type == MULTIARG_INT)
        gpus = arg_gpus.int_value;

#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS)
        starneig_node_init(cores, gpus, STARNEIG_FAST_DM);
    else
#endif
        starneig_node_init(
            cores, gpus, STARNEIG_HINT_SM | STARNEIG_AWAKE_WORKERS);

    return env;
}

static int starpu_simple_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    if (state == NULL)
        return 0;

    starneig_node_finalize();
    return 0;
}

static int starpu_simple_run(hook_solver_state_t state)
{
    struct hook_data_env *env = state;

    int ret = 0;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil_t pencil = (pencil_t) env->data;

        int n = LOCAL_MATRIX_N(pencil->mat_a);
        double *real, *imag, *beta;
        init_supplementary_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

        if (pencil->mat_b != NULL)
            ret = starneig_GEP_SM_Schur(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                real, imag, beta);
        else
            ret = starneig_SEP_SM_Schur(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                real, imag);
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        pencil_t pencil = (pencil_t) env->data;

        int n = STARNEIG_MATRIX_N(pencil->mat_a);
        double *real, *imag, *beta;
        init_supplementary_eigenvalues(n, &real, &imag, &beta, &pencil->supp);

        if (pencil->mat_b != NULL)
            ret = starneig_GEP_DM_Schur(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
                real, imag, beta);
        else
            ret = starneig_SEP_DM_Schur(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                real, imag);
    }
#endif

    return ret;
}

const struct hook_solver schur_starpu_simple_solver = {
    .name = "starneig-simple",
    .desc = "StarPU based subroutine (simplified interface)",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &starpu_simple_print_usage,
    .print_args = &starpu_simple_print_args,
    .check_args = &starpu_simple_check_args,
    .prepare = &starpu_simple_prepare,
    .finalize = &starpu_simple_finalize,
    .run = &starpu_simple_run
};
