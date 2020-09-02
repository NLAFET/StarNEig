///
/// @file This file contains the eigenvalue reordering experiment solver
/// modules.
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

///
/// @brief Wrapper function of dtrsen subroutine.
///
/// @param[inout] data - matrix pencil
///
/// @return 0 if function call was successful, non-zero otherwise
///
static int dtrsen_reorder(pencil_t data)
{
    extern void dtrsen_(char*, char*, int const *, int*, double*, int*, double*,
        int*, double*, double*, int*, double*, double*, double*, int*, int*,
        int*, int*);

    int n = LOCAL_MATRIX_N(data->mat_a);
    int ldq = LOCAL_MATRIX_LD(data->mat_q);
    int lda = LOCAL_MATRIX_LD(data->mat_a);
    double *mat_q = LOCAL_MATRIX_PTR(data->mat_q);
    double *mat_a = LOCAL_MATRIX_PTR(data->mat_a);

    int const *select = get_supplementaty_selected(data->supp);
    double *alphar, *alphai, *beta;
    get_supplementaty_eigenvalues(data->supp, &alphar, &alphai, &beta);
    if (alphar == NULL)
        init_supplementary_eigenvalues(n, &alphar, &alphai, &beta, &data->supp);

    double work[n];
    int iwork;

    int one = 1;
    int info, m;
    double sep, s;
    dtrsen_("N", mat_q != NULL ? "V" : "N", select, &n, mat_a, &lda, mat_q,
        &ldq, alphar, alphai, &m, &s, &sep, work, &n, &iwork, &one, &info);

    return info;
}

///
/// @brief Wrapper function of dtgsen subroutine.
///
/// @param[inout] data - matrix pencil
///
/// @return 0 if function call was successful, non-zero otherwise
///
static int dtgsen_reorder(pencil_t data)
{
    extern void dtgsen_(int const *, int const *, int const *,
        int const *, int const *, double *, int const *, double *,
        int const *, double *, double *, double *, double *, int const *,
        double *, int const *, int *, double *, double *, double *, double *,
        int const *, int *, int const *, int *);

    int n = LOCAL_MATRIX_N(data->mat_a);
    int ldq = LOCAL_MATRIX_LD(data->mat_q);
    int ldz = LOCAL_MATRIX_LD(data->mat_z);
    int lda = LOCAL_MATRIX_LD(data->mat_a);
    int ldb = LOCAL_MATRIX_LD(data->mat_b);
    double *mat_q = LOCAL_MATRIX_PTR(data->mat_q);
    double *mat_z = LOCAL_MATRIX_PTR(data->mat_z);
    double *mat_a = LOCAL_MATRIX_PTR(data->mat_a);
    double *mat_b = LOCAL_MATRIX_PTR(data->mat_b);

    int const *select = get_supplementaty_selected(data->supp);
    double *alphar, *alphai, *beta;
    get_supplementaty_eigenvalues(data->supp, &alphar, &alphai, &beta);
    if (alphar == NULL)
        init_supplementary_eigenvalues(n, &alphar, &alphai, &beta, &data->supp);

    int lwork = 4*n+16;
    double work[lwork];

    int liwork = 1;
    int iwork[liwork];

    int zero = 0;
    int wantq = mat_q != NULL;
    int wantz = mat_z != NULL;

    int info, m;
    dtgsen_(&zero, &wantq, &wantz, select, &n, mat_a, &lda, mat_b, &ldb,
        alphar, alphai, beta, mat_q, &ldq, mat_z, &ldz, &m, NULL, NULL, NULL,
        work, &lwork, iwork, &liwork, &info);

    return info;
}

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
    pencil_t data = (pencil_t ) state;

    if (data->mat_b != NULL)
        return dtgsen_reorder(data);
    else
        return dtrsen_reorder(data);
}

const struct hook_solver reorder_lapack_solver = {
    .name = "lapack",
    .desc = "dtrsen / dtgsen subroutine from LAPACK",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .prepare = &lapack_prepare,
    .finalize = &lapack_finalize,
    .run = &lapack_run
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if defined(PDTRSEN_FOUND) || defined(PDTGSEN_FOUND)

struct scalapack_state {
    int argc;
    char * const *argv;
    struct hook_data_env *env;
};

static const int scalapack_default_flops = 25;
static const int scalapack_default_width = 32;

static void scalapack_print_usage(int argc, char * const *argv)
{
    printf(
        "  --parallel-windows [max,(num)] -- Maximum number of concurrent "
        "computational windows allowed in the algorithm.\n"
        "  --values-per-window [half,(num)] -- Number of eigenvalues in each "
        "window\n"
        "  --window-size [half,(num)] -- Window size\n"
        "  --flops (num) -- Minimal percentage of flops required for "
        "performing matrix-matrix multiplications instead of pipelined "
        "orthogonal transformations\n"
        "  --width (num) -- Width of tile column slabs for row-wise "
        "application of pipelined orthogonal transformations\n"
        "  --move-together [default,(num)] -- Maximum number of eigenvalues "
        "moved together over a process border\n");
}

static void scalapack_print_args(int argc, char * const *argv)
{
    print_multiarg("--parallel-windows", argc, argv, "max", NULL);
    print_multiarg("--values-per-window", argc, argv, "half", NULL);
    print_multiarg("--window-size", argc, argv, "half", NULL);
    printf(" --flops %d", read_int("--flops", argc, argv, NULL,
        scalapack_default_flops));
    printf(" --width %d",  read_int("--width", argc, argv, NULL,
        scalapack_default_width));
    print_multiarg("--move-together", argc, argv, "default", NULL);
}

static int scalapack_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t parallel_windows = read_multiarg(
        "--parallel-windows", argc, argv, argr, "max", NULL);
    struct multiarg_t values_per_window = read_multiarg(
        "--values-per-window", argc, argv, argr, "half", NULL);
    struct multiarg_t window_size = read_multiarg(
        "--window-size", argc, argv, argr, "half", NULL);
    int flops = read_int("--flops", argc, argv, argr, scalapack_default_flops);
    int width = read_int("--width", argc, argv, argr, scalapack_default_width);
    struct multiarg_t move_together = read_multiarg(
        "--move-together", argc, argv, argr, "default", NULL);

    if (parallel_windows.type == MULTIARG_INVALID ||
    (parallel_windows.type == MULTIARG_INT && parallel_windows.int_value < 1)) {
        fprintf(stderr, "Invalid number of concurrent windows.\n");
        return -1;
    }

    if (window_size.type == MULTIARG_INVALID ||
    (window_size.type == MULTIARG_INT && window_size.int_value < 4)) {
        fprintf(stderr, "Invalid window size.\n");
        return -1;
    }

    if (values_per_window.type == MULTIARG_INVALID ||
    (values_per_window.type == MULTIARG_INT &&
    values_per_window.int_value < 2)) {
        fprintf(stderr, "Invalid number of eigenvalues per window.\n");
        return -1;
    }

    if (flops < 0 || 100 < flops) {
        fprintf(stderr, "Invalid number of flops.\n");
        return -1;
    }

    if (width < 1) {
        fprintf(stderr, "Invalid width.\n");
        return -1;
    }

    if (move_together.type == MULTIARG_INVALID ||
    (move_together.type == MULTIARG_INT && move_together.int_value < 2)) {
        fprintf(stderr, "Invalid number of eigenvalues moved together.\n");
        return -1;
    }

    return 0;
}

static hook_solver_state_t scalapack_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    struct scalapack_state *state = malloc(sizeof(struct scalapack_state));

    state->argc = argc;
    state->argv = argv;
    state->env = env;

    return state;
}

static int scalapack_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    free(state);
    return 0;
}

static int has_valid_descr(
    int matrix_size, int section_size, starneig_blacs_descr_t const *descr)
{
    if (descr->m != matrix_size || descr->n != matrix_size)
        return 0;
    if (descr->sm != section_size || descr->sn != section_size)
        return 0;
    return 1;
}

static int scalapack_run(hook_solver_state_t state)
{
#ifdef PDTRSEN_FOUND
    extern void pdtrsen_(
        char const *,       // job
        char const *,       // compq
        int const *,        // select
        int const *,        // para
        int const *,        // n
        double *,           // A matrix
        int const *,        // ia = 1
        int const *,        // ja = 1
        starneig_blacs_descr_t const *, // A descriptor
        double *,           // Q matrix
        int const *,        // iq = 1
        int const *,        // hq = 1
        starneig_blacs_descr_t const *, // Q descriptor
        double *,           // wr
        double *,           // wi
        int *,              // m
        double *,           // s
        double *,           // sep
        double *,           // work
        int const *,        // lwork
        int *,              // iwork
        int const *,        // liwork
        int *);             // info
#endif

#ifdef PDTGSEN_FOUND
    extern void pdtgsen_(
        int const *,        // job
        int const *,        // compq
        int const *,        // compz
        int const *,        // select
        int const *,        // para
        int const *,        // n
        double *,           // S matrix
        int const *,        // is = 1
        int const *,        // js = 1
        starneig_blacs_descr_t const *, // S descriptor
        double *,           // T matrix
        int const *,        // it = 1
        int const *,        // jt = 1
        starneig_blacs_descr_t const *, // T descriptor
        double *,           // Q matrix
        int const *,        // iq = 1
        int const *,        // hq = 1
        starneig_blacs_descr_t const *, // Q descriptor
        double *,           // QZmatrix
        int const *,        // iz = 1
        int const *,        // hz = 1
        starneig_blacs_descr_t const *, // Z descriptor
        double *,           // wr
        double *,           // wi
        double *,           // beta
        int *,              // m
        double *,           // pl
        double *,           // pr
        double *,           // dif
        double *,           // work
        int const *,        // lwork
        int *,              // iwork
        int const *,        // liwork
        int *);             // info
#endif

    int argc = ((struct scalapack_state *) state)->argc;
    char * const *argv = ((struct scalapack_state *) state)->argv;
    struct hook_data_env *env = ((struct scalapack_state *) state)->env;
    pencil_t pencil = env->data;

    struct multiarg_t parallel_windows = read_multiarg(
        "--parallel-windows", argc, argv, NULL, "max", NULL);
    struct multiarg_t values_per_window = read_multiarg(
        "--values-per-window", argc, argv, NULL, "half", NULL);
    struct multiarg_t window_size = read_multiarg(
        "--window-size", argc, argv, NULL, "half", NULL);
    int flops = read_int("--flops", argc, argv, NULL, scalapack_default_flops);
    int width = read_int("--width", argc, argv, NULL, scalapack_default_width);
    struct multiarg_t move_together = read_multiarg(
        "--move-together", argc, argv, NULL, "half", NULL);

    if (pencil->mat_a == NULL) {
        fprintf(stderr, "Missing matrix A.\n");
        return -1;
    }

#ifndef PDTRSEN_FOUND
    if (pencil->mat_b == NULL) {
        fprintf(stderr, "Solver does not support standard cases.\n");
        return -1;
    }
#endif

#ifndef PDTGSEN_FOUND
    if (pencil->mat_b != NULL) {
        fprintf(stderr, "Solver does not support generalized cases.\n");
        return -1;
    }
#endif

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

#ifdef PDTGSEN_FOUND
    starneig_blacs_descr_t desc_b, desc_z;
    double *local_b = NULL, *local_z = NULL;
    if (pencil->mat_b != NULL) {
        STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
            pencil->mat_b, context, &desc_b, (void **)&local_b);
        STARNEIG_BLACS_MATRIX_DESCR_LOCAL(
            pencil->mat_z, context, &desc_z, (void **)&local_z);
    }
#endif

    if (!has_valid_descr(n, sn, &desc_a)) {
        fprintf(stderr, "Matrix A has invalid dimensions.\n");
        return -1;
    }

#ifdef PDTGSEN_FOUND
    if (pencil->mat_b != NULL && !has_valid_descr(n, sn, &desc_b)) {
        fprintf(stderr, "Matrix B has invalid dimension.\n");
        return -1;
    }
#endif

    if (pencil->mat_q != NULL && !has_valid_descr(n, sn, &desc_q)) {
        fprintf(stderr, "Matrix Q has invalid dimension.\n");
        return -1;
    }

#ifdef PDTGSEN_FOUND
    if (pencil->mat_z != NULL && !has_valid_descr(n, sn, &desc_z)) {
        fprintf(stderr, "Matrix Z has invalid dimension.\n");
        return -1;
    }
#endif

    int srows, scols, my_row, my_col;
    starneig_blacs_gridinfo(context, &srows, &scols, &my_row, &my_col);

    int _parallel_windows;
    if (parallel_windows.type == MULTIARG_INT)
        _parallel_windows = MIN(MIN(srows, scols), parallel_windows.int_value);
    else
        _parallel_windows = MIN(srows, scols);

    int _window_size;
    if (window_size.type == MULTIARG_INT)
        _window_size = window_size.int_value;
    else
        _window_size = sn / 2;

    if (sn < _window_size) {
        printf("Invalid window size. Setting window size to %d.\n", sn);
        _window_size = sn;
    }

    int _values_per_window;
    if (values_per_window.type == MULTIARG_INT)
        _values_per_window = values_per_window.int_value;
    else
        _values_per_window = _window_size / 2;

    if (_window_size-2 < _values_per_window) {
        printf("Invalid number of values per window. Setting value to %d.\n",
            _window_size-2);
        _values_per_window = _window_size-2;
    }

    int _move_together;
    if (move_together.type == MULTIARG_INT)
        _move_together = move_together.int_value;
    else
        _move_together = _values_per_window;

    if (_values_per_window < _move_together) {
        printf("Invalid number of eigenvalue moved across node boundary."
            "Setting value to %d\n", _values_per_window);
        _move_together = _values_per_window;
    }

    int para[6] = {
        _parallel_windows,
        _values_per_window,
        _window_size,
        flops,
        width,
        _move_together
    };

    int const *selected = get_supplementaty_selected(pencil->supp);
    double *alphar, *alphai, *beta;
    get_supplementaty_eigenvalues(pencil->supp, &alphar, &alphai, &beta);
    if (alphar == NULL)
        init_supplementary_eigenvalues(
            n, &alphar, &alphai, &beta, &pencil->supp);

    int ia = 1, ja = 1, iq = 1, jq = 1;
#ifdef PDTGSEN_FOUND
    int ib = 1, jb = 1, iz = 1, jz = 1;
#endif

    double *work = NULL;

    // pdtrsen writes to iwork when it computes the workspace size!
    int *iwork = malloc(n*sizeof(int));

    int m, info = -1, lwork = -1, liwork = -1;
    double _work;

    threads_set_mode(THREADS_MODE_SCALAPACK);

    if (pencil->mat_b != NULL) {
#ifdef PDTGSEN_FOUND
        double pl, pr, dif[2];
        pdtgsen_((const int[]){0},
            pencil->mat_q ? (const int[]){1} : (const int[]){0},
            pencil->mat_z ? (const int[]){1} : (const int[]){0},
            selected, para, &n, local_a, &ia, &ja, &desc_a,
            local_b, &ib, &jb, &desc_b, local_q, &iq, &jq, &desc_q,
            local_z, &iz, &jz, &desc_z, alphar, alphai, beta,
            &m, &pl, &pr, dif, &_work, &lwork, iwork, &liwork, &info);
#endif
    }
    else {
#ifdef PDTRSEN_FOUND
        double s, sep;
        pdtrsen_("None", pencil->mat_q ? "V" : "N", selected, para, &n,
            local_a, &ia, &ja, &desc_a, local_q, &iq, &jq,
            &desc_q, alphar, alphai, &m, &s, &sep, &_work, &lwork, iwork,
            &liwork, &info);
#endif
    }

    threads_set_mode(THREADS_MODE_DEFAULT);

    if (info != 0)
        goto cleanup;

    lwork = _work;
    liwork = iwork[0];
    free(iwork);

    work = malloc(lwork*sizeof(double));
    iwork = malloc(liwork*sizeof(int));

    if (pencil->mat_b != NULL) {
#ifdef PDTGSEN_FOUND
        double pl, pr, dif[2];
        pdtgsen_((const int[]){0},
            pencil->mat_q ? (const int[]){1} : (const int[]){0},
            pencil->mat_z ? (const int[]){1} : (const int[]){0},
            selected, para, &n, local_a, &ia, &ja, &desc_a,
            local_b, &ib, &jb, &desc_b, local_q, &iq, &jq, &desc_q,
            local_z, &iz, &jz, &desc_z, alphar, alphai, beta, &m, &pl, &pr, dif,
            work, &lwork, iwork, &liwork, &info);
#endif
    }
    else {
#ifdef PDTRSEN_FOUND
        double s, sep;
        pdtrsen_("None", pencil->mat_q ? "V" : "N", selected, para, &n,
            local_a, &ia, &ja, &desc_a, local_q, &iq, &jq,
            &desc_q, alphar, alphai, &m, &s, &sep, work, &lwork, iwork, &liwork,
            &info);
#endif
    }

cleanup:

    starneig_blacs_gridexit(context);

    free(work);
    free(iwork);

    return info;
}

const struct hook_solver reorder_scalapack_solver = {
    .name = "scalapack",
    .desc = "pdtrsen/pdtgsen subroutine from ScaLAPACK",
    .formats = (hook_data_format_t[]) {
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &scalapack_print_usage,
    .print_args = &scalapack_print_args,
    .check_args = &scalapack_check_args,
    .prepare = &scalapack_prepare,
    .finalize = &scalapack_finalize,
    .run = &scalapack_run
};

#endif // PDTRSEN_FOUND

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct starpu_state {
    int argc;
    char * const *argv;
    struct hook_data_env *env;
};

///
/// @brief Reordering plan descriptor structure.
///
struct plan_descr {
    char const *name;
    char *desc;
    starneig_reorder_plan_t value;
};

///
/// @brief Reordering plans.
///
static const struct plan_descr plans[] = {
    { .name = "default",
        .desc = "Default reordering plan",
        .value = STARNEIG_REORDER_DEFAULT_PLAN },
    { .name = "spp",
        .desc = "One-part reordering plan",
        .value = STARNEIG_REORDER_ONE_PART_PLAN },
    { .name = "mpp",
        .desc = "Multi-part reordering plan",
        .value = STARNEIG_REORDER_MULTI_PART_PLAN }
};

static PRINT_AVAIL(print_avail_plans, "  Available reordering plans:",
    name, desc, plans, 0)

static READ_FROM_ARGV(read_plan, struct plan_descr const,
    name, plans, 0)

///
/// @brief Blueprint descriptor structure.
///
struct blueprint_descr {
    char const *name;           ///< name
    char *desc;                 ///< description
    starneig_reorder_blueprint_t value;  ///< corresponding enumerator value
};

///
/// @brief Blueprints.
///
static const struct blueprint_descr blueprints[] = {
    { .name = "default",
        .desc = "Default blueprint",
        .value = STARNEIG_REORDER_DEFAULT_BLUEPRINT },
    { .name = "dsa",
        .desc = "One-pass forward dummy blueprint",
        .value = STARNEIG_REORDER_DUMMY_INSERT_A },
    { .name = "dsb",
        .desc = "Two-pass backward dummy blueprint",
        .value = STARNEIG_REORDER_DUMMY_INSERT_B },
    { .name = "csa",
        .desc = "One-pass forward chain blueprint",
        .value = STARNEIG_REORDER_CHAIN_INSERT_A },
    { .name = "csb",
        .desc = "Two-pass forward chain blueprint",
        .value = STARNEIG_REORDER_CHAIN_INSERT_B },
    { .name = "csc",
        .desc = "One-pass backward chain blueprint",
        .value = STARNEIG_REORDER_CHAIN_INSERT_C },
    { .name = "csd",
        .desc = "Two-pass backward chain blueprint",
        .value = STARNEIG_REORDER_CHAIN_INSERT_D },
    { .name = "cse",
        .desc = "Two-pass delayed backward chain blueprint",
        .value = STARNEIG_REORDER_CHAIN_INSERT_E },
    { .name = "csf",
        .desc = "Three-pass delayed backward chain blueprint",
        .value = STARNEIG_REORDER_CHAIN_INSERT_F }
};

static PRINT_AVAIL(print_avail_blueprints, "  Available reordering blueprints:",
    name, desc, blueprints, 0)

static READ_FROM_ARGV(read_blueprint, struct blueprint_descr const,
    name, blueprints, 0)

static void starpu_print_usage(int argc, char * const *argv)
{
    printf(
        "  --cores [default,(num)} -- Number of CPU cores\n"
        "  --gpus [default,(num)} -- Number of GPUS\n"
        "  --tile-size [default,(num)} -- Block size\n"
        "  --window-size [default,rounded,(num)] -- Window size\n"
        "  --values-per-chain [default,(num)] -- Number of selected "
        "eigenvalues per window chain\n"
        "  --small-window-size [default,(num)] -- Small window size\n"
        "  --small-window-threshold [default,(num)] -- Small window "
        "threshold\n"
        "  --update-width [default,(num)]] -- Update tasks width\n"
        "  --update-height [default,(num)]] -- Update tasks height\n"
        "  --plan (plan) -- Eigenvalue reordering plan\n"
        "  --blueprint (blueprint) -- Eigenvalue reordering plan\n"
    );

    print_avail_plans();
    print_avail_blueprints();
}

static void starpu_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
    print_multiarg("--tile-size", argc, argv, "default", NULL);
    print_multiarg("--window-size", argc, argv, "default", "rounded", NULL);
    print_multiarg("--values-per-chain", argc, argv, "default", NULL);
    print_multiarg("--small-window-size", argc, argv, "default", NULL);
    print_multiarg("--small-window-threshold", argc, argv, "default", NULL);

    print_multiarg("--update-width", argc, argv, "default", NULL);
    print_multiarg("--update-height", argc, argv, "default", NULL);

    printf(" --plan %s --blueprint %s",
        read_plan("--plan", argc, argv, NULL)->name,
        read_blueprint("--blueprint",  argc, argv, NULL)->name);
}

static int starpu_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);
    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, argr, "default", NULL);
    struct multiarg_t window_size = read_multiarg(
        "--window-size", argc, argv, argr, "default", "rounded", NULL);
    struct multiarg_t values_per_chain = read_multiarg(
        "--values-per-chain", argc, argv, argr, "default", NULL);
    struct multiarg_t small_window_size = read_multiarg(
        "--small-window-size", argc, argv, argr, "default", NULL);
    struct multiarg_t small_window_threshold = read_multiarg(
        "--small-window-threshold", argc, argv, argr, "default", NULL);

    struct multiarg_t update_width = read_multiarg(
        "--update-width", argc, argv, argr, "default", NULL);
    struct multiarg_t update_height = read_multiarg(
        "--update-height", argc, argv, argr, "default", NULL);

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    // check plan and blueprint

    if (read_plan("--plan", argc, argv, argr) == NULL) {
        fprintf(stderr, "Invalid plan.\n");
        return -1;
    }

    if (read_blueprint("--blueprint", argc, argv, argr) == NULL) {
        fprintf(stderr, "Invalid blueprint.\n");
        return -1;
    }

    // check tile size

    if (tile_size.type == MULTIARG_INVALID || (tile_size.type == MULTIARG_INT &&
    tile_size.int_value < 2)) {
        fprintf(stderr, "Invalid tile size.\n");
        return -1;
    }

    // check window size

    if (window_size.type == MULTIARG_INVALID ||
    (window_size.type == MULTIARG_INT && window_size.int_value < 4)) {
        fprintf(stderr, "Invalid window size.\n");
        return -1;
    }

    // check values per chain parameter

    if (values_per_chain.type == MULTIARG_INVALID) {
        fprintf(stderr,
            "Invalid number of selected eigenvalues per window chain.\n");
        return -1;
    }

    if (values_per_chain.type == MULTIARG_INT) {
        if (values_per_chain.int_value < 2) {
            fprintf(stderr,
                "Invalid number of selected eigenvalues per window chain.\n");
            return -1;
        }

        if (window_size.type == MULTIARG_INT &&
        window_size.int_value < values_per_chain.int_value-1) {
            fprintf(stderr,
                "Invalid number of selected eigenvalues per window chain.\n");
            return -1;
        }
    }

    // check small window size

    if (small_window_size.type == MULTIARG_INVALID &&
    (small_window_size.type == MULTIARG_INT &&
    small_window_size.int_value < 4)) {
        fprintf(stderr, "Invalid small window size.\n");
        return -1;
    }

    // check small window threshold

    if (small_window_threshold.type == MULTIARG_INVALID) {
        fprintf(stderr, "Invalid small window threshold.\n");
        return -1;
    }

    if (update_width.type == MULTIARG_INVALID ||
        (update_width.type == MULTIARG_INT && update_width.int_value < 1)) {
        fprintf(stderr, "Invalid update task width.\n");
        return -1;
    }

    if (update_height.type == MULTIARG_INVALID ||
        (update_height.type == MULTIARG_INT && update_height.int_value < 1)) {
        fprintf(stderr, "Invalid update task height.\n");
        return -1;
    }

    return 0;
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
    pencil_t pencil = env->data;

    struct starneig_reorder_conf conf;
    starneig_reorder_init_conf(&conf);

    conf.plan = read_plan("--plan", argc, argv, NULL)->value;
    conf.blueprint = read_blueprint("--blueprint", argc, argv, NULL)->value;

    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, NULL, "default", NULL);
    if (tile_size.type == MULTIARG_INT)
        conf.tile_size = tile_size.int_value;

    struct multiarg_t window_size = read_multiarg(
        "--window-size", argc, argv, NULL, "default", "rounded", NULL);
    if (window_size.type == MULTIARG_STR &&
    !strcmp("rounded", window_size.str_value))
        conf.window_size = STARNEIG_REORDER_ROUNDED_WINDOW_SIZE;
    if (window_size.type == MULTIARG_INT)
        conf.window_size = window_size.int_value;

    struct multiarg_t values_per_chain = read_multiarg(
        "--values-per-chain", argc, argv, NULL, "default", NULL);
    if (values_per_chain.type == MULTIARG_INT)
        conf.values_per_chain = values_per_chain.int_value;

    struct multiarg_t small_window_size = read_multiarg(
        "--small-window-size", argc, argv, NULL, "default", NULL);
    if (small_window_size.type == MULTIARG_INT)
        conf.small_window_size = small_window_size.int_value;

    struct multiarg_t small_window_threshold = read_multiarg(
        "--small-window-threshold", argc, argv, NULL, "default", NULL);
    if (small_window_threshold.type == MULTIARG_INT)
        conf.small_window_threshold = small_window_threshold.int_value;

    struct multiarg_t update_width = read_multiarg(
        "--update-width", argc, argv, NULL, "default", NULL);
    if (update_width.type == MULTIARG_INT)
        conf.update_width = update_width.int_value;

    struct multiarg_t update_height = read_multiarg(
        "--update-height", argc, argv, NULL, "default", NULL);
    if (update_height.type == MULTIARG_INT)
        conf.update_height = update_height.int_value;

    int n = GENERIC_MATRIX_N(pencil->mat_a);
    int *selected = malloc(n*sizeof(int));
    memcpy(selected, get_supplementaty_selected(pencil->supp),
        n*sizeof(int));

    double *alphar, *alphai, *beta;
    get_supplementaty_eigenvalues(pencil->supp, &alphar, &alphai, &beta);
    if (alphar == NULL)
        init_supplementary_eigenvalues(
            n, &alphar, &alphai, &beta, &pencil->supp);

    starneig_error_t ret = STARNEIG_SUCCESS;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        if (pencil->mat_b != NULL)
            ret = starneig_GEP_SM_ReorderSchur_expert(&conf, n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                alphar, alphai, beta);
        else
            ret = starneig_SEP_SM_ReorderSchur_expert(&conf, n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                alphar, alphai);
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        if (pencil->mat_b != NULL)
            ret = starneig_GEP_DM_ReorderSchur_expert(&conf, selected,
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
                alphar, alphai, beta);
        else
            ret = starneig_SEP_DM_ReorderSchur_expert(&conf, selected,
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                alphar, alphai);
    }
#endif

    free(selected);

    return ret;
}

const struct hook_solver reorder_starpu_solver = {
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
    pencil_t pencil = env->data;

    int n = GENERIC_MATRIX_N(pencil->mat_a);
    int *selected = malloc(n*sizeof(int));
    memcpy(selected, get_supplementaty_selected(pencil->supp),
        n*sizeof(int));

    double *alphar, *alphai, *beta;
    get_supplementaty_eigenvalues(pencil->supp, &alphar, &alphai, &beta);
    if (alphar == NULL)
        init_supplementary_eigenvalues(
            n, &alphar, &alphai, &beta, &pencil->supp);

    starneig_error_t ret = STARNEIG_SUCCESS;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        if (pencil->mat_b != NULL)
            ret = starneig_GEP_SM_ReorderSchur(n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z),
                alphar, alphai, beta);
        else
            ret = starneig_SEP_SM_ReorderSchur(n, selected,
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                alphar, alphai);
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_STARNEIG ||
    env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        if (pencil->mat_b != NULL)
            ret = starneig_GEP_DM_ReorderSchur(selected,
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z),
                alphar, alphai, beta);
        else
            ret = starneig_SEP_DM_ReorderSchur(selected,
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                alphar, alphai);
    }
#endif

    free(selected);

    return ret;
}

const struct hook_solver reorder_starpu_simple_solver = {
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
