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
#include "../common/local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "../common/starneig_pencil.h"
#endif
#include "../common/threads.h"
#include <starneig/starneig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cblas.h>
#include <omp.h>

#ifdef MAGMA_FOUND
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <magma_auxiliary.h>
#include <magma_d.h>
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

    extern void dgehrd_(
        int const *,    // the order of the matrix A
        int const *,    // left bound
        int const *,    // right bound
        double *,       // input/output matrix
        int const *,    // input/output matrix leading dimension
        double *,       // tau (scalar factors)
        double *,       // work space
        int const *,    // work space size
        int *);         // info

    extern void dormhr_(
        char const *,   // side
        char const *,   // transpose
        int const *,    // row count
        int const *,    // column count
        int const *,    // left bound
        int const *,    // right bound
        double const *, // elementary reflectors
        int const *,    // elementary reflector leading dimension
        double const *, // scalar factors (tau)
        double *,       // input/output matrix
        int const *,    // input/output matrix leading dimension
        double *,       // work space
        int const *,    // work space size
        int *);         // info

    extern void dgeqrf_(int const *, int const *, double *, int const *,
        double *, double *, int const *, int *);

    extern void dormqr_(char const *, char const *, int const *, int const *,
        int const *, double const *, int const *, double const *, double *,
        int const *, double*, const int *, int *);

    extern void dgghd3_(char const *, char const *, int const *, int const *,
        int const *, double *, int const *, double *, int const *, double *,
        int const *, double *, int const *, double *, int const *, int *);

    int n = LOCAL_MATRIX_N(data->mat_a);
    double *A = LOCAL_MATRIX_PTR(data->mat_a);
    int ldA = LOCAL_MATRIX_LD(data->mat_a);
    double *Q = LOCAL_MATRIX_PTR(data->mat_q);
    int ldQ = LOCAL_MATRIX_LD(data->mat_q);

    double *B = NULL;
    int ldB = 0;
    double *Z = NULL;
    int ldZ = 0;

    double *tau = NULL;
    double *work = NULL;
    int info, ilo = 1, ihi = n;

    threads_set_mode(THREADS_MODE_LAPACK);

    if (data->mat_b != NULL) {
        B = LOCAL_MATRIX_PTR(data->mat_b);
        ldB = LOCAL_MATRIX_LD(data->mat_b);
        Z = LOCAL_MATRIX_PTR(data->mat_z);
        ldZ = LOCAL_MATRIX_LD(data->mat_z);

        //
        // allocate workspace
        //

        int lwork = 0;

        {
            int _lwork = -1;
            double dlwork;

            dgeqrf_(&n, &n, B, &ldB, tau, &dlwork, &_lwork, &info);
            if (info != 0)
                goto cleanup;

            lwork = MAX(lwork, dlwork);
        }

        {
            int _lwork = -1;
            double dlwork;

            dormqr_("L", "T", &n, &n, &n,
                B, &ldB, tau, A, &ldA, &dlwork, &_lwork, &info);
            if (info != 0)
                goto cleanup;

            lwork = MAX(lwork, dlwork);
        }

        {
            int _lwork = -1;
            double dlwork;

            dormqr_("R", "N", &n, &n, &n,
                B, &ldB, tau, Q, &ldQ, &dlwork, &_lwork, &info);
            if (info != 0)
                goto cleanup;

            lwork = MAX(lwork, dlwork);
        }

        {
            int _lwork = -1;
            double dlwork;

            dgghd3_("V", "V", &n, &ilo, &ihi,
                A, &ldA, B, &ldB, Q, &ldQ, Z, &ldZ, &dlwork, &_lwork, &info);
            if (info != 0)
                goto cleanup;

            lwork = MAX(lwork, dlwork);
        }

        tau = malloc(n*sizeof(double));
        work = malloc(lwork*sizeof(double));

        //
        // reduce
        //

        // form B = ~Q * R
        dgeqrf_(&n, &n, B, &ldB, tau, work, &lwork, &info);
        if (info != 0)
            goto cleanup;

        // A <- ~Q^T * A
        dormqr_("L", "T", &n, &n, &n, B, &ldB, tau, A, &ldA, work, &lwork,
            &info);
        if (info != 0)
            goto cleanup;

        // Q <- Q * ~Q
        dormqr_("R", "N", &n, &n, &n, B, &ldB, tau, Q, &ldQ, work, &lwork,
            &info);
        if (info != 0)
            goto cleanup;

        // clean B (B <- R)
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                B[(size_t)i*ldB+j] = 0.0;

        // reduce (A,B) to Hessenberg-triangular form
        dgghd3_("V", "V", &n, &ilo, &ihi,
            A, &ldA, B, &ldB, Q, &ldQ, Z, &ldZ, work, &lwork, &info);
        if (info != 0)
            goto cleanup;
    }
    else {

        int lwork = -1;
        double dlwork;

        // request optimal work space size
        dgehrd_(&n, &ilo, &ihi, A, &ldA,
            tau, &dlwork, &lwork, &info);

        if (info != 0)
            goto cleanup;

        lwork = dlwork;
        work = malloc(lwork*sizeof(double));
        tau = malloc(n*sizeof(double));

        // reduce
        dgehrd_(&n, &ilo, &ihi, A, &ldA,
            tau, work, &lwork, &info);

        if (info != 0)
            goto cleanup;

        free(work);
        work = NULL;

        // request optimal work space size
        lwork = -1;
        dormhr_("Right", "No transpose", &n, &n, &ilo, &ihi, A, &ldA, tau,
            Q, &ldQ, &dlwork, &lwork, &info);

        if (info != 0)
            goto cleanup;

        lwork = dlwork;
        work = malloc(lwork*sizeof(double));

        // form Q
        dormhr_("Right", "No transpose", &n, &n, &ilo, &ihi, A, &ldA, tau,
            Q, &ldQ, work, &lwork, &info);

        if (info != 0)
            goto cleanup;

        for (int i = 0; i < n; i++)
            for (int j = i+2; j < n; j++)
                A[i*ldA+j] = 0.0;
    }

cleanup:

    threads_set_mode(THREADS_MODE_DEFAULT);

    free(work);
    free(tau);
    return info;
}

const struct hook_solver hessenberg_lapack_solver = {
    .name = "lapack",
    .desc = "LAPACK's dgehrd/dgghrd subroutine",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .prepare = &lapack_prepare,
    .finalize = &lapack_finalize,
    .run = &lapack_run
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if defined(PDGEHRD_FOUND) && defined(PDORMHR_FOUND) && defined(PDLASET_FOUND)

static hook_solver_state_t scalapack_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    return env;
}

static int scalapack_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    return 0;
}

static int has_valid_descr(
    int matrix_size, int section_size, const starneig_blacs_descr_t *descr)
{
    if (descr->m != matrix_size || descr->n != matrix_size)
        return 0;
    if (descr->sm != section_size || descr->sn != section_size)
        return 0;
    return 1;
}

static int scalapack_run(hook_solver_state_t state)
{
    extern void pdgehrd_(int const *, int const *, int const *, double *,
        int const *, int const *, const starneig_blacs_descr_t *, double *,
        double *, int const *, int *);

    extern void pdormhr_(char const *, char const *, int const *, int const *,
        int const *, int const *, double *, int const *, int const *,
        const starneig_blacs_descr_t *, double *, double *, int const *,
        int const *, const starneig_blacs_descr_t *, double *, int const *,
        int *);

    extern void pdlaset_(char const *, int const *, int const *,
        double const *, double const *, double *, int const *, int const *,
        const starneig_blacs_descr_t *);

    threads_set_mode(THREADS_MODE_SCALAPACK);

    struct hook_data_env *env = state;
    pencil_t pencil = (pencil_t) env->data;

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

    if (!has_valid_descr(n, sn, &desc_a)) {
        fprintf(stderr, "Matrix A has invalid dimensions.\n");
        return -1;
    }

    if (pencil->mat_q != NULL && !has_valid_descr(n, sn, &desc_q)) {
        fprintf(stderr, "Matrix Q has invalid dimension.\n");
        return -1;
    }

    int ilo = 1, ihi = n, ia = 1, ja = 1, ic = 1, jc = 1, lwork, info;
    double *work = NULL, *tau = NULL, _work;

    lwork = -1;
    pdgehrd_(&n, &ilo, &ihi, NULL, &ia, &ja, &desc_a, NULL,
        &_work, &lwork, &info);

    if (info)
        goto cleanup;

    lwork = _work;
    work = malloc(lwork*sizeof(double));
    tau = malloc(n*sizeof(double));

    pdgehrd_(&n, &ilo, &ihi, local_a, &ia, &ja, &desc_a, tau,
        work, &lwork, &info);

    if (info)
        goto cleanup;

    lwork = -1;
    pdormhr_("Right", "No transpose", &n, &n, &ilo, &ihi, NULL,
        &ia, &ja, &desc_a, NULL, NULL, &ic, &jc,
        &desc_q, &_work, &lwork, &info);

    if (info)
        goto cleanup;

    free(work);
    lwork = _work;
    work = malloc(lwork*sizeof(double));

    pdormhr_("Right", "No transpose", &n, &n, &ilo, &ihi, local_a,
        &ia, &ja, &desc_a, tau, local_q, &ic, &jc,
        &desc_q, work, &lwork, &info);

    {
        int nm1 = n-2, one = 1, three = 3;
        double dzero = 0.0;
        pdlaset_("Lower", &nm1, &nm1, &dzero, &dzero, local_a, &three,
            &one, &desc_a);
    }

cleanup:

    threads_set_mode(THREADS_MODE_DEFAULT);
    starneig_blacs_gridexit(context);

    free(work);
    free(tau);

    return info;
}

const struct hook_solver hessenberg_scalapack_solver = {
    .name = "scalapack",
    .desc = "pdgehrd subroutine from scaLAPACK",
    .formats = (hook_data_format_t[]) {
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .prepare = &scalapack_prepare,
    .finalize = &scalapack_finalize,
    .run = &scalapack_run
};

#endif // PDGEHRD_FOUND

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
        "  --tile-size [default,(num)] -- tile size\n"
        "  --panel-width [default,(num)] -- Panel width\n"
        "  --parallel-worker-size [default,(num)]\n"
    );
}

static int starpu_check_args(int argc, char * const *argv, int *argr)
{
    struct multiarg_t arg_cores = read_multiarg(
        "--cores", argc, argv, argr, "default", NULL);
    struct multiarg_t arg_gpus = read_multiarg(
        "--gpus", argc, argv, argr, "default", NULL);
    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, argr, "default", NULL);
    struct multiarg_t panel_width = read_multiarg(
        "--panel-width", argc, argv, argr, "default", NULL);
    struct multiarg_t parallel_worker_size = read_multiarg(
        "--parallel-worker-size", argc, argv, argr, "default", NULL);

    if (arg_cores.type == MULTIARG_INVALID)
        return -1;

    if (arg_gpus.type == MULTIARG_INVALID)
        return -1;

    if (tile_size.type == MULTIARG_INVALID ||
    (tile_size.type == MULTIARG_INT && tile_size.int_value < 1)) {
        fprintf(stderr, "Invalid tile size.\n");
        return -1;
    }

    if (panel_width.type == MULTIARG_INVALID ||
    (panel_width.type == MULTIARG_INT && panel_width.int_value < 1)) {
        fprintf(stderr, "Invalid panel width.\n");
        return -1;
    }

    if (parallel_worker_size.type == MULTIARG_INVALID ||
    (parallel_worker_size.type == MULTIARG_INVALID &&
    parallel_worker_size.int_value < 1)) {
        fprintf(stderr, "Invalid parallel worker size.\n");
        return -1;
    }

    return 0;
}

static void starpu_print_args(int argc, char * const *argv)
{
    print_multiarg("--cores", argc, argv, "default", NULL);
    print_multiarg("--gpus", argc, argv, "default", NULL);
    print_multiarg("--tile-size", argc, argv, "default", NULL);
    print_multiarg("--panel-width", argc, argv, "default", NULL);
    print_multiarg(
        "--parallel-worker-size", argc, argv, "default", NULL);
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

    struct starneig_hessenberg_conf conf;
    starneig_hessenberg_init_conf(&conf);

    struct multiarg_t tile_size = read_multiarg(
        "--tile-size", argc, argv, NULL, "default", NULL);
    struct multiarg_t panel_width = read_multiarg(
        "--panel-width", argc, argv, NULL, "default", NULL);
    struct multiarg_t parallel_worker_size = read_multiarg(
        "--parallel-worker-size", argc, argv, NULL, "default", NULL);

    if (tile_size.type == MULTIARG_INT)
        conf.tile_size = tile_size.int_value;
    if (panel_width.type == MULTIARG_INT)
        conf.panel_width = panel_width.int_value;
    if (parallel_worker_size.type == MULTIARG_INT)
        conf.parallel_worker_size = parallel_worker_size.int_value;

    int ret = 0;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil_t pencil = (pencil_t) env->data;
        if (pencil->mat_b != NULL) {
            ret = starneig_GEP_SM_HessenbergTriangular(
                LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z)
            );
        }
        else {
            ret = starneig_SEP_SM_Hessenberg_expert(&conf,
                LOCAL_MATRIX_N(pencil->mat_a), 0, LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q)
            );
        }
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        pencil_t pencil = (pencil_t) env->data;
        if (pencil->mat_b != NULL) {
#ifdef STARNEIG_GEP_DM_HESSENBERGTRIANGULAR
            ret = starneig_GEP_DM_HessenbergTriangular(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z));
#else
            fprintf(stderr,
                "Solver does not support generalized cases in distributed "
                "memory.\n");
            return -1;
#endif
        }
        else {
#ifdef STARNEIG_SEP_DM_HESSENBERG
            ret = starneig_SEP_DM_Hessenberg(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q));
#else
            fprintf(stderr,
                "Solver does not support standard cases in distributed "
                "memory.\n");
            return -1;
#endif
        }
    }
#endif

    return ret;
}

const struct hook_solver hessenberg_starpu_solver = {
    .name = "starneig",
    .desc = "StarPU based subroutine",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
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

static int starpu_simple_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    if (state == NULL)
        return 0;

    starneig_node_finalize();

    free(state);
    return 0;
}

static int starpu_simple_run(hook_solver_state_t state)
{
    struct hook_data_env *env = ((struct starpu_state *) state)->env;

    int ret = 0;

    if (env->format == HOOK_DATA_FORMAT_PENCIL_LOCAL) {
        pencil_t pencil = (pencil_t) env->data;

        if (pencil->mat_b != NULL) {
            ret = starneig_GEP_SM_HessenbergTriangular(
                LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_b), LOCAL_MATRIX_LD(pencil->mat_b),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q),
                LOCAL_MATRIX_PTR(pencil->mat_z), LOCAL_MATRIX_LD(pencil->mat_z)
            );
        }
        else {
            ret = starneig_SEP_SM_Hessenberg(LOCAL_MATRIX_N(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_a), LOCAL_MATRIX_LD(pencil->mat_a),
                LOCAL_MATRIX_PTR(pencil->mat_q), LOCAL_MATRIX_LD(pencil->mat_q)
            );
        }
    }
#ifdef STARNEIG_ENABLE_MPI
    if (env->format == HOOK_DATA_FORMAT_PENCIL_BLACS) {
        pencil_t pencil = (pencil_t) env->data;

        if (pencil->mat_b != NULL) {
#ifdef STARNEIG_GEP_DM_HESSENBERGTRIANGULAR
            ret = starneig_GEP_DM_HessenbergTriangular(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_b),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q),
                STARNEIG_MATRIX_HANDLE(pencil->mat_z));
#else
            fprintf(stderr,
                "Solver does not support distributed memory in generalized "
                "cases.\n");
            return -1;
#endif
        }
        else {
#ifdef STARNEIG_SEP_DM_HESSENBERG
            ret = starneig_SEP_DM_Hessenberg(
                STARNEIG_MATRIX_HANDLE(pencil->mat_a),
                STARNEIG_MATRIX_HANDLE(pencil->mat_q));
#else
            fprintf(stderr,
                "Solver does not support distributed memory in standard "
                "cases.\n");
            return -1;
#endif
        }
    }
#endif

    return ret;
}

const struct hook_solver hessenberg_starpu_simple_solver = {
    .name = "starneig-simple",
    .desc = "StarPU based subroutine (simplified interface)",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef MAGMA_FOUND

static hook_solver_state_t magma_dgehrd_prepare(
    int argc, char * const *argv, struct hook_data_env *env)
{
    if (magma_init() != MAGMA_SUCCESS)
        return NULL;
    return (hook_solver_state_t) env->data;
}

static int magma_dgehrd_finalize(
    hook_solver_state_t state, struct hook_data_env *env)
{
    magma_finalize();
    return 0;
}

static int magma_dgehrd_run(hook_solver_state_t state)
{
    pencil_t data = (pencil_t) state;

    int n = LOCAL_MATRIX_N(data->mat_a);
    double *A = LOCAL_MATRIX_PTR(data->mat_a);
    double *Q = LOCAL_MATRIX_PTR(data->mat_q);
    int ldA = LOCAL_MATRIX_LD(data->mat_a);
    int ldQ = LOCAL_MATRIX_LD(data->mat_q);

    int info;

    double *tau = NULL;
    double *work = NULL;
    double *dT = NULL;

    threads_set_mode(THREADS_MODE_LAPACK);

    // request optimal work space size
    double _work;
    magma_dgehrd(n, 1, n, A, ldA, tau, &_work, -1, dT, &info);

    if (info != 0)
        goto finalize;

    tau = malloc(LOCAL_MATRIX_N(data->mat_a)*sizeof(double));

    int lwork = _work;
    work = malloc(lwork*sizeof(double));

    int nb = magma_get_dgehrd_nb(LOCAL_MATRIX_N(data->mat_a));
    cudaMalloc((void**)&dT, nb*n*sizeof(double));

    // reduce
    magma_dgehrd(n, 1, n, A, ldA, tau, work, lwork, dT, &info);

    if (info != 0)
        goto finalize;

    free(work);
    work = NULL;

    // copy A -> Q
    for (int i = 0; i < n; i++)
        memcpy(Q+i*ldQ, A+i*ldA, n*sizeof(double));

    // form Q
    magma_dorghr(n, 1, n, Q, ldQ, tau, dT, nb, &info);

    if (info != 0)
        goto finalize;

    // zero entries below the first sub-diagonal
    for (int i = 0; i < n-1; i++)
        memset(A+i*ldA+i+2, 0, (n-i-2)*sizeof(double));

finalize:

    threads_set_mode(THREADS_MODE_DEFAULT);

    cudaFree(dT);
    free(work);
    free(tau);
    return info;
}

const struct hook_solver hessenberg_magma_solver = {
    .name = "magma",
    .desc = "MAGMA's dgehrd subroutine",
    .formats = (hook_data_format_t[]) { HOOK_DATA_FORMAT_PENCIL_LOCAL, 0 },
    .prepare = &magma_dgehrd_prepare,
    .finalize = &magma_dgehrd_finalize,
    .run = &magma_dgehrd_run
};

#endif // MAGMA_FOUND
