///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
/// @section LICENSE
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
#include "cpu.h"
#include "../common/common.h"
#include "../common/sanity.h"
#include "../common/tiles.h"
#include <starpu.h>
#include <starpu_scheduler.h>
#include <hwloc.h>
#include <cblas.h>
#include <omp.h>

void starneig_hessenberg_ext_cpu_process_panel_single(
    void *buffers[], void *cl_args)
{
    // LAPACK subroutine that generates a real elementary reflector H
    extern void dlarfg_(int const *, double *, double *, int const *, double *);

    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y2 = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY2 = STARPU_MATRIX_GET_LD(buffers[2]);

    double *P = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldP = STARPU_MATRIX_GET_LD(buffers[3]);

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + 4;

    starneig_join_sub_window(0, m, 0, nb, &packing_info, ldP, A_i, P, 0);

    int ldA = divceil(m, 8)*8;
    double *A = malloc(n*ldA*sizeof(double));
    starneig_join_sub_window(0, m, 0, n, &packing_info, ldA, A_i, A, 0);

    //
    // loop over column in the panel
    //

    for (int i = 0; i < nb; i++) {

        double *v = V+i*ldV+i;
        double *p = P+i*ldP;

        // update the current column if necessary
        if (0 < i) {

            // A <- A - Y2 * V' (update column from the right)
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, i,
                -1.0, Y2, ldY2, V+i-1, ldV, 1.0, p, 1);

            //
            // update column from the left
            //

            // we use the last column of T as a work space
            double *w = T+(nb-1)*ldT;

            // w <- V1' * b1 (upper part of V and column)
            cblas_dcopy(i, p, 1, w, 1);
            cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasUnit,
                i, V, ldV, w, 1);

            // w <- w + V2' * b2 (lower part of V and column)
            cblas_dgemv(CblasColMajor, CblasTrans, m-i, i,
                1.0, V+i, ldV, p+i, 1, 1.0, w, 1);

            // w <- T' * w
            cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                i, T, ldT, w, 1);

            // b2 <- b2 - V2 * w
            cblas_dgemv(CblasColMajor, CblasNoTrans, m-i, i,
                -1.0, V+i, ldV, w, 1, 1.0, p+i, 1);

            // b1 <- b1 - V1 * w
            cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                i, V, ldV, w, 1);
            cblas_daxpy(i, -1.0, w, 1, p, 1);
        }

        //
        // form the reflector and zero the sub-diagonal elements
        //

        int height = m-i;
        memcpy(v, p+i, height*sizeof(double));

        double tau;
        dlarfg_(&height, p+i, v+1, (const int[]){1}, &tau);
        v[0] = 1.0;

        for (int j = i+1; j < m; j++)
            p[j] = 0.0;

        //
        // update Y2
        //

        // Y2(:,i) <- trailing matrix times v
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, n-i-1,
            1.0, A+(i+1)*ldA, ldA, v, 1, 0.0, Y2+i*ldY2, 1);

        // w <- V' * v (shared result)
        cblas_dgemv(CblasColMajor, CblasTrans, m-i, i,
            1.0, V+i, ldV, v, 1, 0.0, T+i*ldT, 1);

        // Y2(:,i) <- Y2(:,i) - Y * w
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, i,
            -1.0, Y2, ldY2, T+i*ldT, 1, 1.0, Y2+i*ldY2, 1);

        cblas_dscal(m, tau, Y2+i*ldY2, 1);

        //
        // update T
        //

        // w <- tau * w
        cblas_dscal(i, -tau, T+i*ldT, 1);

        // T(0:i,i) = T * w
        cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans,
            CblasNonUnit, i, T, ldT, T+i*ldT, 1);

        T[i*ldT+i] = tau;
    }

    STARNEIG_SANITY_CHECK_HESSENBERG(0, nb, m, ldP, 0, P, NULL);

    // copy panel back
    starneig_join_sub_window(0, m, 0, nb, &packing_info, ldP, A_i, P, 1);

    free(A);
}

void starneig_hessenberg_ext_cpu_process_panel_bind(void *buffers[], void *cl_args)
{
    // LAPACK subroutine that generates a real elementary reflector H
    extern void dlarfg_(int const *, double *, double *, int const *, double *);

    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y2 = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY2 = STARPU_MATRIX_GET_LD(buffers[2]);

    double *P = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldP = STARPU_MATRIX_GET_LD(buffers[3]);

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + 4;

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    double tau = 0.0;

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
    // get the scheduling context
    unsigned ctx = starpu_task_get_current()->sched_ctx;

    // calculate per thread trailing matrix segment height
    int pp = divceil(m, starpu_sched_ctx_get_nworkers(ctx));

    int *workers;
    starpu_sched_ctx_get_workers_list(ctx, &workers);

    #pragma omp parallel num_threads(starpu_sched_ctx_get_nworkers(ctx))
#else
    // calculate per thread trailing matrix segment height
    int pp = divceil(m, starpu_combined_worker_get_size());

    hwloc_cpuset_t master = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology, master, HWLOC_CPUBIND_THREAD);

    #pragma omp parallel num_threads(starpu_combined_worker_get_size())
#endif
    {
        int tid = omp_get_thread_num();

        // join panel tiles
        if (tid*pp < m)
            starneig_join_sub_window(tid*pp, MIN(m, (tid+1)*pp), 0, nb,
                &packing_info, ldP, A_i, P+tid*pp, 0);
        #pragma omp barrier

        //
        // bind OpenMP threads, join panel tiles and allocate local workspace
        //

        // store the old binding (this might not be necessary)
        hwloc_cpuset_t old = hwloc_bitmap_alloc();
        hwloc_get_cpubind(topology, old, HWLOC_CPUBIND_THREAD);

        // find a suitable CPU core for a binding mask

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
        hwloc_cpuset_t set = starpu_worker_get_hwloc_cpuset(workers[tid]);
#else
        hwloc_cpuset_t set = hwloc_bitmap_alloc();
        hwloc_bitmap_zero(set);

        int core = hwloc_bitmap_first(master);
        for (int i = 0; i < tid; i++)
            core = hwloc_bitmap_next(master, core);

        hwloc_bitmap_set(set, core);
#endif

        // bind the thread
        hwloc_set_cpubind(topology, set, HWLOC_CPUBIND_THREAD);

        double *A = NULL;
        int ldA = 0;

        // allocate workspace and copy the matching section of the trailing
        // matrix
        if (tid*pp < m) {
            ldA = divceil(MIN(pp, m-tid*pp), 8)*8;
            A = hwloc_alloc_membind(topology, n*ldA*sizeof(double), set,
                HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
            starneig_join_sub_window(tid*pp, MIN(m, (tid+1)*pp), 0, n,
                &packing_info, ldA, A_i, A, 0);
        }

        //
        // loop over column in the panel
        //

        for (int i = 0; i < nb; i++) {

            double *v = V+i*ldV+i;
            double *p = P+i*ldP;

            // update the current column if necessary
            #pragma omp single
            if (0 < i) {

                // A <- A - Y2 * V' (update column from the right)
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, i,
                    -1.0, Y2, ldY2, V+i-1, ldV, 1.0, p, 1);

                //
                // update column from the left
                //

                // we use the last column of T as a work space
                double *w = T+(nb-1)*ldT;

                // w <- V1' * b1 (upper part of V and column)
                cblas_dcopy(i, p, 1, w, 1);
                cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasUnit,
                    i, V, ldV, w, 1);

                // w <- w + V2' * b2 (lower part of V and column)
                cblas_dgemv(CblasColMajor, CblasTrans, m-i, i,
                    1.0, V+i, ldV, p+i, 1, 1.0, w, 1);

                // w <- T' * w
                cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                    i, T, ldT, w, 1);

                // b2 <- b2 - V2 * w
                cblas_dgemv(CblasColMajor, CblasNoTrans, m-i, i,
                    -1.0, V+i, ldV, w, 1, 1.0, p+i, 1);

                // b1 <- b1 - V1 * w
                cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                    i, V, ldV, w, 1);
                cblas_daxpy(i, -1.0, w, 1, p, 1);
            }

            #pragma omp single
            {
                //
                // form the reflector and zero the sub-diagonal elements
                //

                int height = m-i;
                memcpy(v, p+i, height*sizeof(double));
                dlarfg_(&height, p+i, v+1, (const int[]){1}, &tau);
                v[0] = 1.0;

                for (int j = i+1; j < m; j++)
                    p[j] = 0.0;
            }

            //
            // update Y2
            //

            // Y2(:,i) <- trailing matrix times v
            if (tid*pp < m) {
                cblas_dgemv(CblasColMajor, CblasNoTrans, MIN(pp, m-tid*pp),
                    n-i-1, 1.0, A+(i+1)*ldA, ldA, v, 1, 0.0,
                    Y2+i*ldY2+tid*pp, 1);
            }
            #pragma omp barrier

            #pragma omp single
            {
                // w <- V' * v (shared result)
                cblas_dgemv(CblasColMajor, CblasTrans, m-i, i,
                    1.0, V+i, ldV, v, 1, 0.0, T+i*ldT, 1);

                // Y2(:,i) <- Y2(:,i) - Y * w
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, i,
                    -1.0, Y2, ldY2, T+i*ldT, 1, 1.0, Y2+i*ldY2, 1);

                cblas_dscal(m, tau, Y2+i*ldY2, 1);

                //
                // update T
                //

                // w <- tau * w
                cblas_dscal(i, -tau, T+i*ldT, 1);

                // T(0:i,i) = T * w
                cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans,
                    CblasNonUnit, i, T, ldT, T+i*ldT, 1);

                T[i*ldT+i] = tau;
            }
        }

        // copy panel back
        if (tid*pp < m)
            starneig_join_sub_window(tid*pp, MIN(m, (tid+1)*pp), 0, nb,
                &packing_info, ldP, A_i, P+tid*pp, 1);

        hwloc_set_cpubind(topology, old, HWLOC_CPUBIND_THREAD);
        hwloc_bitmap_free(old);
        hwloc_bitmap_free(set);
        hwloc_free(topology, A, n*ldA*sizeof(double));
    }

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
    free(workers);
#else
    hwloc_bitmap_free(master);
#endif

    hwloc_topology_destroy(topology);

    STARNEIG_SANITY_CHECK_HESSENBERG(0, nb, m, ldP, 0, P, NULL);
}

void starneig_hessenberg_ext_cpu_update_trail_single(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y2 = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY2 = STARPU_MATRIX_GET_LD(buffers[2]);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int nA = STARPU_MATRIX_GET_NY(buffers[3]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[3]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[4]);
    int mW = STARPU_MATRIX_GET_NX(buffers[4]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[4]);

    int max_width = MIN(nA, mW);

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + 5;

    for (int i = 0; i < n; i += max_width) {

        //
        // join tiles and update from the right
        //

        starneig_join_sub_window(
            0, m, i, MIN(n, i+max_width), &packing_info, ldA, A_i, A, 0);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            m, MIN(max_width, n-i), nb, -1.0,
            Y2, ldY2, V+i+nb-1, ldV, 1.0, A, ldA);

        //
        // update from the left
        //

        int width = MIN(max_width, n-i);
        if (0 < width) {
            for (int k = 0; k < nb; k++)
                cblas_dcopy(width, A+k, ldA, W+k*ldW, 1);

            cblas_dtrmm(
                CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                CblasUnit, width, nb, 1.0, V, ldV, W, ldW);

            if (nb < m)
                cblas_dgemm(
                    CblasColMajor, CblasTrans, CblasNoTrans, width, nb,
                    m-nb, 1.0, A+nb, ldA, V+nb, ldV, 1.0, W, ldW);

            cblas_dtrmm(
                CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, width, nb, 1.0, T, ldT, W, ldW);

            if (nb < m)
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasTrans, m-nb, width,
                    nb, -1.0, V+nb, ldV, W, ldW, 1.0, A+nb, ldA);

            cblas_dtrmm(
                CblasColMajor, CblasRight, CblasLower, CblasTrans,
                CblasUnit, width, nb, 1.0, V, ldV, W, ldW);

            for (int k = 0; k < nb; k++)
                cblas_daxpy(width, -1.0, W+k*ldW, 1, A+k, ldA);
        }

        //
        // copy tiles back
        //

        starneig_join_sub_window(
            0, m, i, MIN(n, i+max_width), &packing_info, ldA, A_i, A, 1);
    }
}

void starneig_hessenberg_ext_cpu_update_trail_bind(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *Y2 = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldY2 = STARPU_MATRIX_GET_LD(buffers[2]);

    double *A = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int nA = STARPU_MATRIX_GET_NY(buffers[3]);
    int ldA = STARPU_MATRIX_GET_LD(buffers[3]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[4]);
    int mW = STARPU_MATRIX_GET_NX(buffers[4]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[4]);

    int max_width = MIN(nA, mW);

    struct starpu_matrix_interface **A_i =
        (struct starpu_matrix_interface **)buffers + 5;

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
    // get the scheduling context
    unsigned ctx = starpu_task_get_current()->sched_ctx;

    int ppm = divceil(m, starpu_sched_ctx_get_nworkers(ctx));
    int ppn = divceil(max_width, starpu_sched_ctx_get_nworkers(ctx));

    int *workers;
    starpu_sched_ctx_get_workers_list(ctx, &workers);

    #pragma omp parallel num_threads(starpu_sched_ctx_get_nworkers(ctx))
#else
    int ppm = divceil(m, starpu_combined_worker_get_size());
    int ppn = divceil(max_width, starpu_combined_worker_get_size());

    hwloc_cpuset_t master = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology, master, HWLOC_CPUBIND_THREAD);

    #pragma omp parallel num_threads(starpu_combined_worker_get_size())
#endif
    {
        int tid = omp_get_thread_num();

        // store the old binding (this might not be necessary)
        hwloc_cpuset_t old = hwloc_bitmap_alloc();
        hwloc_get_cpubind(topology, old, HWLOC_CPUBIND_THREAD);

        // find a suitable CPU core for a binding mask

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
        hwloc_cpuset_t set = starpu_worker_get_hwloc_cpuset(workers[tid]);
#else
        hwloc_cpuset_t set = hwloc_bitmap_alloc();
        hwloc_bitmap_zero(set);

        int core = hwloc_bitmap_first(master);
        for (int i = 0; i < tid; i++)
            core = hwloc_bitmap_next(master, core);

        hwloc_bitmap_set(set, core);
#endif

        // bind the thread
        hwloc_set_cpubind(topology, set, HWLOC_CPUBIND_THREAD);

        double *_A = A + tid * ppn * ldA;
        double *_W = W + tid * ppn;

        for (int i = 0; i < n; i += max_width) {

            //
            // join tiles and update from the right
            //

            int height = MIN(ppm, m-tid*ppm);

            if (0 < height) {
                starneig_join_sub_window(
                    tid*ppm, tid*ppm+height, i, MIN(n, i+max_width),
                    &packing_info, ldA, A_i, A+tid*ppm, 0);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    height, MIN(max_width, n-i), nb, -1.0,
                    Y2+tid*ppm, ldY2, V+i+nb-1, ldV, 1.0, A+tid*ppm, ldA);
            }

            #pragma omp barrier

            //
            // update from the left
            //

            int width = MIN(MIN(ppn, max_width-tid*ppn), n-i-tid*ppn);
            if (0 < width) {
                for (int k = 0; k < nb; k++)
                    cblas_dcopy(width, _A+k, ldA, _W+k*ldW, 1);

                cblas_dtrmm(
                    CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasUnit, width, nb, 1.0, V, ldV, _W, ldW);

                if (nb < m)
                    cblas_dgemm(
                        CblasColMajor, CblasTrans, CblasNoTrans, width, nb,
                        m-nb, 1.0, _A+nb, ldA, V+nb, ldV, 1.0, _W, ldW);

                cblas_dtrmm(
                    CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, width, nb, 1.0, T, ldT, _W, ldW);

                if (nb < m)
                    cblas_dgemm(
                        CblasColMajor, CblasNoTrans, CblasTrans, m-nb, width,
                        nb, -1.0, V+nb, ldV, _W, ldW, 1.0, _A+nb, ldA);

                cblas_dtrmm(
                    CblasColMajor, CblasRight, CblasLower, CblasTrans,
                    CblasUnit, width, nb, 1.0, V, ldV, _W, ldW);

                for (int k = 0; k < nb; k++)
                    cblas_daxpy(width, -1.0, _W+k*ldW, 1, _A+k, ldA);
            }

            #pragma omp barrier

            //
            // copy tiles back
            //

            if (0 < height) {
                starneig_join_sub_window(
                    tid*ppm, tid*ppm+height, i, MIN(n, i+max_width),
                    &packing_info, ldA, A_i, A+tid*ppm, 1);
            }

            #pragma omp barrier
        }

        hwloc_set_cpubind(topology, old, HWLOC_CPUBIND_THREAD);
        hwloc_bitmap_free(old);
        hwloc_bitmap_free(set);
    }

#if 1 < STARPU_MAJOR_VERSION || 2 < STARPU_MINOR_VERSION
    free(workers);
#else
    hwloc_bitmap_free(master);
#endif

    hwloc_topology_destroy(topology);
}

void starneig_hessenberg_ext_cpu_update_right(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[3]);

    struct starpu_matrix_interface **X_i =
        (struct starpu_matrix_interface **)buffers + 4;

    starneig_join_window(&packing_info, ldX, X_i, X, 0);

    for (int j = 0; j < nb; j++)
        cblas_dcopy(m, X+j*ldX, 1, W+j*ldW, 1);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
        m, nb, 1.0, V, ldV, W, ldW);

    if (nb < n)
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, m, nb, n-nb,
            1.0, X+nb*ldX, ldX, V+nb, ldV, 1.0, W, ldW);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, nb, 1.0, T, ldT, W, ldW);

    if (nb < n)
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasTrans, m, n-nb, nb,
            -1.0, W, ldW, V+nb, ldV, 1.0, X+nb*ldX, ldX);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
        m, nb, 1.0, V, ldV, W, ldW);

    for (int j = 0; j < nb; j++)
        cblas_daxpy(m, -1.0, W+j*ldW, 1, X+j*ldX, 1);

    starneig_join_window(&packing_info, ldX, X_i, X, 1);
}

void starneig_hessenberg_ext_cpu_update_left(void *buffers[], void *cl_args)
{
    struct packing_info packing_info;
    int nb;
    starpu_codelet_unpack_args(cl_args, &packing_info, &nb);

    int m = packing_info.rend - packing_info.rbegin;
    int n = packing_info.cend - packing_info.cbegin;

    double *V = (double *) STARPU_MATRIX_GET_PTR(buffers[0]);
    int ldV = STARPU_MATRIX_GET_LD(buffers[0]);

    double *T = (double *) STARPU_MATRIX_GET_PTR(buffers[1]);
    int ldT = STARPU_MATRIX_GET_LD(buffers[1]);

    double *X = (double *) STARPU_MATRIX_GET_PTR(buffers[2]);
    int ldX = STARPU_MATRIX_GET_LD(buffers[2]);

    double *W = (double *) STARPU_MATRIX_GET_PTR(buffers[3]);
    int ldW = STARPU_MATRIX_GET_LD(buffers[3]);

    struct starpu_matrix_interface **X_i =
        (struct starpu_matrix_interface **)buffers + 4;

    starneig_join_window(&packing_info, ldX, X_i, X, 0);

    for (int j = 0; j < nb; j++)
        cblas_dcopy(n, X+j, ldX, W+j*ldW, 1);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
        n, nb, 1.0, V, ldV, W, ldW);

    if (nb < m)
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans, n, nb, m-nb,
            1.0, X+nb, ldX, V+nb, ldV, 1.0, W, ldW);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
        n, nb, 1.0, T, ldT, W, ldW);

    if (nb < m)
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasTrans, m-nb, n, nb,
            -1.0, V+nb, ldV, W, ldW, 1.0, X+nb, ldX);

    cblas_dtrmm(
        CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
        n, nb, 1.0, V, ldV, W, ldW);

    for (int j = 0; j < nb; j++)
        cblas_daxpy(n, -1.0, W+j*ldW, 1, X+j, ldX);

    starneig_join_window(&packing_info, ldX, X_i, X, 1);
}
