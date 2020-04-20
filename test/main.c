///
/// @file
///
/// @brief This file contains the main function for the test program.
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
#include "common/common.h"
#include "common/parse.h"
#include "common/threads.h"
#include "hessenberg/experiment.h"
#include "schur/experiment.h"
#include "reorder/experiment.h"
#include "eigenvectors/experiment.h"
#include "misc/full_chain.h"
#include "misc/partial_hessenberg.h"
#include "misc/validator.h"

#include <starneig/starneig.h>
#ifdef STARNEIG_ENABLE_MPI
#include <starneig/distr_helpers.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#ifdef STARNEIG_ENABLE_MPI
#include <mpi.h>
#endif

///
/// @brief Experiment modules.
///
static const struct experiment_descr experiments[] = {
    { .name = "hessenberg",
        .desc = "Hessenberg reduction experiment",
        .print_usage = &hook_experiment_print_usage,
        .check_args = &hook_experiment_check_args,
        .print_args = &hook_experiment_print_args,
        .run = &hook_experiment_run,
        .info = &hessenberg_experiment
    },
    { .name = "schur",
        .desc = "Schur reduction experiment",
        .print_usage = &hook_experiment_print_usage,
        .check_args = &hook_experiment_check_args,
        .print_args = &hook_experiment_print_args,
        .run = &hook_experiment_run,
        .info = &schur_experiment
    },
    { .name = "reorder",
        .desc = "Eigenvalue reordering experiment",
        .print_usage = &hook_experiment_print_usage,
        .check_args = &hook_experiment_check_args,
        .print_args = &hook_experiment_print_args,
        .run = &hook_experiment_run,
        .info = &reorder_experiment
    },
    { .name = "eigenvectors",
        .desc = "Eigenvectors experiment",
        .print_usage = &hook_experiment_print_usage,
        .check_args = &hook_experiment_check_args,
        .print_args = &hook_experiment_print_args,
        .run = &hook_experiment_run,
        .info = &eigenvectors_experiment
    },
    { .name = "full-chain",
        .desc = "Full chain experiment",
        .print_usage = &hook_experiment_print_usage,
        .check_args = &hook_experiment_check_args,
        .print_args = &hook_experiment_print_args,
        .run = &hook_experiment_run,
        .info = &full_chain_experiment
    },
    { .name = "partial-hessenberg",
        .desc = "Partial Hessenberg reduction experiment",
        .print_usage = &partial_hessenberg_print_usage,
        .check_args = &partial_hessenberg_check_args,
        .print_args = &partial_hessenberg_print_args,
        .run = &partial_hessenberg_run
    },
    { .name = "validator",
        .desc = "Validation experiment",
        .print_usage = &hook_experiment_print_usage,
        .check_args = &hook_experiment_check_args,
        .print_args = &hook_experiment_print_args,
        .run = &hook_experiment_run,
        .info = &validator_experiment
    },
};

static PRINT_AVAIL(print_avail_experiments, "Available experiment modules:",
    name, desc, experiments, -1)

static READ_FROM_ARGV(read_experiment, struct experiment_descr const,
    name, experiments, -1)

///
/// @brief Prints instructions.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
static void print_usage(int argc, char * const *argv)
{
    printf(
        "Usage: %s (options)\n"
        "\n"
        "Global options:\n"
#ifdef STARNEIG_ENABLE_MPI
        "  --mpi -- Enable MPI\n"
        "  --mpi-mode [serialized,multiple] -- MPI mode\n"
#endif
#ifdef STARNEIG_ENABLE_CUDA
        "  --no-pinning -- Disable memory pinning\n"
#endif
        "  --seed (num) -- Random number generator seed\n"
        "  --experiment (experiment) -- Experiment module\n",
        argv[0]);

    thread_print_usage(argc, argv);

    print_avail_experiments();
    printf("\n");
}

///
/// @brief Main function.
///
/// @param[in] argc
///         The command line argument count.
///
/// @param[in] argv
///         The command line arguments.
///
/// @returns EXIT_SUCCESS if successful, EXIT_FAILURE otherwise.
///
int main(int argc, char * const *argv)
{
    if (argc < 2) {
        print_usage(argc, argv);
        return EXIT_SUCCESS;
    }

    int ret = EXIT_SUCCESS;

#ifdef STARNEIG_ENABLE_MPI
    int mpi_initialized = 0;
#endif

    //
    // an array for tracking which command line arguments have been processed
    //

    int argr[argc];
    for (int i = 1; i < argc; i++)
        argr[i] = 0;

    //
    // load experiment module
    //

    if (!read_opt("--experiment", argc, argv, argr)) {
        fprintf(stderr, "Missing experiment module.\n");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    struct experiment_descr const *experiment =
        read_experiment("--experiment", argc, argv, argr);

    if (experiment == NULL || experiment->run == NULL) {
        fprintf(stderr, "Invalid experiment module.\n");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    //
    // print experiment module usage information if no additional arguments
    // were given
    //

    if (argc == 3) {
        print_usage(argc, argv);

        printf("Experiment module (%s) specific options:\n", experiment->name);
        if (experiment->print_usage != NULL)
            experiment->print_usage(argc, argv, experiment->info);
        goto cleanup;
    }

#ifdef STARNEIG_ENABLE_MPI

    //
    // initialize MPI driver
    //

    int mpi = read_opt("--mpi", argc, argv, argr);
    MPI_Comm comm = MPI_COMM_NULL;

    if (mpi) {
        struct multiarg_t mpi_mode = read_multiarg(
            "--mpi-mode", argc, argv, argr, "serialized", "multiple", NULL);

        if (mpi_mode.type != MULTIARG_STR) {
            fprintf(stderr, "Invalid MPI mode.\n");
            ret = EXIT_FAILURE;
            goto cleanup;
        }

        printf("MPI INIT...\n");
        int thread_support;

        if (strcmp(mpi_mode.str_value, "serialized") == 0) {
            MPI_Init_thread(
                &argc, (char ***)&argv, MPI_THREAD_SERIALIZED, &thread_support);
        }
        if (strcmp(mpi_mode.str_value, "multiple") == 0) {
            MPI_Init_thread(
                &argc, (char ***)&argv, MPI_THREAD_MULTIPLE, &thread_support);
        }

        mpi_initialized = 1;

        if (thread_support < MPI_THREAD_SERIALIZED) {
            fprintf(stderr,
                "MPI_THREAD_SERIALIZED is not supported. Exiting...\n");
            ret = EXIT_FAILURE;
            goto cleanup;
        }

        if (strcmp(mpi_mode.str_value, "multiple") == 0 &&
        thread_support < MPI_THREAD_MULTIPLE)
            fprintf(stderr, "Warning: MPI_THREAD_MULTIPLE is not supported.\n");

        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        starneig_mpi_set_comm(comm);
    }
#endif

#ifdef STARNEIG_ENABLE_CUDA

    int disable_pinning = read_opt("--no-pinning", argc, argv, argr);
    if (disable_pinning) {
        set_pinning(0);
        starneig_node_disable_pinning();
    }
    else {
        set_pinning(1);
        starneig_node_enable_pinning();
    }

#endif

    //
    // read/generate random seed
    //

    unsigned seed = read_uint("--seed", argc, argv, argr, (unsigned)time(NULL));

#ifdef STARNEIG_ENABLE_MPI
    if (mpi_initialized)
        MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif

    init_prand(seed);

    //
    // check thread count arguments
    //

    ret = thread_check_args(argc, argv, argr);
    if (ret) {
        fprintf(stderr, "Invalid arguments.\n");
        goto cleanup;
    }

    //
    // check experiment module command line arguments
    //

    if (experiment->check_args != NULL) {
        ret = experiment->check_args(argc, argv, argr, experiment->info);
        if (ret) {
            fprintf(stderr, "Invalid arguments.\n");
            goto cleanup;
        }
    }

    //
    // make sure that all command line arguments were processed
    //

    int valid_args = 1;
    for (int i = 1; i < argc; i++) {
        if (!argr[i]) {
            fprintf(stderr, "Invalid argument %s.\n", argv[i]);
            valid_args = 0;
        }
    }

    if (!valid_args) {
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    //
    // print command line arguments
    //

    printf("TEST:");
#ifdef STARNEIG_ENABLE_MPI
    if (mpi) {
        printf(" --mpi");
        print_multiarg(
            "--mpi-mode", argc, argv, "serialized", "multiple", NULL);
    }
#endif
#ifdef STARNEIG_ENABLE_CUDA
    if (disable_pinning)
        printf(" --disable-pinning");
#endif
    printf(" --seed %d --experiment %s", seed, experiment->name);

    thread_print_args(argc, argv);

    if (experiment->print_args != NULL)
        experiment->print_args(argc, argv, experiment->info);

    printf("\n");

    //
    // run the experiment
    //

    threads_init(argc, argv);

    ret = experiment->run(argc, argv, experiment->info);

cleanup:

    //
    // cleanup
    //

#ifdef STARNEIG_ENABLE_MPI
    if (mpi_initialized) {
        MPI_Comm_free(&comm);
        MPI_Finalize();
    }
#endif

    return ret;
}
