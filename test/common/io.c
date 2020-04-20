///
/// @file This file contains the input and output functionality of the test
/// program.
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
#include "io.h"
#include "common.h"
#include "parse.h"
#include "init.h"
#include "crawler.h"
#include "supplementary.h"
#include "select_distr.h"
#include "local_pencil.h"
#ifdef STARNEIG_ENABLE_MPI
#include "starneig_pencil.h"
#endif
#include "hook_experiment.h"
#include "../3rdparty/matrixmarket/mmio.h"
#include <unistd.h>

void read_mtx_dimensions_from_file(char const *name, int *m, int *n)
{
    FILE *file = fopen(name, "r");
    if (file == NULL) {
        fprintf(stderr, "Invalid filename.\n");
        abort();
    }

    MM_typecode matcode;
    if (mm_read_banner(file, &matcode) != 0) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    if (!mm_is_matrix(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Invalid matrix type.\n");
        abort();
    }

    int nz;
    if (mm_read_mtx_crd_size(file, m, n, &nz) != 0) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    fclose(file);
}

matrix_t read_mtx_matrix_from_file(char const *name, init_helper_t helper)
{
    FILE *file = fopen(name, "r");
    if (file == NULL) {
        fprintf(stderr, "Invalid filename.\n");
        abort();
    }

    MM_typecode matcode;
    if (mm_read_banner(file, &matcode) != 0) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    if (!mm_is_matrix(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Invalid matrix type.\n");
        abort();
    }

    int m, n, nz;
    if (mm_read_mtx_crd_size(file, &m, &n, &nz) != 0) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    if (m != n) {
        fprintf(stderr, "Invalid matrix dimension.\n");
        abort();
    }

    fclose(file);

    return read_mtx_sub_matrix_from_file(0, m, name, helper);
}

matrix_t read_mtx_sub_matrix_from_file(
    int begin, int end, char const *name, init_helper_t helper)
{
    FILE *file = fopen(name, "r");
    if (file == NULL) {
        fprintf(stderr, "Invalid filename.\n");
        abort();
    }

    MM_typecode matcode;
    if (mm_read_banner(file, &matcode) != 0) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    if (!mm_is_matrix(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Invalid matrix type.\n");
        abort();
    }

    int m, n, nz;
    if (mm_read_mtx_crd_size(file, &m, &n, &nz) != 0) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    if (m != n) {
        fprintf(stderr, "Invalid matrix dimension.\n");
        abort();
    }

    if (begin == 0 && end == m)
        printf("READING A %d X %d MATRIX ...\n", m, n);
    else
        printf("READING A %d X %d SUBMATRIX from A %d X %d MATRIX...\n",
            end-begin, end-begin, m, n);

    matrix_t matrix = generate_zero(end-begin, end-begin, helper);

    if (matrix->type == LOCAL_MATRIX) {
        double *A = LOCAL_MATRIX_PTR(matrix);
        size_t ldA = LOCAL_MATRIX_LD(matrix);

        for (int k = 0; k < nz; k++) {
            int i, j;
            double val;
            if (fscanf(file, "%d %d %lg\n", &i, &j, &val) == EOF) {
                fprintf(stderr, "Invalid file.\n");
                abort();
            }

            if (begin <= i-1 && i-1 < end && begin <= j-1 && j-1 < end)
                A[(j-1-begin)*ldA+i-1-begin] = val;
        }
    }

#ifdef STARNEIG_ENABLE_MPI
    if (matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX) {
        struct starneig_distr_block *blocks;
        int num_blocks;
        starneig_distr_matrix_get_blocks(
            STARNEIG_MATRIX_HANDLE(matrix), &blocks, &num_blocks);

        for (int k = 0; k < nz; k++) {
            int i, j;
            double val;
            if (fscanf(file, "%d %d %lg\n", &i, &j, &val) == EOF) {
                fprintf(stderr, "Invalid file.\n");
                abort();
            }

            for (int l = 0; l < num_blocks; l++) {
                if (
                    blocks[l].glo_row <= i-1-begin &&
                        i-1-begin < blocks[l].glo_row+blocks[l].row_blksz &&
                    blocks[l].glo_col <= j-1-begin &&
                        j-1-begin < blocks[l].glo_col+blocks[l].col_blksz
                ) {
                    double *A = blocks[l].ptr;
                    size_t ldA = blocks[l].ld;
                    int _i = i-1-begin-blocks[l].glo_row;
                    int _j = j-1-begin-blocks[l].glo_col;

                    A[_j*ldA+_i] = val;

                    break;
                }
            }
        }
    }
#endif

    fclose(file);

    return matrix;
}

////////////////////////////////////////////////////////////////////////////////

static int write_raw_crawler(
    int offset, int width, int m, int n, int count, size_t *lds,
    void **ptrs, void *arg)
{
    FILE *file = arg;

    double *A = ptrs[0];
    size_t ldA = lds[0];

    for (int i = 0; i < width; i++) {
        int ret = fwrite(&A[i*ldA], sizeof(double), m, file);
        if (ret < m) {
            fprintf(stderr,
                "write_raw_crawler failed to write the matrix.\n");
            abort();
        }
    }

    return width;
}

void write_raw_matrix_to_file(char const *name, matrix_t matrix)
{
    FILE *file = fopen(name, "wb");
    if (file == NULL) {
        fprintf(stderr, "Invalid filename.\n");
        abort();
    }

    int m = GENERIC_MATRIX_M(matrix);
    int n = GENERIC_MATRIX_N(matrix);

    if (fprintf(file, "STARNEIG RAW REAL DOUBLE M %d N %d\n", m, n) < 0) {
        fprintf(stderr, "write_raw_crawler write error.\n");
        abort();
    }

    crawl_matrices(CRAWLER_R, CRAWLER_PANEL,
        &write_raw_crawler, file, 0, matrix, NULL);

    fclose(file);
}

void read_raw_dimensions_from_file(char const *name, int *m, int *n)
{
    FILE *file = fopen(name, "rb");
    if (file == NULL) {
        fprintf(stderr, "Invalid filename.\n");
        abort();
    }

    if (fscanf(file, "STARNEIG RAW REAL DOUBLE M %d N %d\n", m, n) == EOF) {
        fprintf(stderr, "Invalid file.\n");
        abort();
    }

    if (*m < 1 || *n < 1)  {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        abort();
    }

    fclose(file);
}

matrix_t read_raw_matrix_from_file(char const *name, init_helper_t helper)
{
    int m, n;
    read_raw_dimensions_from_file(name, &m, &n);
    return read_raw_sub_matrix_from_file(0, MIN(m, n), name, helper);
}

matrix_t read_raw_sub_matrix_from_file(
    int begin, int end, char const *name, init_helper_t helper)
{
    int m, n;
    read_raw_dimensions_from_file(name, &m, &n);

    FILE *file = fopen(name, "rb");

    // skip header
    fpos_t data_begin;
    while (fgetc(file) != '\n');
    fgetpos(file, &data_begin);

    if (begin < 0 || end < begin || m < end || n < end) {
        fprintf(stderr, "Invalid submatrix dimension.\n");
        abort();
    }

    if (begin == 0 && end == m && end == n)
        printf("READING A %d X %d MATRIX ...\n", m, n);
    else
        printf("READING DIAGONAL SUBMATRIX [%d,%d[ FROM A %d X %d MATRIX ...\n",
            begin, end, m, n);

    matrix_t matrix = init_matrix(end-begin, end-begin, helper);

    if (matrix->type == LOCAL_MATRIX) {
        double *A = LOCAL_MATRIX_PTR(matrix);
        size_t ldA = LOCAL_MATRIX_LD(matrix);

        for (int i = 0; i < n; i++) {
            fsetpos(file, &data_begin);
            fseek(file, ((size_t)(begin+i)*n+begin) * sizeof(double), SEEK_CUR);
            int ret = fread(&A[i*ldA], sizeof(double), end-begin, file);
            if (ret < end-begin) {
                fprintf(stderr,
                    "read_raw_crawler encountered an invalid file.\n");
                abort();
            }
        }

    }
#ifdef STARNEIG_ENABLE_MPI
    if (matrix->type == STARNEIG_MATRIX || matrix->type == BLACS_MATRIX) {
        struct starneig_distr_block *blocks;
        int num_blocks;
        starneig_distr_matrix_get_blocks(
            STARNEIG_MATRIX_HANDLE(matrix), &blocks, &num_blocks);

        for (int k = 0; k < num_blocks; k++) {
            double *A = blocks[k].ptr;
            size_t ldA = blocks[k].ld;

            for (int i = 0; i < blocks[k].col_blksz; i++) {
                fsetpos(file, &data_begin);
                fseek(file,
                    ((size_t)(begin+blocks[k].glo_col+i)*n +
                    begin+blocks[k].glo_row) * sizeof(double), SEEK_CUR);

                int ret = fread(
                    &A[i*ldA], sizeof(double), blocks[k].row_blksz, file);
                if (ret < blocks[k].row_blksz) {
                    fprintf(stderr,
                        "read_raw_crawler encountered an invalid file.\n");
                    abort();
                }
            }
        }
    }
#endif

    fclose(file);

    return matrix;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void store_raw_pencil_print_usage(int argc, char * const *argv)
{
    printf(
        "  --store-raw-output (output data filename in format XXX_%%s.xxx)\n"
    );
}

static void store_raw_pencil_print_args(int argc, char * const *argv)
{
    printf(" --store-raw-output %s",
        read_str("--store-raw-output", argc, argv, NULL, "output_%s.dat"));
}

static int store_raw_pencil_check_args(int argc, char * const *argv, int *argr)
{
    char const *input =
        read_str("--store-raw-output", argc, argv, argr, "output_%s.dat");

    if ((input = strstr(input, "%")) == NULL)
        return 1;
    if (*(++input) != 's')
        return 1;
    if (strstr(input, "%") != NULL)
        return 1;

    return 0;
}

static int store_raw_pencil_init(int argc, char * const *argv, int repeat,
    int warmup, hook_state_t *state)
{
    *state = (hook_state_t) read_str(
        "--store-raw-output", argc, argv, NULL, "output_%s.dat");
    return 0;
}

static hook_return_t store_raw_pencil_after_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    char const *name = state;
    pencil_t pencil = (pencil_t) env->data;

    store_supplementary(name, pencil->supp);

    char *filename = malloc(strlen(name) + 10);

    if (pencil->mat_a) {
        sprintf(filename, name, "A");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_a);
    }

    if (pencil->mat_b) {
        sprintf(filename, name, "B");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_b);
    }

    if (pencil->mat_q) {
        sprintf(filename, name, "Q");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_q);
    }

    if (pencil->mat_z) {
        sprintf(filename, name, "Z");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_z);
    }

    if (pencil->mat_x) {
        sprintf(filename, name, "X");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_x);
    }

    if (pencil->mat_ca) {
        sprintf(filename, name, "CA");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_ca);
    }

    if (pencil->mat_cb) {
        sprintf(filename, name, "CB");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_cb);
    }

    free(filename);

    return HOOK_SUCCESS;
}

const struct hook_t store_raw_pencil = {
    .name = "store-raw",
    .desc = "Writes the output matrix pencil to files",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &store_raw_pencil_print_usage,
    .print_args = &store_raw_pencil_print_args,
    .check_args = &store_raw_pencil_check_args,
    .init = &store_raw_pencil_init,
    .after_solver_run = &store_raw_pencil_after_solver_run,
};

const struct hook_descr_t default_store_raw_pencil_descr = {
    .is_enabled = 0,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &store_raw_pencil
};

////////////////////////////////////////////////////////////////////////////////

static void store_raw_input_pencil_print_usage(int argc, char * const *argv)
{
    printf(
        "  --store-raw-input (input data filename in format XXX_%%s.xxx)\n"
    );
}

static void store_raw_input_pencil_print_args(int argc, char * const *argv)
{
    printf(" --store-raw-input %s",
        read_str("--store-raw-input", argc, argv, NULL, "input_%s.dat"));
}

static int store_raw_input_pencil_check_args(
    int argc, char * const *argv, int *argr)
{
    char const *input =
        read_str("--store-raw-input", argc, argv, argr, "input_%s.dat");

    if ((input = strstr(input, "%")) == NULL)
        return 1;
    if (*(++input) != 's')
        return 1;
    if (strstr(input, "%") != NULL)
        return 1;

    return 0;
}

static int store_raw_input_pencil_init(int argc, char * const *argv, int repeat,
    int warmup, hook_state_t *state)
{
    *state = (hook_state_t) read_str(
        "--store-raw-input", argc, argv, NULL, "input_%s.dat");
    return 0;
}

static hook_return_t store_raw_input_pencil_before_solver_run(
    int iter, hook_state_t state, struct hook_data_env *env)
{
    char const *name = state;
    pencil_t pencil = (pencil_t) env->data;

    store_supplementary(name, pencil->supp);

    char *filename = malloc(strlen(name) + 10);

    if (pencil->mat_a) {
        sprintf(filename, name, "A");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_a);
    }

    if (pencil->mat_b) {
        sprintf(filename, name, "B");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_b);
    }

    if (pencil->mat_q) {
        sprintf(filename, name, "Q");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_q);
    }

    if (pencil->mat_z) {
        sprintf(filename, name, "Z");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_z);
    }

    if (pencil->mat_x) {
        sprintf(filename, name, "X");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_x);
    }

    if (pencil->mat_ca) {
        sprintf(filename, name, "CA");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_ca);
    }

    if (pencil->mat_cb) {
        sprintf(filename, name, "CB");
        printf("WRITING TO %s...\n", filename);
        write_raw_matrix_to_file(filename, pencil->mat_cb);
    }

    free(filename);

    return HOOK_SUCCESS;
}

const struct hook_t store_raw_input_pencil = {
    .name = "store-raw-input",
    .desc = "Writes the input matrix pencil to files",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &store_raw_input_pencil_print_usage,
    .print_args = &store_raw_input_pencil_print_args,
    .check_args = &store_raw_input_pencil_check_args,
    .init = &store_raw_input_pencil_init,
    .before_solver_run = &store_raw_input_pencil_before_solver_run,
};

const struct hook_descr_t default_store_raw_input_pencil_descr = {
    .is_enabled = 0,
    .default_mode = HOOK_MODE_NORMAL,
    .hook = &store_raw_input_pencil
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void mtx_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --left-input (mtx filename) -- Left-hand side matrix input "
        "file name\n"
        "  --right-input (mtx filename) -- Right-hand side matrix input "
        "file name\n"
        "  --input-begin (num) -- First matrix row/column to be read\n"
        "  --input-end (num) -- Last matrix row/column to be read + 1\n"
    );

    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void mtx_initializer_print_args(int argc, char * const *argv)
{
    char const *left_input = read_str("--left-input", argc, argv, NULL, NULL);
    char const *right_input = read_str("--right-input", argc, argv, NULL, NULL);

    printf(" --left-input %s", left_input);
    if (right_input)
        printf(" --right-input %s", right_input);
    int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
    int input_end = read_int("--input-end", argc, argv, NULL, -1);
    if (0 <= input_begin && 0 <= input_end)
        printf(" --input-begin %d --input-end %d", input_begin, input_end);

    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int mtx_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    char const *left_input = read_str("--left-input", argc, argv, argr, NULL);
    char const *right_input = read_str("--right-input", argc, argv, argr, NULL);

    if (left_input == NULL) {
        fprintf(stderr, "Left-hand side input filename is missing.\n");
        return 1;
    }

    if (access(left_input, R_OK) != 0) {
        fprintf(stderr, "Left-hand side input file does not exists.\n");
        return 1;
    }

    if (right_input != NULL && access(right_input, R_OK) != 0) {
        fprintf(stderr, "Right-hand side input file does not exists.\n");
        return 1;
    }

    int input_begin = read_int("--input-begin", argc, argv, argr, -1);
    int input_end = read_int("--input-end", argc, argv, argr, -1);

    if (input_begin < 0 && input_end < input_begin)
        return 1;

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static struct hook_data_env* mtx_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT...\n");

    char const *left_input = read_str("--left-input", argc, argv, NULL, NULL);
    char const *right_input = read_str("--right-input", argc, argv, NULL, NULL);
    int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
    int input_end = read_int("--input-end", argc, argv, NULL, -1);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t data = env->data = init_pencil();

    int m, n;
    read_mtx_dimensions_from_file(left_input, &m, &n);

    if (input_begin == -1)
        input_begin = 0;
    if (input_end == -1)
        input_end = n;

    init_helper_t helper = init_helper_init_hook(
        "", format, m, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    data->mat_a = read_mtx_sub_matrix_from_file(
        input_begin, input_end, left_input, helper);
    data->mat_q = generate_identity(n, n, helper);

    if (right_input != NULL) {
        data->mat_b = read_mtx_sub_matrix_from_file(
            input_begin, input_end, right_input, helper);
        data->mat_z = generate_identity(n, n, helper);
    }

    init_helper_free(helper);

    return env;
}

const struct hook_initializer_t mtx_initializer = {
    .name = "read-mtx",
    .desc = "Reads the matrix pencil from a mtx file",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &mtx_initializer_print_usage,
    .print_args = &mtx_initializer_print_args,
    .check_args = &mtx_initializer_check_args,
    .init = &mtx_initializer_init
};

////////////////////////////////////////////////////////////////////////////////

static void raw_initializer_print_usage(int argc, char * const *argv)
{
    printf(
        "  --input (input filename in format XXX_%%s.xxx)\n"
        "  --input-only -- Read only necessary input matrices\n"
        "  --input-begin (num) -- First matrix row/column to be read\n"
        "  --input-end (num) -- Last matrix row/column to be read + 1\n"
    );

    init_helper_print_usage("", INIT_HELPER_ALL, argc, argv);
}

static void raw_initializer_print_args(int argc, char * const *argv)
{
    printf(" --input %s", read_str("--input", argc, argv, NULL, NULL));
    if (read_opt("--input-only", argc, argv, NULL))
        printf(" --input-only");
    int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
    int input_end = read_int("--input-end", argc, argv, NULL, -1);
    if (0 <= input_begin && 0 <= input_end)
        printf(" --input-begin %d --input-end %d", input_begin, input_end);

    init_helper_print_args("", INIT_HELPER_ALL, argc, argv);
}

static int raw_initializer_check_args(
    int argc, char * const *argv, int *argr)
{
    char const *input = read_str("--input", argc, argv, argr, NULL);
    if (input == NULL)
        return 1;

    char const *pos = input;
    if ((pos = strstr(pos, "%")) == NULL)
        return 1;
    if (*(++pos) != 's')
        return 1;
    if (strstr(pos, "%") != NULL)
        return 1;

    char *filename = malloc(strlen(input) + 2);
    sprintf(filename, input, "A");

    if (access(filename, R_OK) != 0) {
        fprintf(stderr, "Input file does not exists.\n");
        free(filename);
        return 1;
    }

    free(filename);

    read_opt("--input-only", argc, argv, argr);

    int input_begin = read_int("--input-begin", argc, argv, argr, -1);
    int input_end = read_int("--input-end", argc, argv, argr, -1);
    if (input_begin < 0 && input_end < input_begin)
        return 1;

    return init_helper_check_args("", INIT_HELPER_ALL, argc, argv, argr);
}

static struct hook_data_env* raw_initializer_init(
    hook_data_format_t format, int argc, char * const *argv)
{
    printf("INIT... \n");

    char const *input = read_str("--input", argc, argv, NULL, NULL);
    int input_only = read_opt("--input-only", argc, argv, NULL);
    int input_begin = read_int("--input-begin", argc, argv, NULL, -1);
    int input_end = read_int("--input-end", argc, argv, NULL, -1);

    struct hook_data_env *env = malloc(sizeof(struct hook_data_env));
    env->format = format;
    env->copy_data = (hook_data_env_copy_t) copy_pencil;
    env->free_data = (hook_data_env_free_t) free_pencil;
    pencil_t pencil = env->data = init_pencil();

    char *filename = malloc(strlen(input) + 10);

    sprintf(filename, input, "A");

    int m, n;
    read_raw_dimensions_from_file(filename, &m, &n);

    if (input_begin == -1)
        input_begin = 0;
    if (input_end == -1)
        input_end = n;

    load_supplementary(input_begin, input_end, input, &pencil->supp);

    init_helper_t helper = init_helper_init_hook(
        "", format, m, n, PREC_DOUBLE | NUM_REAL, argc, argv);

    if (access(filename, R_OK) == 0) {
        printf("READING FROM %s...\n", filename);
        pencil->mat_a = read_raw_sub_matrix_from_file(
            input_begin, input_end, filename, helper);
    }

    if (0 < input_begin || input_end < n) {
        printf("INITIALIZING Q WITH A RANDOM HOUSEHOLDER REFLECTOR...\n");
        pencil->mat_q = generate_random_householder(n, helper);
    }
    else {
        sprintf(filename, input, "Q");
        if (access(filename, R_OK) == 0) {
            printf("READING FROM %s...\n", filename);
            pencil->mat_q = read_raw_matrix_from_file(filename, helper);
        }
        sprintf(filename, input, "CA");
        if (!input_only && access(filename, R_OK) == 0) {
            printf("READING FROM %s...\n", filename);
            pencil->mat_ca = read_raw_matrix_from_file(filename, helper);
        }
    }

    sprintf(filename, input, "B");
    if (access(filename, R_OK) == 0) {
        printf("READING FROM %s...\n", filename);
        pencil->mat_b = read_raw_sub_matrix_from_file(
            input_begin, input_end, filename, helper);

        if (0 < input_begin || input_end < n) {
            printf("INITIALIZING Z WITH A RANDOM HOUSEHOLDER REFLECTOR...\n");
            pencil->mat_z =
                generate_random_householder(n, helper);
        }
        else {
            sprintf(filename, input, "Z");
            if (access(filename, R_OK) == 0) {
                printf("READING FROM %s...\n", filename);
                pencil->mat_z = read_raw_matrix_from_file(filename, helper);
            }
            sprintf(filename, input, "CB");
            if (!input_only && access(filename, R_OK) == 0) {
                printf("READING FROM %s...\n", filename);
                pencil->mat_cb = read_raw_matrix_from_file(filename, helper);
            }
        }
    }

    init_helper_free(helper);
    free(filename);

    return env;
}

const struct hook_initializer_t raw_initializer = {
    .name = "read-raw",
    .desc = "Reads the matrix pencil from a file",
    .formats = (hook_data_format_t[]) {
        HOOK_DATA_FORMAT_PENCIL_LOCAL,
#ifdef STARNEIG_ENABLE_MPI
        HOOK_DATA_FORMAT_PENCIL_STARNEIG,
#endif
#ifdef STARNEIG_ENABLE_BLACS
        HOOK_DATA_FORMAT_PENCIL_BLACS,
#endif
        0 },
    .print_usage = &raw_initializer_print_usage,
    .print_args = &raw_initializer_print_args,
    .check_args = &raw_initializer_check_args,
    .init = &raw_initializer_init
};
