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
#include "supplementary.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

static void ** create_supplementary(
    supplementary_type_t type, struct supplementary **supp)
{
    struct supplementary **iter = supp;
    while (*iter != NULL) {
        assert((*iter)->type != type);
        iter = &(*iter)->next;
    }

    *iter = malloc(sizeof(struct supplementary));
    (*iter)->type = type;
    (*iter)->ptr = NULL;
    (*iter)->next = NULL;

    return &(*iter)->ptr;
}

static void * get_load(
    supplementary_type_t type, struct supplementary const *supp)
{
    struct supplementary const *iter = supp;
    while (iter != NULL) {
        if (iter->type == type)
            return iter->ptr;
        iter = iter->next;
    }

    return NULL;
}

////////////////////////////////////////////////////////////////////////////////

struct selected_load {
    int *selected;
    size_t size;
};

static void free_selected(void *ptr)
{
    struct selected_load *load = ptr;

    if (load == NULL)
        return;

    if (load->selected != NULL)
        free(load->selected);
    free(load);
}

static void * copy_selected(void const *ptr)
{
    struct selected_load const *load = ptr;

    if (load == NULL)
        return NULL;

    struct selected_load *new_load = malloc(sizeof(struct selected_load));

    if (load->selected != NULL) {
        new_load->selected = malloc(load->size*sizeof(int));
        new_load->size = load->size;
        memcpy(new_load->selected, load->selected, load->size*sizeof(int));
    }
    else {
        new_load->selected = NULL;
        new_load->size = 0;
    }

    return new_load;
}

#ifdef STARNEIG_ENABLE_MPI

static void broadcast_selected(int root, MPI_Comm communicator, void **ptr)
{
    struct selected_load **load = (struct selected_load **) ptr;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int defined = 0;
    if (root == my_rank && *load != NULL && (*load)->selected != NULL)
        defined = 1;
    MPI_Bcast(&defined, 1, MPI_INT, root, MPI_COMM_WORLD);

    // sanitize receiving ranks
    if (root != my_rank) {
        free_selected(*load);
        *load = NULL;
    }

    if (defined) {
        if (root != my_rank)
            *load = malloc(sizeof(struct selected_load));

        MPI_Bcast(&(*load)->size, sizeof((*load)->size), MPI_BYTE,
            root, MPI_COMM_WORLD);

        if (root != my_rank)
            (*load)->selected = malloc((*load)->size*sizeof(int));

        MPI_Bcast((*load)->selected, (*load)->size, MPI_INT,
            root, MPI_COMM_WORLD);
    }
}

#endif // STARNEIG_ENABLE_MPI

static void print_selected(void *ptr)
{
    struct selected_load *load = ptr;

    if (load == NULL)
        return;

    printf("Selected:\n");
    for (int i = 0; i < load->size; i++) {
        if (load->selected[i])
            printf("  selected ");
        else
            printf("  -------- ");
    }
    printf("\n");
}

void init_supplementary_selected(
    size_t size, int **selected, struct supplementary **supp)
{
    if (selected != NULL)
        *selected = NULL;

    struct selected_load **load = (struct selected_load **)
        create_supplementary(SUPPLEMENTARY_SELECTED, supp);

    if (0 < size) {
        *load = malloc(sizeof(struct selected_load));
        (*load)->selected = malloc(size*sizeof(int));
        (*load)->size = size;
        if (selected != NULL)
            *selected = (*load)->selected;
    }
}

int* get_supplementaty_selected(struct supplementary const *supp)
{
    struct selected_load *load = get_load(SUPPLEMENTARY_SELECTED, supp);

    if (load != NULL)
        return load->selected;

    return NULL;
}

////////////////////////////////////////////////////////////////////////////////

struct eigenvalues_load {
    double *real;
    double *imag;
    double *beta;
    size_t size;
};

static void free_eigenvalues(void *ptr)
{
    struct eigenvalues_load *load = ptr;

    if (load == NULL)
        return;

    free(load->real);
    free(load->imag);
    free(load->beta);
    free(load);
}

static void * copy_eigenvalues(void const *ptr)
{
    struct eigenvalues_load const *load = ptr;

    if (load == NULL)
        return NULL;

    struct eigenvalues_load *new_load = malloc(sizeof(struct eigenvalues_load));

    if (0 < load->size) {
        new_load->real = malloc(load->size*sizeof(double));
        new_load->imag = malloc(load->size*sizeof(double));
        new_load->beta = malloc(load->size*sizeof(double));
        memcpy(new_load->real, load->real, load->size*sizeof(double));
        memcpy(new_load->imag, load->imag, load->size*sizeof(double));
        memcpy(new_load->beta, load->beta, load->size*sizeof(double));
        new_load->size = load->size;
    }
    else {
        new_load->real = NULL;
        new_load->imag = NULL;
        new_load->beta = NULL;
        new_load->size = 0;
    }

    return new_load;
}

#ifdef STARNEIG_ENABLE_MPI

static void broadcast_eigenvalues(int root, MPI_Comm communicator, void **ptr)
{
    struct eigenvalues_load **load = (struct eigenvalues_load **) ptr;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int defined = 0;
    if (root == my_rank && *load != NULL && 0 < (*load)->size)
        defined = 1;
    MPI_Bcast(&defined, 1, MPI_INT, root, MPI_COMM_WORLD);

    // sanitize receiving ranks
    if (root != my_rank) {
        free_eigenvalues(*load);
        *load = NULL;
    }

    if (defined) {
        if (root != my_rank)
            *load = malloc(sizeof(struct eigenvalues_load));

        MPI_Bcast(&(*load)->size, sizeof((*load)->size), MPI_BYTE,
            root, MPI_COMM_WORLD);

        if (root != my_rank) {
            (*load)->real = malloc((*load)->size*sizeof(double));
            (*load)->imag = malloc((*load)->size*sizeof(double));
            (*load)->beta = malloc((*load)->size*sizeof(double));
        }

        MPI_Bcast((*load)->real, (*load)->size, MPI_DOUBLE,
            root, MPI_COMM_WORLD);
        MPI_Bcast((*load)->imag, (*load)->size, MPI_DOUBLE,
            root, MPI_COMM_WORLD);
        MPI_Bcast((*load)->beta, (*load)->size, MPI_DOUBLE,
            root, MPI_COMM_WORLD);
    }
}

#endif // STARNEIG_ENABLE_MPI

static void print_eigenvalues(void *ptr)
{
    struct eigenvalues_load *load = ptr;

    if (load == NULL)
        return;

    printf("Eigenvalues:\n");
    for (int i = 0; i < load->size; i++) {
        if (load->real != NULL)
            printf("%10f ", load->real[i]);
        else
            printf("  -------- ");
    }
    printf("\n");
    for (int i = 0; i < load->size; i++) {
        if (load->imag != NULL)
            printf("%10f ", load->imag[i]);
        else
            printf("  -------- ");
    }
    printf("\n");
    for (int i = 0; i < load->size; i++) {
        if (load->beta != NULL)
            printf("%10f ", load->beta[i]);
        else
            printf("  -------- ");
    }
    printf("\n");
}

void init_supplementary_eigenvalues(
    size_t size, double **real, double **imag, double **beta,
    struct supplementary **supp)
{
    if (real != NULL)
        *real = NULL;
    if (imag != NULL)
        *imag = NULL;
    if (beta != NULL)
        *beta = NULL;

    struct eigenvalues_load **load = (struct eigenvalues_load **)
        create_supplementary(SUPPLEMENTARY_EIGENVALUES, supp);

    if (0 < size) {
        *load = malloc(sizeof(struct eigenvalues_load));
        (*load)->real = malloc(size*sizeof(double));
        (*load)->imag = malloc(size*sizeof(double));
        (*load)->beta = malloc(size*sizeof(double));
        for (int i = 0; i < size; i++) {
            (*load)->real[i] = 0.0;
            (*load)->imag[i] = 0.0;
            (*load)->beta[i] = 1.0;
        }
        (*load)->size = size;
        if (real != NULL)
            *real = (*load)->real;
        if (imag != NULL)
            *imag = (*load)->imag;
        if (beta != NULL)
            *beta = (*load)->beta;
    }
}

void get_supplementaty_eigenvalues(struct supplementary const *supp,
    double **real, double **imag, double **beta)
{
    struct eigenvalues_load *load = get_load(SUPPLEMENTARY_EIGENVALUES, supp);

    *real = *imag = *beta = NULL;
    if (load != NULL) {
        *real = load->real;
        *imag = load->imag;
        *beta = load->beta;
    }
}

////////////////////////////////////////////////////////////////////////////////

static void print_known_eigenvalues(void *ptr)
{
    struct eigenvalues_load *load = ptr;

    if (load == NULL)
        return;

    printf("Known eigenvalues:\n");
    for (int i = 0; i < load->size; i++) {
        if (load->real != NULL)
            printf("%10f ", load->real[i]);
        else
            printf("  -------- ");
    }
    printf("\n");
    for (int i = 0; i < load->size; i++) {
        if (load->imag != NULL)
            printf("%10f ", load->imag[i]);
        else
            printf("  -------- ");
    }
    printf("\n");
    for (int i = 0; i < load->size; i++) {
        if (load->beta != NULL)
            printf("%10f ", load->beta[i]);
        else
            printf("  -------- ");
    }
    printf("\n");
}

void load_known_eigenvalues(int begin, int end, char const *name, void **ptr)
{
    struct eigenvalues_load **load = (struct eigenvalues_load **) ptr;

    FILE *file = fopen(name, "rb");
    if (file == NULL) {
        fprintf(stderr,
            "load_known_eigenvalues encountered an invalid filename.\n");
        abort();
    }

    int n;
    if (fscanf(file,
    "STARNEIG SUPPLEMENTARY KNOWN EIGENVALUES N %d\n", &n) == EOF) {
        fprintf(stderr,
            "load_known_eigenvalues encountered an invalid file.\n");
        abort();
    }
    fseek(file, 0, SEEK_SET);

    if (n < end)  {
        fprintf(stderr,
            "load_known_eigenvalues encountered an invalid file.\n");
        abort();
    }

    printf("READING %d KNOWN EIGENVALUES...\n", end-begin);

    fpos_t data_begin;
    while (fgetc(file) != '\n');
    fgetpos(file, &data_begin);

    *load = malloc(sizeof(struct eigenvalues_load));
    (*load)->size = end-begin;

    {
        (*load)->real = malloc((*load)->size*sizeof(double));
        fsetpos(file, &data_begin);
        fseek(file, (0*n + begin) * sizeof(double), SEEK_CUR);
        int ret = fread((*load)->real , sizeof(double), end-begin, file);
        if (ret < end-begin) {
            fprintf(stderr,
                "load_known_eigenvalues encountered an invalid file.\n");
            abort();
        }
    }

    {
        (*load)->imag = malloc((*load)->size*sizeof(double));
        fsetpos(file, &data_begin);
        fseek(file, (1*n + begin) * sizeof(double), SEEK_CUR);
        int ret = fread((*load)->imag , sizeof(double), end-begin, file);
        if (ret < end-begin) {
            fprintf(stderr,
                "load_known_eigenvalues encountered an invalid file.\n");
            abort();
        }
    }

    {
        (*load)->beta = malloc((*load)->size*sizeof(double));
        fsetpos(file, &data_begin);
        fseek(file, (2*n + begin) * sizeof(double), SEEK_CUR);
        int ret = fread((*load)->beta , sizeof(double), end-begin, file);
        if (ret < end-begin) {
            fprintf(stderr,
                "load_known_eigenvalues encountered an invalid file.\n");
            abort();
        }
    }

    fclose(file);
}

void store_known_eigenvalues(char const *name, void const *ptr)
{
    struct eigenvalues_load *load = (struct eigenvalues_load *) ptr;

    FILE *file = fopen(name, "wb");
    if (file == NULL) {
        fprintf(stderr,
            "store_known_eigenvalues encountered an invalid filename.\n");
        abort();
    }

    if (fprintf(file,
    "STARNEIG SUPPLEMENTARY KNOWN EIGENVALUES N %d\n", (int) load->size) < 0) {
        fprintf(stderr,
            "store_known_eigenvalues encountered a write error.\n");
        abort();
    }

    if (fwrite(load->real , sizeof(double), load->size, file) < load->size) {
        fprintf(stderr,
            "store_known_eigenvalues encountered a write error.\n");
        abort();
    }

    if (fwrite(load->imag , sizeof(double), load->size, file) < load->size) {
        fprintf(stderr,
            "store_known_eigenvalues encountered a write error.\n");
        abort();
    }

    if (fwrite(load->beta , sizeof(double), load->size, file) < load->size) {
        fprintf(stderr,
            "store_known_eigenvalues encountered a write error.\n");
        abort();
    }

    fclose(file);
}

void init_supplementary_known_eigenvalues(
    size_t size, double **real, double **imag, double **beta,
    struct supplementary **supp)
{
    if (real != NULL)
        *real = NULL;
    if (imag != NULL)
        *imag = NULL;
    if (beta != NULL)
        *beta = NULL;

    struct eigenvalues_load **load = (struct eigenvalues_load **)
        create_supplementary(SUPPLEMENTARY_KNOWN_EIGENVALUES, supp);

    if (0 < size) {
        *load = malloc(sizeof(struct eigenvalues_load));
        (*load)->real = malloc(size*sizeof(double));
        (*load)->imag = malloc(size*sizeof(double));
        (*load)->beta = malloc(size*sizeof(double));
        for (int i = 0; i < size; i++) {
            (*load)->real[i] = 0.0;
            (*load)->imag[i] = 0.0;
            (*load)->beta[i] = 1.0;
        }
        (*load)->size = size;
        if (real != NULL)
            *real = (*load)->real;
        if (imag != NULL)
            *imag = (*load)->imag;
        if (beta != NULL)
            *beta = (*load)->beta;
    }
}

void get_supplementaty_known_eigenvalues(struct supplementary const *supp,
    double **real, double **imag, double **beta)
{
    struct eigenvalues_load *load =
        get_load(SUPPLEMENTARY_KNOWN_EIGENVALUES, supp);

    *real = *imag = *beta = NULL;
    if (load != NULL) {
        *real = load->real;
        *imag = load->imag;
        *beta = load->beta;
    }
}

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Supplementary data handler descriptor.
///
struct handler {
    supplementary_type_t type;                      ///< type
    char const *name;                               ///< name
    void (*free)(void *);                           ///< free function
    void * (*copy)(void const *);                   ///< copy function
    void (*load)(int, int, char const *, void **);  ///< load function
    void (*store)(char const *, void const *);      ///< store function
#ifdef STARNEIG_ENABLE_MPI
    void (*broadcast)(int, MPI_Comm, void **);      ///< broadcast function
#endif
    void (*print)(void *);                          ///< print function
};

///
/// @brief Supplementary data handlers.
///
static const struct handler handlers[] = {
    {
        .type = SUPPLEMENTARY_SELECTED,
        .free = free_selected,
        .copy = copy_selected,
#ifdef STARNEIG_ENABLE_MPI
        .broadcast = broadcast_selected,
#endif
        .print = print_selected
    },
    {
        .type = SUPPLEMENTARY_EIGENVALUES,
        .free = free_eigenvalues,
        .copy = copy_eigenvalues,
#ifdef STARNEIG_ENABLE_MPI
        .broadcast = broadcast_eigenvalues,
#endif
        .print = print_eigenvalues
    },
    {
        .type = SUPPLEMENTARY_KNOWN_EIGENVALUES,
        .name = "known_eigenvalues",
        .free = free_eigenvalues,
        .copy = copy_eigenvalues,
        .load = load_known_eigenvalues,
        .store = store_known_eigenvalues,
#ifdef STARNEIG_ENABLE_MPI
        .broadcast = broadcast_eigenvalues,
#endif
        .print = print_known_eigenvalues
    }
};

///
/// @brief Returns a supplementary data handler that matches a give type.
///
/// @param[in] type  supplementary data type
///
/// @return matching supplementary data handler if one exists, NULL otherwise
///
static struct handler const * get_handler(supplementary_type_t type)
{
    for (int i = 0; i < sizeof(handlers)/sizeof(handlers[0]); i++)
        if (handlers[i].type == type)
            return &handlers[i];
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void free_supplementary(struct supplementary *supp)
{
    struct supplementary *iter = supp;
    while (iter != NULL) {
        struct supplementary *next = iter->next;
        struct handler const * handler = get_handler(iter->type);
        assert(handler != NULL && handler->free != NULL);
        handler->free(iter->ptr);
        free(iter);
        iter = next;
    }
}

struct supplementary * copy_supplementary(struct supplementary const *supp)
{
    struct supplementary *new = NULL;

    struct supplementary const *iter = supp;
    while (iter != NULL) {
        struct handler const * handler = get_handler(iter->type);
        assert(handler != NULL && handler->copy != NULL);

        void **load = create_supplementary(iter->type, &new);
        *load = handler->copy(iter->ptr);

        iter = iter->next;
    }

    return new;
}

void load_supplementary(
    int begin, int end, char const *name, struct supplementary **supp)
{
    for (int i = 0; i < sizeof(handlers)/sizeof(handlers[0]); i++) {
        if (handlers[i].load != NULL) {
            assert(handlers[i].name != NULL);
            char *filename = malloc(strlen(name) + strlen(handlers[i].name));
            sprintf(filename, name, handlers[i].name);

            if (access(filename, R_OK) == 0) {
                printf("READING FROM %s...\n", filename);
                void **load = create_supplementary(handlers[i].type, supp);
                handlers[i].load(begin, end, filename, load);
            }

            free(filename);
        }
    }
}

void store_supplementary(char const *name, struct supplementary *supp)
{
    struct supplementary *iter = supp;
    while (iter != NULL) {
        struct handler const * handler = get_handler(iter->type);
        if (handler->store != NULL && iter->ptr != NULL) {
            assert(handler->name != NULL);
            char *filename = malloc(strlen(name) + strlen(handler->name));
            sprintf(filename, name, handler->name);

            printf("WRITING TO %s...\n", filename);
            handler->store(filename, iter->ptr);
        }
        iter = iter->next;
    }
}

#ifdef STARNEIG_ENABLE_MPI

void broadcast_supplementary(
    int root, MPI_Comm communicator, struct supplementary **supp)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == root) {
        struct supplementary *iter = *supp;
        while (iter != NULL) {

            int defined = 1;
            MPI_Bcast(&defined, 1, MPI_INT, root, MPI_COMM_WORLD);

            MPI_Bcast(&iter->type, sizeof(supplementary_type_t), MPI_BYTE,
                root, MPI_COMM_WORLD);

            struct handler const * handler = get_handler(iter->type);
            assert(handler != NULL && handler->broadcast != NULL);

            handler->broadcast(root, communicator, &iter->ptr);

            iter = iter->next;
        }
        int defined = 0;
        MPI_Bcast(&defined, 1, MPI_INT, root, MPI_COMM_WORLD);
    }
    else {
        if (*supp != NULL) {
            free_supplementary(*supp);
            *supp = NULL;
        }

        int defined = 0;
        MPI_Bcast(&defined, 1, MPI_INT, root, MPI_COMM_WORLD);
        while (defined) {

            supplementary_type_t type;
            MPI_Bcast(&type, sizeof(supplementary_type_t), MPI_BYTE,
                root, MPI_COMM_WORLD);

            struct handler const * handler = get_handler(type);
            assert(handler != NULL && handler->broadcast != NULL);

            void **load = create_supplementary(type, supp);
            handler->broadcast(root, communicator, load);

            MPI_Bcast(&defined, 1, MPI_INT, root, MPI_COMM_WORLD);
        }
    }
}

#endif // STARNEIG_ENABLE_MPI

void print_supplementary(struct supplementary const *supp)
{
    struct supplementary const *iter = supp;
    while (iter != NULL) {
        struct handler const * handler = get_handler(iter->type);
        assert(handler != NULL);
        if (handler->print != NULL)
            handler->print(iter->ptr);
        iter = iter->next;
    }
}
