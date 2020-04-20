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
#include "hook_experiment.h"
#include "hook_converter.h"
#include "parse.h"
#include "checks.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#include <starpu.h>

#ifdef STARNEIG_ENABLE_MPI
#include <mpi.h>
#endif

#ifdef STARNEIG_ENABLE_MPI
#include "starneig_pencil.h"
#endif

///
/// @brief Data format descriptor.
///
struct data_format {
    hook_data_format_t type;    ///< data format label
    char const *name;           ///< data format name
    char const *desc;           ///< data format description
    int distributed;            ///< non-zero if the data format is distributed
};

///
/// @brief Data formats.
///
static const struct data_format data_formats[] = {
    {
        .type = HOOK_DATA_FORMAT_ANY,
        .name = "any",
        .desc = "Any data format (for internal use only)"
    },
    {
        .type = HOOK_DATA_FORMAT_GENERIC,
        .name = "generic",
        .desc = "Generic data format"
    },
    {
        .type = HOOK_DATA_FORMAT_PENCIL_LOCAL,
        .name = "pencil-local",
        .desc = "Local matrix pencil data format"
    },
#ifdef STARNEIG_ENABLE_MPI
    {
        .type = HOOK_DATA_FORMAT_PENCIL_STARNEIG,
        .name = "pencil-starneig",
        .desc = "StarNEig library style distributed matrix pencil data format",
        .distributed = 1
    },
#endif
#ifdef STARNEIG_ENABLE_BLACS
    {
        .type = HOOK_DATA_FORMAT_PENCIL_BLACS,
        .name = "pencil-starneig-blacs",
        .desc = "BLACS compatible StarNEig library style distributed matrix "
        "pencil data format",
        .distributed = 1
    }
#endif
};

///
/// @brief Returns a data format that matches a given data format label.
///
/// @param[in] label  data format label
///
/// @return matching data format if one exists; NULL otherwise
///
static struct data_format const * get_data_format(hook_data_format_t label)
{
    for (int i = 0; i < sizeof(data_formats)/sizeof(data_formats[0]); i++)
        if (data_formats[i].type == label)
            return &data_formats[i];
    return NULL;
}

///
/// @brief Converts a data format label to a string.
///
/// @param[in] format  data format label
///
/// @return matching string
///
static char const * data_format_label_to_str(hook_data_format_t label)
{
    const struct data_format *format = get_data_format(label);
    if (format != NULL)
        return format->name;
    return "invalid";
}

///
/// @brief Converts a string to a data format label.
///
/// @param[in] str  string
///
/// @return matching data format label
///
static hook_data_format_t str_to_data_format_label(char const *str)
{
    for (int i = 0; i < sizeof(data_formats)/sizeof(data_formats[0]); i++)
        if (strcmp(str, data_formats[i].name) == 0)
            return data_formats[i].type;

    return HOOK_DATA_FORMAT_INVALID;
}

///
/// @brief Checks whether a list of data format labels contains a given value.
///
/// @param[in] format  given data format label
/// @param[in] list    data format label list
///
/// @return non-zero if the test returns a true value, zero otherwise
///
static int list_contains_data_format_label(
    hook_data_format_t label, hook_data_format_t * const list)
{
    if (list == NULL)
        return 0;

    for (hook_data_format_t *i = list; *i != 0; i++)
        if (*i == label)
            return 1;

    return 0;
}

///
/// @brief Prints the contents of a data format label list.
///
/// @param[in] list  data format list
///
static void print_data_format_list(hook_data_format_t * const list)
{
    for (hook_data_format_t *i = list; *i != 0; i++)
        printf(" %s", data_format_label_to_str(*i));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static void free_hook_data_env(struct hook_data_env *env)
{
    if (env == NULL)
        return;

    if (env->free_data != NULL && env->data != NULL)
        env->free_data(env->data);

    free(env);
}

static struct hook_data_env * copy_hook_data_env(
    struct hook_data_env const *env)
{
    if (env == NULL)
        return NULL;

    struct hook_data_env *copy =
        malloc(sizeof(struct hook_data_env));

    copy->format = env->format;
    copy->copy_data = env->copy_data;
    copy->free_data = env->free_data;

    copy->data = NULL;
    if (env->data != NULL)
        copy->data = env->copy_data(env->data);

    return copy;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Data converters.
///
static struct hook_data_converter const * converters[] = {
#ifdef STARNEIG_ENABLE_MPI
    &local_starneig_converter,
    &starneig_local_converter,
#endif
#ifdef STARNEIG_ENABLE_BLACS
    &local_blacs_converter,
    &blacs_local_converter,
    &starneig_blacs_converter,
    &blacs_starneig_converter,
#endif
    0 };

///
/// @brief Returns a matching data converter.
///
/// @param[in] from  source data format label
/// @param[in] to    destination data format label
///
/// @return matching data format converter if one exists, NULL otherwise
///
static struct hook_data_converter const * get_converter(
    hook_data_format_t from, hook_data_format_t to)
{
    for (struct hook_data_converter const **i = converters; *i != NULL; i++)
        if ((*i)->from == from && (*i)->to == to)
            return *i;

    return NULL;
}

///
/// @brief Converts a data envelope to a desired data format.
///
/// @param[in]     argc    command line argument count
/// @param[in]     argv    command line arguments
/// @param[in]     format  data format label
/// @param[in,out] env     data envelope
///
/// @return zero if the conversion was successful, non-zero otherwise
///
static int convert(int argc, char * const *argv,
    hook_data_format_t label, struct hook_data_env *env)
{
    if (env->format != label) {
        struct hook_data_converter const *converter =
            get_converter(env->format, label);
        if (converter == NULL)
            return -1;

        printf("CONVERT %s -> %s ...\n",
            data_format_label_to_str(env->format),
            data_format_label_to_str(label));
        return converter->convert(argc, argv, env);
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Data type for a hook list element.
///
struct hook_list_elem_t {
    hook_mode_t mode;              ///< hook mode
    struct hook_t const *hook;     ///< hook
    struct hook_list_elem_t *next; ///< next element
};

///
/// @brief Data type for a hook list.
///
struct hook_list_t {
    struct hook_list_elem_t *head; ///< head of the list
    struct hook_list_elem_t *tail; ///< tail of the list
};

///
/// @brief Frees a hook list.
///
/// @param[in,out] list  hook list
///
static void free_hook_list(struct hook_list_t *list)
{
    if (list == NULL)
        return;

    struct hook_list_elem_t *i = list->head;
    while (i != NULL) {
        struct hook_list_elem_t *next = i->next;
        free(i);
        i = next;
    }

    free(list);
}

///
/// @brief Creates a hook list element.
///
/// @param[in] mode  hook mode
/// @param[in] hook  hook
///
/// @return matching hook list element
///
static struct hook_list_elem_t * create_hook_list_elem(
    hook_mode_t mode, struct hook_t const *hook)
{
    struct hook_list_elem_t *elem = malloc(sizeof(struct hook_list_elem_t));
    elem->mode = mode;
    elem->hook = hook;
    elem->next = NULL;
    return elem;
}

///
/// @brief Adds a hook list element to the end of a hook list.
///
/// @param[in,out] elem  hook list element
/// @param[in,out] list  hook list
///
static void add_to_hook_list(
    struct hook_list_elem_t *elem, struct hook_list_t *list)
{
    assert(elem != NULL && list != NULL);

    // link list
    if (list->head == NULL) {
        // empty list
        list->head = elem;
        list->tail = elem;
    }
    else {
        // add the new element to the end of the list
        list->tail->next = elem;
        list->tail = elem;
    }
}

///
/// @brief Converts a hook mode to a C string.
///
/// @param[in] mode  hook mode
///
/// @return matching C string
///
const char * mode_to_str(hook_mode_t mode)
{
    switch (mode) {
        case HOOK_MODE_NORMAL:
            return "normal";
        case HOOK_MODE_STRICT:
            return "strict";
        case HOOK_MODE_ALL:
            return "all";
        default:
            return "unknown";
    }
}

///
/// @brief Reads a hook and it's mode from a command line argument string.
///
/// @param[in] value   hook's name and mode in a C string format
/// @param[in] descrs  NULL pointer terminated array of hook descriptors
///
/// @return a matching hook list element if the string is valid; NULL otherwise
///
static struct hook_list_elem_t * read_hook_from_argv(
    char const *value, struct hook_descr_t const **descrs)
{
    char *param_cpy = NULL;

    // extract name and mode
    param_cpy = malloc((strlen(value)+1)*sizeof(char));
    strcpy(param_cpy, value);
    char const *name = strtok(param_cpy, ":");
    char const *mode_str = strtok(NULL, ":");

    // we should have nothing left
    if (strtok(NULL, ":") != NULL)
        goto failure;

    // find a matching hook descriptor
    struct hook_descr_t const *descr = NULL;
    {
        for (struct hook_descr_t const **i = descrs; *i != NULL; i++) {
            if (strcmp((*i)->hook->name, name) == 0) {
                descr = *i;
                break;
            }
        }
        if (descr == NULL)
            goto failure;
    }

    // find a matching mode
    hook_mode_t mode;
    {
        if (mode_str == NULL)
            mode = descr->default_mode; // default mode
        else if (strcmp(mode_str, "normal") == 0)
            mode = HOOK_MODE_NORMAL;
        else if (strcmp(mode_str, "strict") == 0)
            mode = HOOK_MODE_STRICT;
        else if (strcmp(mode_str, "all") == 0)
            mode = HOOK_MODE_ALL;
        else
            goto failure;
    }

    free(param_cpy);

    return create_hook_list_elem(mode, descr->hook);

failure:

    free(param_cpy);
    fprintf(stderr, "Invalid hook parameter %s.\n", value);

    return NULL;
}

///
/// @brief Reads hooks and hook modes from the command line arguments.
///
/// @param[in] descrs    NULL pointer terminated array of hook descriptors
/// @param[in] name      command line argument name
/// @param[in] argc      command line argument count
/// @param[in] argv      command line arguments
/// @param[in,out] argr  array that tracks which command line arguments have
///                      been processed
///
/// @return a hook list if the command line arguments are valid; NULL otherwise
///
static struct hook_list_t * read_hooks(
    struct hook_descr_t const **descrs,
    char const *name, int argc, char * const *argv, int *argr)
{
    // set up an empty list
    struct hook_list_t *list =
        malloc(sizeof(struct hook_list_t));
    list->head = list->tail = NULL;

    // locate hooks from command line arguments
    int begin = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(name, argv[i]) == 0) {
            begin = i+1;
            if (argr != NULL)
                argr[i]++;
            break;
        }
    }

    //
    // form hook list
    //

    if (0 < begin) {
        // process hooks that are given as command line arguments
        for (int i = begin; i < argc && strncmp(argv[i], "--", 2) != 0; i++) {
            struct hook_list_elem_t *elem =
                read_hook_from_argv(argv[i], descrs);
            if (elem == NULL) {
                free_hook_list(list);
                return NULL;
            }
            add_to_hook_list(elem, list);
            if (argr != NULL)
                argr[i]++;
        }
    }
    else {
        // process default hooks
        for (struct hook_descr_t const **i = descrs; *i != NULL; i++) {
            if ((*i)->is_enabled)
                add_to_hook_list(create_hook_list_elem(
                    (*i)->default_mode, (*i)->hook), list);
        }
    }

    return list;
}

///
/// @brief Check whether all hooks in a hook list support a given data format
/// directly.
///
/// @param[in] label  data format label
/// @param[in] hooks  hook list
///
/// @return non-zero if the data format is supported, 0 otherwise
///
static int hooks_support_data_format_directly(
    hook_data_format_t label, struct hook_list_t const *hooks)
{
    for (struct hook_list_elem_t *i = hooks->head; i != NULL; i = i->next)
        if (!list_contains_data_format_label(label, i->hook->formats))
            return 0;
    return 1;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Hook trigger return value enumerator.
///
///  "Delayed reporting" records the failure but the experiment is is allowed to
///  continue; failures are reported in the end. An "immediate fail" ends the
///  experiment immediately.
///
typedef enum {
    TRIGGER_SUCCESS,        ///< trigger was processed successfully
    TRIGGER_WARNING,        ///< trigger caused a warning
    TRIGGER_DELAYED_REPORT, ///< trigger requested delayed reporting
    TRIGGER_IMMEDIATE_FAIL  ///< trigger caused an immediate fail
} trigger_return_t;

///
/// @brief Data type for a hook container.
///
///  Hook's mode and state are stored to a special container.
///
struct hook_container_t {
    hook_mode_t mode;              ///< hook mode
    hook_state_t state;            ///< hook state
    struct hook_t const *hook;     ///< hook
    struct hook_container_t *next; ///< next container
};

///
/// @brief Data type for a hook container list.
///
struct hook_container_list_t {
    struct hook_container_t *head; ///< head of the list
    struct hook_container_t *tail; ///< tail of the list
};

///
/// @brief Data type for a hook trigger.
///
/// @param[in]     i          iteration counter
/// @param[in,out] container  hook container
/// @param[in,out] env        data envelope (NULL if called outside the
///                           experiment loop)
///
/// @return 0 if the hook was processed successfully, non-zero otherwise
///
typedef hook_return_t (*trigger_t)(
    int iter, struct hook_container_t *container, struct hook_data_env *env);

///
/// @brief Trigger for a after_data_init hook handle.
///
static hook_return_t trigger_after_data_init(
    int iter, struct hook_container_t *container, struct hook_data_env *env)
{
    if (container->hook->after_data_init)
        return container->hook->after_data_init(iter, container->state, env);
    return HOOK_SUCCESS;
}

///
/// @brief Trigger for before_solver_run hook handle.
///
static hook_return_t trigger_before_solver_run(
    int iter, struct hook_container_t *container, struct hook_data_env *env)
{
    if (container->hook->before_solver_run)
        return container->hook->before_solver_run(iter, container->state, env);
    return HOOK_SUCCESS;
}

///
/// @brief Trigger for after_solver_run hook handle.
///
static hook_return_t trigger_after_solver_run(
    int iter, struct hook_container_t *container, struct hook_data_env *env)
{
    if (container->hook->after_solver_run)
        return container->hook->after_solver_run(iter, container->state, env);
    return HOOK_SUCCESS;
}

///
/// @brief Trigger for summary hook handle.
///
static hook_return_t trigger_summary(
    int iter, struct hook_container_t *container, struct hook_data_env *env)
{
    if (container->hook->summary)
        return container->hook->summary(iter, container->state, env);
    return HOOK_SUCCESS;
}

///
/// @brief Creates a hook container.
///
/// @param[in] mode   hook mode
/// @param[in] state  hook state
/// @param[in] hook   hook
///
/// @return matching hook container
///
static struct hook_container_t * create_hook_container(
    hook_mode_t mode, hook_state_t state, struct hook_t const *hook)
{
    struct hook_container_t *container =
        malloc(sizeof(struct hook_container_t));
    container->mode = mode;
    container->state = state;
    container->hook = hook;
    container->next = NULL;
    return container;
}

///
/// @brief Adds a hook container to the end of a hook container list.
///
/// @param[in,out] container  hook container
/// @param[in,out] list       hook container list
///
static void add_to_hook_container_list(
    struct hook_container_t *container, struct hook_container_list_t *list)
{
    assert(container != NULL && list != NULL);

    // link list
    if (list->head == NULL) {
        // empty list
        list->head = container;
        list->tail = container;
    }
    else {
        // add the new container to the end of the list
        list->tail->next = container;
        list->tail = container;
    }
}

///
/// @brief Frees and cleanups a hook container list.
///
/// @param[in,out] list  hook container list
///
/// @return 0 if the function call was successful, non-zero otherwise
///
static int free_hook_container_list(struct hook_container_list_t *list)
{
    if (list == NULL)
        return 0;

    int failures = 0;

    struct hook_container_t *i = list->head;
    while (i != NULL) {
        struct hook_container_t *next = i->next;
        if (i->hook->clean != NULL) {
            int ret = i->hook->clean(i->state);
            if (ret) {
                fprintf(stderr,
                    "Error while cleaning hook %s state.\n", i->hook->name);
                failures++;
            }
        }
        free(i);
        i = next;
    }

    free(list);

    return failures;
}

///
/// @brief Initializes hooks.
///
/// @param[in] hooks   hook list
/// @param[in] argc    command line argument count
/// @param[in] argv    command line arguments
/// @param[in] repeat  regular repetition count
/// @param[in] warmup  warmup repetition count
///
/// @return a hook container list if the arguments are valid; NULL otherwise
///
static struct hook_container_list_t * init_hooks(
    struct hook_list_t const *hooks,
    int argc, char * const *argv, int repeat, int warmup)
{
    if (hooks == NULL)
        return NULL;

    // set up an empty list
    struct hook_container_list_t *list =
        malloc(sizeof(struct hook_container_list_t));
    list->head = list->tail = NULL;

    // process available hooks
    for (struct hook_list_elem_t *i = hooks->head; i != NULL; i = i->next) {

        // initialize state if necessary
        hook_state_t state = NULL;
        if (i->hook->init != NULL) {
            int ret = i->hook->init(argc, argv, repeat, warmup, &state);
            if (ret) {
                fprintf(stderr, "Error while initializing hook %s state.\n",
                    i->hook->name);
                free_hook_container_list(list);
                list = NULL;
                break;
            }
        }

        // create a hook container and add it to the list
        add_to_hook_container_list(create_hook_container(
            i->mode, state, i->hook), list);
    }

    return list;
}

///
/// @brief Triggers/executes hooks.
///
/// @param[in]     i           iteration counter
/// @param[in]     trigger     hook trigger
/// @param[in,out] containers  hook container list
/// @param[in,out] env         data envelope (NULL if called outside the
///                            experiment loop)
///
/// @return trigger return value
///
static trigger_return_t trigger_hook(int iter, trigger_t trigger,
    struct hook_container_list_t *list, struct hook_data_env *env)
{
    if (list == NULL)
        return TRIGGER_SUCCESS;

    int soft_fails = 0;
    int warnings = 0;

    for (struct hook_container_t *i = list->head; i != NULL; i = i->next) {

        hook_return_t hook_ret = trigger(iter, i, env);

        switch (hook_ret) {
            case HOOK_HARD_FAIL:
                return TRIGGER_IMMEDIATE_FAIL;
            case HOOK_SOFT_FAIL:
                soft_fails++;
                break;
            case HOOK_WARNING:
                warnings++;
                break;
            default:
                break;
        }
    }

    if (0 < soft_fails)
        return TRIGGER_DELAYED_REPORT;
    if (0 < warnings)
        return TRIGGER_WARNING;

    return TRIGGER_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Check whether all hook supplementers support a given data format
/// directly.
///
/// @param[in] label          data format label
/// @param[in] supplementers  hook supplementers
///
/// @return non-zero if the data format is supported, 0 otherwise
///
static int supplementers_support_data_format_directly(
    hook_data_format_t label, struct hook_supplementer_t const **supplementers)
{
    struct hook_supplementer_t const **iter = supplementers;
    while (*iter != NULL) {
        if (!list_contains_data_format_label(label, (*iter)->formats))
            return 0;
        iter++;
    }
    return 1;
}

///
/// @brief Finds a valid data format combination that is directly supported by a
/// given initializer, a given set of supplementers and a given set of hooks.
///
/// @param[in]  solver_format  preset solver data format label
/// @param[in]  initializer    hook initializer
/// @param[in]  supplementers  hook supplementers
/// @param[in]  hooks          hook list
/// @param[out] init_format    returns initializer data format label
/// @param[out] hook_format    returns hook data format label
///
/// @return 0 if a valid data format combination was found, non-zero otherwise
///
static int get_valid_direct_data_formats(
    hook_data_format_t solver_format,
    struct hook_initializer_t const *initializer,
    struct hook_supplementer_t const **supplementers,
    struct hook_list_t const *hooks,
    hook_data_format_t *init_format,
    hook_data_format_t *hook_format)
{
    // if the initializer, supplementers and hooks support the solver format
    // directly
    if (list_contains_data_format_label(solver_format, initializer->formats) &&
    supplementers_support_data_format_directly(solver_format, supplementers) &&
    hooks_support_data_format_directly(solver_format, hooks)) {

        *init_format = solver_format;
        *hook_format = solver_format;
        return 0;
    }

    *init_format = HOOK_DATA_FORMAT_INVALID;
    *hook_format = HOOK_DATA_FORMAT_INVALID;

    return -1;
}

///
/// @brief Finds a valid data format combination that is supported by a given
/// initializer, a given set of supplementers and a given set of hooks.
///
/// @param[in]  solver_format  preset solver data format label
/// @param[in]  initializer    hook initializer
/// @param[in]  supplementers  hook supplementers
/// @param[in]  hooks          hook list
/// @param[out] init_format    returns initializer data format label
/// @param[out] hook_format    returns hook data format label
///
/// @return 0 if a valid data format combination was found, non-zero otherwise
///
static int get_valid_data_formats(
    hook_data_format_t solver_format,
    struct hook_initializer_t const *initializer,
    struct hook_supplementer_t const **supplementers,
    struct hook_list_t const *hooks,
    hook_data_format_t *init_format,
    hook_data_format_t *hook_format)
{
    int mpi = 0;
#ifdef STARNEIG_ENABLE_MPI
    if (get_data_format(solver_format)->distributed)
        MPI_Initialized(&mpi);
#endif

    // if the initializer, supplementers and hooks support the solver format
    // directly
    if (list_contains_data_format_label(solver_format, initializer->formats) &&
    supplementers_support_data_format_directly(solver_format, supplementers) &&
    hooks_support_data_format_directly(solver_format, hooks)) {

        *init_format = solver_format;
        *hook_format = solver_format;
        return 0;
    }

    // otherwise, i loops over distributed data formats that are supported by
    // the initializer
    for (hook_data_format_t *i = initializer->formats; mpi && *i != 0; i++) {
        if (get_data_format(*i)->distributed &&
        supplementers_support_data_format_directly(*i, supplementers)) {

            // if i can be converted to solver_format and hooks support i
            // directly
            if (get_converter(*i, solver_format) != NULL &&
            hooks_support_data_format_directly(*i, hooks)) {

                *init_format = *i;
                *hook_format = *i;
                return 0;
            }

            // if i can be converted to solver_format and hooks support
            // solver_format directly
            if (get_converter(*i, solver_format) != NULL &&
            hooks_support_data_format_directly(solver_format, hooks)) {

                *init_format = *i;
                *hook_format = solver_format;
                return 0;
            }
        }
    }

    // otherwise, i loops over data formats that are supported by the
    // initializer
    for (hook_data_format_t *i = initializer->formats; *i != 0; i++) {
        if (supplementers_support_data_format_directly(*i, supplementers)) {

            // if i can be converted to solver_format and hooks support i
            // directly
            if (get_converter(*i, solver_format) != NULL &&
            hooks_support_data_format_directly(*i, hooks)) {

                *init_format = *i;
                *hook_format = *i;
                return 0;
            }

            // if i can be converted to solver_format and hooks support
            // solver_format directly
            if (get_converter(*i, solver_format) != NULL &&
            hooks_support_data_format_directly(solver_format, hooks)) {

                *init_format = *i;
                *hook_format = solver_format;
                return 0;
            }
        }
    }

    *init_format = HOOK_DATA_FORMAT_INVALID;
    *hook_format = HOOK_DATA_FORMAT_INVALID;

    return -1;
}

///
/// @brief Finds a valid data format combination that is supported by a given
/// initializer, a given set of supplementers and a given solver and a given set
// of hooks.
///
/// @param[in]  initializer    hook initializer
/// @param[in]  supplementers  hook supplementers
/// @param[in]  solver         hook solver
/// @param[in]  hooks          hook list
/// @param[out] init_format    returns initializer data format label
/// @param[out] solver_format  returns solver data format label
/// @param[out] hook_format    returns a hook data format label
///
/// @return 0 if a valid data format combination was found, non-zero otherwise
///
static int get_default_data_formats(
    struct hook_initializer_t const *initializer,
    struct hook_supplementer_t const **supplementers,
    struct hook_solver const *solver,
    struct hook_list_t const *hooks,
    hook_data_format_t *init_format,
    hook_data_format_t *solver_format,
    hook_data_format_t *hook_format)
{
    int mpi = 0;
#ifdef STARNEIG_ENABLE_MPI
    MPI_Initialized(&mpi);
#endif

    // i loops over data formats that are supported by the solver and look for
    // a one that is directly supported
    for (hook_data_format_t *i = solver->formats; *i != 0; i++) {

        // if MPI is enabled, then only distributed formats should be considered
        if (!mpi || (mpi && get_data_format(*i)->distributed)) {
            int ret = get_valid_direct_data_formats(*i, initializer,
                supplementers, hooks, init_format, hook_format);
            if (ret == 0) {
                *solver_format = *i;
                return 0;
            }
        }
    }

    // i loops over data formats that are supported by the solver
    for (hook_data_format_t *i = solver->formats; *i != 0; i++) {

        // if MPI is enabled, then only distributed formats should be considered
        if (!mpi || (mpi && get_data_format(*i)->distributed)) {
            int ret = get_valid_data_formats(*i, initializer, supplementers,
                hooks, init_format, hook_format);
            if (ret == 0) {
                *solver_format = *i;
                return 0;
            }
        }
    }

    *init_format = HOOK_DATA_FORMAT_INVALID;
    *solver_format = HOOK_DATA_FORMAT_INVALID;
    *hook_format = HOOK_DATA_FORMAT_INVALID;

    return -1;
}

///
/// @brief Reads data format from the command line arguments.
///
/// @param[in] name      command line argument name
/// @param[in] argc      command line argument count
/// @param[in] argv      command line arguments
/// @param[in,out] argr  an array that tracks which command line arguments have
///                      been processed
/// @param[in] def       default data format
///
/// @return matching data format label
///
static hook_data_format_t read_format(
    char const *name, int argc, char * const *argv, int *argr,
    hook_data_format_t def)
{
    return str_to_data_format_label(
        read_str(name, argc, argv, argr, data_format_label_to_str(def)));
}

///
/// @brief Prints available data formats.
///
static void print_avail_formats(int argc, char * const *argv)
{
    printf("\nAvailable data formats:\n");
    for (int i = 0; i < sizeof(data_formats)/sizeof(data_formats[0]); i++)
        printf("    %s : %s\n", data_formats[i].name, data_formats[i].desc);
}

///
/// @brief Prints available data converters.
///
static void print_avail_converters(int argc, char * const *argv)
{
    printf("\n========== Available data converters ==========\n");
    for (struct hook_data_converter const **i = converters; *i != NULL; i++) {
        printf("\n%s -> %s\n",
            data_format_label_to_str((*i)->from),
            data_format_label_to_str((*i)->to));

        if ((*i)->print_usage != NULL)
            (*i)->print_usage(argc, argv);
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Prints available hooks.
///
/// @param[in] descrs  NULL pointer terminated array of hook descriptors
/// @param[in] argc    command line argument count
/// @param[in] argv    command line arguments
///
static void print_avail_hooks(
    struct hook_descr_t const **descrs, int argc, char * const *argv)
{
    if (*descrs == NULL)
        return;

    printf("\n========== Available hooks ==========\n");
    for (struct hook_descr_t const **i = descrs; *i != NULL; i++) {
        const char *str = (*i)->is_enabled ?
            "\n[%s:%s] : %s" : "\n'%s:%s' : %s";
        printf(str, (*i)->hook->name,
            mode_to_str((*i)->default_mode), (*i)->hook->desc);
        printf(" (in:");
        print_data_format_list((*i)->hook->formats);
        printf(")\n");

        if ((*i)->hook->print_usage != NULL)
            (*i)->hook->print_usage(argc, argv);
    }
}

///
/// @brief Prints hook related command line arguments.
///
/// @param[in] hooks  hook list
/// @param[in] argc   command line argument count
/// @param[in] argv   command line arguments
///
static void print_args_hooks(
   struct hook_list_t const *hooks, int argc, char * const *argv)
{
    if (hooks == NULL)
        return;

    // print active hooks
    printf(" --hooks");
    for (struct hook_list_elem_t *i = hooks->head; i != NULL; i = i->next)
        printf(" %s:%s", i->hook->name, mode_to_str(i->mode));

    // print individual hook arguments
    for (struct hook_list_elem_t *i = hooks->head; i != NULL; i = i->next)
        if (i->hook->print_args)
            i->hook->print_args(argc, argv);
}

///
/// @brief Checks hook related command line arguments.
///
/// @param[in]     hooks  hook list
/// @param[in]     argc   command line argument count
/// @param[in]     argv   command line arguments
/// @param[in,out] argr   array that tracks which command line arguments
///                       have been processed
///
/// @return 0 if the command line arguments are valid, non-zero otherwise
///
static int check_args_hooks(
    struct hook_list_t const *hooks, int argc, char * const *argv, int *argr)
{
    int failures = 0;

    if (hooks == NULL)
        return 0;

    for (struct hook_list_elem_t *i = hooks->head; i != NULL; i = i->next)
        if (i->hook->check_args && i->hook->check_args(argc, argv, argr))
            failures++;

    return failures;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Reads a hook initializer from the command line.
///
/// @param[in] initializers  NULL pointer terminated array of hook initializers
/// @param[in] name          argument name
/// @param[in] argc          command line argument count
/// @param[in] argv          command line arguments
/// @param[in,out] argr      array that tracks which command line arguments have
///                          been processed
///
/// @return a pointer to the requested hook initializer
///
static struct hook_initializer_t const * read_initializer(
    struct hook_initializer_t const **initializers, char const *name,
    int argc, char * const *argv, int *argr)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            struct hook_initializer_t const **j = initializers;
            while (*j != NULL) {
                if (strcmp((*j)->name, argv[i+1]) == 0) {
                    if (argr != NULL)
                        argr[i] = argr[i+1] = 1;
                    return *j;
                }
                j++;
            }
            return NULL;
        }
    }
    return initializers[0];
}

///
/// @brief Prints available hook initializers.
///
/// @param[in] initializers  NULL pointer terminated array of hook initializers
/// @param[in] argc          command line argument count
/// @param[in] argv          command line arguments
///
static void print_avail_initializers(
    struct hook_initializer_t const **initializers,
    int argc, char * const *argv)
{
    struct hook_initializer_t const *initializer =
        read_initializer(initializers, "--init", argc, argv, NULL);

    if (initializer == NULL)
        return;

    printf("\n========== Available initializer modules ==========\n");
    struct hook_initializer_t const **i = initializers;
    while (*i != NULL) {
        const char *str = *i == initializer ?
            "\n[%s] : %s" : "\n'%s' : %s";
        printf(str, (*i)->name, (*i)->desc);
        printf(" (out:");
        print_data_format_list((*i)->formats);
        printf(")\n");

        if ((*i)->print_usage != NULL)
            (*i)->print_usage(argc, argv);

        i++;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// @brief Reads a hook solver from the command line.
///
/// @param[in] solvers   NULL pointer terminated array of hook solvers
/// @param[in] name      argument name
/// @param[in] argc      command line argument count
/// @param[in] argv      command line arguments
/// @param[in,out] argr  array that tracks which command line arguments have
///                      been processed
///
/// @return a pointer to the requested hook solver
///
static struct hook_solver const * read_solver(
    struct hook_solver const **solvers, char const *name,
    int argc, char * const *argv, int *argr)
{
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(name, argv[i]) == 0) {
            struct hook_solver const **j = solvers;
            while (*j != NULL) {
                if (strcmp((*j)->name, argv[i+1]) == 0) {
                    if (argr != NULL)
                        argr[i] = argr[i+1] = 1;
                    return *j;
                }
                j++;
            }
            return NULL;
        }
    }
    return solvers[0];
}

///
/// @brief Prints available hook solvers.
///
/// @param[in] solvers  NULL pointer terminated array of hook solvers
/// @param[in] argc     command line argument count
/// @param[in] argv     command line arguments
///
static void print_avail_solvers(
    struct hook_solver const **solvers, int argc, char * const *argv)
{
    struct hook_solver const *solver =
        read_solver(solvers, "--solver", argc, argv, NULL);

    if (solver == NULL)
        return;

    printf("\n========== Available solvers ==========\n");
    struct hook_solver const **i = solvers;
    while (*i != NULL) {
        const char *str = *i == solver ?
            "\n[%s] : %s" : "\n'%s' : %s";
        printf(str, (*i)->name, (*i)->desc);
        printf(" (in:"); print_data_format_list((*i)->formats); printf(")");
        printf("\n");

        if ((*i)->print_usage != NULL)
            (*i)->print_usage(argc, argv);

        i++;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void hook_experiment_print_usage(
    int argc, char * const *argv, const experiment_info_t info)
{
    struct hook_experiment_descr const *descr = info;

    printf(
        "  --data-format (format) -- Data format\n"
        "  --init (initializer) -- Initialization module\n"
        "  --solver (solver) -- Solver module\n"
        "  --hooks (hook:mode, ...) -- Hooks and modes\n"
        "  --no-reinit -- Do not reinitialize after each repetition\n"
        "  --repeat (num) -- Repeated experiment\n"
        "  --warmup (num) -- Perform \"warmups\"\n"
        "  --keep-going -- Try to recover from a solver failure\n"
        "  --abort -- Call abort() in failure\n"
    );

    print_avail_formats(argc, argv);
    print_avail_converters(argc, argv);
    print_avail_initializers(descr->initializers, argc, argv);
    print_avail_solvers(descr->solvers, argc, argv);
    print_avail_hooks(descr->hook_descrs, argc, argv);

    if (*descr->supplementers != NULL) {
        printf("\n========== Supplementer options ==========\n");
        struct hook_supplementer_t const **iter = descr->supplementers;
        while (*iter != NULL) {
            if ((*iter)->print_usage) {
                printf("\n%s supplementer specific options:\n", (*iter)->name);
                (*iter)->print_usage(argc, argv);
            }
            iter++;
        }
    }

    if (descr->print_usage != NULL) {
        printf("\n========== Other options ==========\n");
        descr->print_usage(argc, argv);
    }

    printf("\n");
}

void hook_experiment_print_args(
    int argc, char * const *argv, const experiment_info_t info)
{
    struct hook_experiment_descr const *descr = info;

    struct hook_solver const *solver =
        read_solver(descr->solvers, "--solver", argc, argv, NULL);

    struct hook_initializer_t const *initializer =
        read_initializer(descr->initializers, "--init", argc, argv, NULL);

    struct hook_list_t *hooks =
        read_hooks(descr->hook_descrs, "--hooks", argc, argv, NULL);

    // get default data formats
    hook_data_format_t init_format, solver_format, hook_format;
    get_default_data_formats(initializer, descr->supplementers, solver, hooks,
        &init_format, &solver_format, &hook_format);

    // read data formats from the command line
    solver_format =
        read_format("--data-format", argc, argv, NULL, solver_format);
    get_valid_data_formats(solver_format, initializer, descr->supplementers,
        hooks, &init_format, &hook_format);

    printf(" --data-format %s", data_format_label_to_str(solver_format));

    if (init_format != hook_format) {
        struct hook_data_converter const *converter =
            get_converter(init_format, hook_format);
        if (converter->print_args != NULL)
            converter->print_args(argc, argv);
    }

    if (hook_format != solver_format) {
        struct hook_data_converter const *converter =
            get_converter(hook_format, solver_format);
        if (converter->print_args != NULL)
            converter->print_args(argc, argv);
    }

    if (solver_format != hook_format) {
        struct hook_data_converter const *converter =
            get_converter(solver_format, hook_format);
        if (converter->print_args != NULL)
            converter->print_args(argc, argv);
    }

    printf(" --init %s", initializer->name);

    if (initializer->print_args != NULL)
        initializer->print_args(argc, argv);

    if (*descr->supplementers != NULL) {
        struct hook_supplementer_t const **iter = descr->supplementers;
        while (*iter != NULL) {
            if ((*iter)->print_args)
                (*iter)->print_args(argc, argv);
            iter++;
        }
    }

    printf(" --solver %s", solver->name);

    if (solver->print_args != NULL)
        solver->print_args(argc, argv);

    print_args_hooks(hooks, argc, argv);

    int repeat = read_int("--repeat", argc, argv, NULL, 1);
    int warmup = read_int("--warmup", argc, argv, NULL, 0);

    // arguments should be printed only when they have an effect

    if (1 < repeat + warmup && read_opt("--no-reinit", argc, argv, NULL))
        printf(" --no-reinit");

    printf(" --repeat %d --warmup %d", repeat, warmup);

    if (read_opt("--keep-going", argc, argv, NULL))
        printf(" --keep-going");

    if (read_opt("--abort", argc, argv, NULL))
        printf(" --abort");

    free_hook_list(hooks);
}


int hook_experiment_check_args(
    int argc, char * const *argv, int *argr, const experiment_info_t info)
{
    struct hook_experiment_descr const *descr = info;

    int ret;
    struct hook_list_t *hooks = NULL;

    struct hook_initializer_t const *initializer =
        read_initializer(descr->initializers, "--init", argc, argv, argr);
    if (initializer == NULL) {
        fprintf(stderr, "Invalid initialization module.\n");
        ret = -1; goto cleanup;
    }

    if (initializer->check_args != NULL) {
        ret = initializer->check_args(argc, argv, argr);
        if (ret) goto cleanup;
    }

    if (*descr->supplementers != NULL) {
        struct hook_supplementer_t const **iter = descr->supplementers;
        while (*iter != NULL) {
            if ((*iter)->check_args) {
                ret = (*iter)->check_args(argc, argv, argr);
                if (ret) goto cleanup;
            }
            iter++;
        }
    }


    struct hook_solver const *solver =
        read_solver(descr->solvers, "--solver", argc, argv, argr);
    if (solver == NULL) {
        fprintf(stderr, "Invalid solver module.\n");
        ret = -1; goto cleanup;
    }

    if (solver->check_args != NULL) {
        ret = solver->check_args(argc, argv, argr);
        if (ret) goto cleanup;
    }

    hooks = read_hooks(descr->hook_descrs, "--hooks", argc, argv, argr);
    if (hooks == NULL) {
        fprintf(stderr, "Invalid hook list.\n");
        ret = -1; goto cleanup;
    }

    // get default data formats
    hook_data_format_t init_format, solver_format, hook_format;
    ret = get_default_data_formats(initializer, descr->supplementers,
        solver, hooks, &init_format, &solver_format, &hook_format);

    if (ret) {
        fprintf(stderr, "Cannot find compatible data formats.\n");
        ret = -1; goto cleanup;
    }

    // read data formats from the command line
    solver_format =
        read_format("--data-format", argc, argv, argr, solver_format);
    ret = get_valid_data_formats(solver_format, initializer,
        descr->supplementers, hooks, &init_format, &hook_format);

    if (ret) {
        fprintf(stderr, "Cannot find compatible data formats.\n");
        ret = -1; goto cleanup;
    }

    int mpi = 0;
#ifdef STARNEIG_ENABLE_MPI
    MPI_Initialized(&mpi);
#endif

    if (!mpi && get_data_format(solver_format)->distributed) {
        fprintf(stderr,
            "Requested data format is distributed but MPI is not enabled.\n");
        ret = -1; goto cleanup;
    }

    // check converters

    if (init_format != hook_format) {
        struct hook_data_converter const *converter =
            get_converter(init_format, hook_format);
        if (converter->check_args != NULL) {
            ret = converter->check_args(argc, argv, argr);
            if (ret) goto cleanup;
        }
    }

    if (hook_format != solver_format) {
        struct hook_data_converter const *converter =
            get_converter(hook_format, solver_format);
        if (converter->check_args != NULL) {
            ret = converter->check_args(argc, argv, argr);
            if (ret) goto cleanup;
        }
    }

    if (solver_format != hook_format) {
        struct hook_data_converter const *converter =
            get_converter(solver_format, hook_format);
        if (converter->check_args != NULL) {
            ret = converter->check_args(argc, argv, argr);
            if (ret) goto cleanup;
        }
    }

    ret = check_args_hooks(hooks, argc, argv, argr);
    if (ret) goto cleanup;

    if (read_int("--repeat", argc, argv, argr, 1) < 1) {
        fprintf(stderr, "Invalid number of repetitions.\n");
        ret = -1; goto cleanup;
    }

    if (read_int("--warmup", argc, argv, argr, 1) < 0) {
        fprintf(stderr, "Invalid number of warmup repetitions.\n");
        ret = -1; goto cleanup;
    }

    read_opt("--no-reinit", argc, argv, argr);

    int keep_going = read_opt("--keep-going", argc, argv, argr);
    int _abort = read_opt("--abort", argc, argv, argr);

    if (keep_going && _abort) {
        fprintf(stderr,
            "--keep-going and --abort cannot be set simultaneously.\n");
        ret = -1; goto cleanup;
    }

cleanup:

    free_hook_list(hooks);
    return ret;
}

int hook_experiment_run(
    int argc, char * const *argv, const experiment_info_t info)
{
    int failures = 0;
    int warnings = 0;

    struct hook_experiment_descr const *descr = info;

    //
    // set relevant variables to valid initial values
    //

    struct hook_data_env *original_data = NULL;
    struct hook_data_env *data = NULL;
    struct hook_list_t *hook_list = NULL;
    struct hook_container_list_t *hooks = NULL;
    hook_solver_state_t solver_state = NULL;
    double *time = NULL;

    //
    // check MPI status
    //

    int my_rank = 0;

#ifdef STARNEIG_ENABLE_MPI
    int world_size = 1;
    {
        int mpi;
        MPI_Initialized(&mpi);

        if (mpi) {
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        }
    }
#endif

    //
    // parse arguments
    //

    struct hook_initializer_t const *initializer =
        read_initializer(descr->initializers, "--init", argc, argv, NULL);

    struct hook_solver const *solver =
        read_solver(descr->solvers, "--solver", argc, argv, NULL);

    hook_list = read_hooks(
        descr->hook_descrs, "--hooks", argc, argv, NULL);

    hook_data_format_t init_format, solver_format, hook_format;
    get_default_data_formats(initializer, descr->supplementers, solver,
        hook_list, &init_format, &solver_format, &hook_format);

    solver_format =
        read_format("--data-format", argc, argv, NULL, solver_format);
    get_valid_data_formats(solver_format, initializer, descr->supplementers,
        hook_list, &init_format, &hook_format);

    int repeat = read_int("--repeat", argc, argv, NULL, 1);
    int warmup = read_int("--warmup", argc, argv, NULL, 0);

    int reinit =
        1 < repeat + warmup && !read_opt("--no-reinit", argc, argv, NULL);

    int keep_going = read_opt("--keep-going", argc, argv, NULL);
    int _abort = read_opt("--abort", argc, argv, NULL);

    //
    // initialize hooks
    //

    if (my_rank == 0 || get_data_format(hook_format)->distributed) {
        hooks = init_hooks(hook_list, argc, argv, repeat, warmup);

        if (hooks == NULL) {
            fprintf(stderr, "Error while initializing hooks.\n");
            failures++;
            if (_abort)
                abort();
            goto cleanup;
        }
    }

    //
    // experiment loop
    //

    time = malloc(repeat*sizeof(double));

    for (int i = -warmup; i < repeat; i++) {

        if (0 < warmup && i == 0)
            printf(
                "================================"
                "================================\n");

        if (i < 0)
            printf("WARMUP %d / %d ...\n", i + warmup + 1, warmup);
        else if (1 < repeat)
            printf("REPEAT %d / %d ...\n", i + 1, repeat);

        if (my_rank == 0 || get_data_format(init_format)->distributed) {

            //
            // prepare (copy or reinitialize) data
            //

            // either reinitialize before each run or this is the first run
            if (reinit || i == -warmup) {
                free_hook_data_env(data);
                data = initializer->init(init_format, argc, argv);

                if (data == NULL) {
                    fprintf(stderr, "Error while initializing data.\n");
                    failures++;
                    if (_abort)
                        abort();
                    goto cleanup;
                }

                struct hook_supplementer_t const **iter = descr->supplementers;
                while (*iter != NULL) {
                    (*iter)->supplement(data, argc, argv);
                    iter++;
                }
            }

            // no re-initialization, multiple runs
            if (!reinit && 1 < warmup + repeat) {
                if(i == -warmup) {
                    original_data = copy_hook_data_env(data);
                }
                else {
                    free_hook_data_env(data);
                    data = copy_hook_data_env(original_data);
                }
            }
        }
        else {
            // create an empty envelope
            data = malloc(sizeof(struct hook_data_env));
            memset(data, 0, sizeof(struct hook_data_env));
            data->format = init_format;
        }

        //
        // concert to hook format
        //

        if (init_format != hook_format)
            convert(argc, argv, hook_format, data);

        if (my_rank == 0 || get_data_format(hook_format)->distributed) {

            //
            // process after_data_init hooks
            //

            switch (
            trigger_hook(i, trigger_after_data_init, hooks, data)) {
                case TRIGGER_SUCCESS:
                    break;
                case TRIGGER_WARNING:
                    warnings++;
                    break;
                case TRIGGER_DELAYED_REPORT:
                    failures++;
                    if (_abort)
                        abort();
                    break;
                default:
                    fprintf(stderr, "after_data_init hook failed.\n");
                    failures++;
                    if (_abort)
                        abort();
                    goto cleanup;
            }

            //
            // process before_solver_run hooks
            //

            switch (
            trigger_hook(i, trigger_before_solver_run, hooks, data)) {
                case TRIGGER_SUCCESS:
                    break;
                case TRIGGER_WARNING:
                    warnings++;
                    break;
                case TRIGGER_DELAYED_REPORT:
                    failures++;
                    if (_abort)
                        abort();
                    break;
                default:
                    fprintf(stderr, "before_solver_run hook failed.\n");
                    failures++;
                    if (_abort)
                        abort();
                    goto cleanup;
            }
        }

        //
        // convert to solver format
        //

        if (hook_format != solver_format)
            convert(argc, argv, solver_format, data);

        //
        // prepare solver
        //

#ifdef STARNEIG_ENABLE_MPI
        if (1 < world_size)
            MPI_Barrier(MPI_COMM_WORLD);
#endif

        printf("PREPARE...\n");
        fflush(stdout);

        solver_state = solver->prepare(argc, argv, data);

        if (solver_state == NULL) {
            fprintf(stderr, "Solver prepare function failed.\n");
            failures++;
            if (_abort)
                abort();
            goto cleanup;
        }

        //
        // execute solver
        //

        {
#ifdef STARNEIG_ENABLE_MPI
            if (1 < world_size)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            printf("PROCESS...\n");
            fflush(stdout);

            struct timespec start, stop;
            clock_gettime(CLOCK_REALTIME, &start);

            int ret = solver->run(solver_state);

#ifdef STARNEIG_ENABLE_MPI
            if (1 < world_size)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            clock_gettime(CLOCK_REALTIME, &stop);

            double current_time = stop.tv_sec*1e+3+stop.tv_nsec*1e-6 -
                (start.tv_sec*1e+3+start.tv_nsec*1e-6);

            if (i < 0)
                printf("WARMUP TIME = %.0f MS\n", current_time);
            else
                printf("EXPERIMENT TIME = %.0f MS\n", current_time);

            if (0 <= i)
                time[i] = current_time;

            if (ret) {
                fprintf(stderr, "Solver exited with an error code %d.\n", ret);
                if (!keep_going) {
                    failures++;
                    if (_abort)
                        abort();
                    goto cleanup;
                }
            }
        }

        //
        // finalize solver
        //

        {
            printf("FINALIZE...\n");
            fflush(stdout);

            int ret = solver->finalize(solver_state, data);
            solver_state = NULL;

            if (ret) {
                fprintf(stderr, "Solver finalize function failed.\n");
                failures++;
                if (_abort)
                    abort();
                goto cleanup;
            }

        }

        //
        // convert to hook format
        //

        if (solver_format != hook_format)
            convert(argc, argv, hook_format, data);

        if (my_rank == 0 || get_data_format(hook_format)->distributed) {

            //
            // process after_solver_run hooks
            //

            switch (
            trigger_hook(i, trigger_after_solver_run, hooks, data)) {
                case TRIGGER_SUCCESS:
                    break;
                case TRIGGER_WARNING:
                    warnings++;
                    break;
                case TRIGGER_DELAYED_REPORT:
                    failures++;
                    if (_abort)
                        abort();
                    break;
                default:
                    fprintf(stderr, "after_solver_run hook failed.\n");
                    failures++;
                    if (_abort)
                        abort();
                    goto cleanup;
            }
        }

        //
        // post experiment loop iteration cleanup
        //

        free_hook_data_env(data);
        data = NULL;
    }


    if (my_rank == 0 || get_data_format(hook_format)->distributed) {

        //
        // print statistics
        //

        printf(
            "================================================================"
            "\n");

        qsort(time, repeat, sizeof(double), &double_compare);
        printf("TIME = %.0f MS "\
            "[avg %.0f MS, cv %.2f, min %.0f MS, max %.0f MS]\n",
            double_median(repeat, time), double_mean(repeat, time),
            double_cv(repeat, time), time[0], time[repeat-1]);

        //
        // process after_solver_run hooks
        //

        switch (trigger_hook(repeat, trigger_summary, hooks, NULL)) {
            case TRIGGER_SUCCESS:
                break;
            case TRIGGER_WARNING:
                warnings++;
                break;
            case TRIGGER_DELAYED_REPORT:
                failures++;
                if (_abort)
                    abort();
                break;
            default:
                fprintf(stderr, "summary hook failed.\n");
                failures++;
                if (_abort)
                    abort();
                goto cleanup;
        }
    }

cleanup:

    //
    // cleanup
    //

    free(time);
    free_hook_data_env(original_data);
    free_hook_data_env(data);
    free_hook_list(hook_list);
    free_hook_container_list(hooks);
    solver->finalize(solver_state, NULL);

    if (0 < warnings)
        fprintf(stderr, "%d WARNINGS ENCOUNTERED.\n", warnings);

    if (0 < failures)
        fprintf(stderr, "%d FAILURES ENCOUNTERED.\n", failures);

    return failures;
}
