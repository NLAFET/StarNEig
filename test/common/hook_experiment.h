///
/// @file
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
///
///
///
///  +----------------+
///  | experiment_t   |
///  |  * print_usage ---> hook_experiment_print_usage(..., info)
///  |  * print_args ----> hook_experiment_print_args(..., info)
///  |  * check_args ----> hook_experiment_check_args(..., info)
///  |  * run -----------> hook_experiment_run(..., info)
///  |  * info        |           ^
///  +-----|----------+           '-- hook_data_env_t, hook_state_t
///        |
///  +-----v-------------------+     +--------------------+
///  | hook_experiment_descr |  -->| hook_initializer_t |
///  |  * initializers ----------'  *+--------------------+
///  |  * solvers ---------------    +---------------+
///  |  * hooks                | '-->| hook_solver |
///  +-----|-------------------+    *+---------------+
///        |
///  +-----v*----------+
///  | hook_descr_t    |
///  |  * is_enabled   |
///  |  * default_mode |
///  |  * hooks        |
///  +-----|-----------+
///        |
///  +-----v--+
///  | hook_t |
///  +-----|--+
///        |
///  +-----v*--------+
///  | hook_handle_t |
///  +---------------+
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

#ifndef STARNEIG_TEST_COMMON_HOOK_EXPERIMENT_H
#define STARNEIG_TEST_COMMON_HOOK_EXPERIMENT_H

#include <starneig_test_config.h>
#include <starneig/configuration.h>
#include "experiment.h"

///
/// @brief Data type for generic solver input/output data.
///
typedef void * hook_data_t;

///
/// @brief Data format label enumerator.
///
typedef enum {
    HOOK_DATA_FORMAT_INVALID                 = 1, ///< invalid data format label
    HOOK_DATA_FORMAT_ANY                     = 2, ///< any data format label
    HOOK_DATA_FORMAT_GENERIC                 = 3, ///< generic data
    HOOK_DATA_FORMAT_PENCIL_LOCAL            = 4, ///< local matrix pencil
#ifdef STARNEIG_ENABLE_MPI
    HOOK_DATA_FORMAT_PENCIL_STARNEIG         = 5, ///< StarNEig pencil
    HOOK_DATA_FORMAT_PENCIL_BLACS            = 6, ///< StarNEig/BLACS pencil
#endif
} hook_data_format_t;

///
/// @brief Data type for a hook data envelope free function.
///
/// @param[in] data - data
///
typedef void (*hook_data_env_free_t)(hook_data_t data);

///
/// @brief Data type for a hook data envelope copy function.
///
/// @param[in] data - data
///
/// @return copied data
///
typedef hook_data_t (*hook_data_env_copy_t)(const hook_data_t data);

///
/// @brief Data type for a hook data envelope.
///
///  Each hook initializer should produce the data in this format. This format
///  contains information on how the data should be copied/replicated and how
///  the resources that were allocated for the data should be released.
///
struct hook_data_env {
    hook_data_env_copy_t copy_data;             ///< copy function
    hook_data_env_free_t free_data;             ///< cleanup function
    hook_data_format_t format;                  ///< data format label
    hook_data_t data;                           ///< data
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Data type for a generic hook state.
///
///  Each hook has a state that is valid throughout a whole hook_experiment_run
///  function call. A hook can provide state initialization and cleanup
///  functions.
///
typedef void * hook_state_t;

///
/// @brief Hook return value enumerator.
///
typedef enum {
    HOOK_SUCCESS,   ///< successful hook execution
    HOOK_WARNING,   ///< hook execution warning
    HOOK_SOFT_FAIL, ///< soft hook execution failure (can be recovered)
    HOOK_HARD_FAIL  ///< hard hook execution failure (cannot be recovered)
} hook_return_t;



///
/// @brief Data type for a hook handle.
///
/// @param[in] iter       iteration counter
/// @param[in,out] state  hook state
/// @param[in,out] env    hook data envelope (NULL if called outside the
///                       experiment loop)
///
/// @return hook return code
///
typedef hook_return_t (*hook_handle_t)(
    int iter, hook_state_t state, struct hook_data_env *env);

///
/// @brief Data type for a hook.
///
///  A hook consists of multiple handles. Some handles are executed during each
///  experiment loop iteration; some handles are executed only once. In a
///  typical use case, handles record the iteration specific data to the hook
///  state and separate summary handle summarizes the aggregated data.
///
struct hook_t {
    char const *name;                ///< hook name
    char const *desc;                ///< hook description
    hook_data_format_t *formats;     ///< supported data format labels

    ///
    /// @brief Hook usage information / instruction printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_usage)(int argc, char * const *argv);

    ///
    /// @brief Hook command line arguments printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Hook's command line argument validation function.
    ///
    /// @param[in] argc      command line argument count
    /// @param[in] argv      command line arguments
    /// @param[in,out] argr  array that tracks which command line arguments have
    ///                      been processed
    ///
    /// @return 0 if the arguments are valid, non-zero otherwise
    ///
    int (*check_args)(int argc, char * const *argv, int *argr);

    ///
    /// @brief Hook state initialization function.
    ///
    /// @param[in] argc    command line argument count
    /// @param[in] argv    command line arguments
    /// @param[in] repeat  regular repetition count
    /// @param[in] warmup  warmup repetition count
    /// @param[out] state  returns a valid hook state
    ///
    /// @return 0 if the initialization was successful, non-zero otherwise
    ///
    int (*init)(int argc, char * const *argv, int repeat, int warmup,
        hook_state_t *state);

    ///
    /// @brief Hook state cleanup function.
    ///
    /// @param[inout] state - hook state
    ///
    /// @return 0 if the cleanup was successful, non-zero otherwise
    ///
    int (*clean)(hook_state_t state);

    hook_handle_t after_data_init;   ///< executed after data initialization
    hook_handle_t before_solver_run; ///< executed before solver
    hook_handle_t after_solver_run;  ///< executed after solver
    hook_handle_t summary;           ///< executed in conclusion
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Data type for a hook experiment initializer.
///
struct hook_initializer_t {
    char const *name;               ///< hook initializer name
    char const *desc;               ///< hook initializer description
    hook_data_format_t *formats;    ///< supported data format labels

    ///
    /// @brief Hook initializer instruction printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_usage)(int argc, char * const *argv);

    ///
    /// @brief Hook initializer command line argument printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Hook initializer command line argument validation function.
    ///
    /// @param[in] argc      command line argument count
    /// @param[in] argv      command line arguments
    /// @param[in,out] argr  array that tracks which command line arguments have
    ///                      been processed
    ///
    /// @return 0 if the arguments are valid, non-zero otherwise
    ///
    int (*check_args)(
        int argc, char * const *argv, int *argr);

    ///
    /// @brief Hook initializer data initialization function.
    ///
    /// @param[in] format  requested data format label
    /// @param[in] argc    command line argument count
    /// @param[in] argv    command line arguments
    ///
    /// @return valid hook data envelope; NULL if the initialization failed
    ///
    struct hook_data_env* (*init)(
        hook_data_format_t format, int argc, char * const *argv);
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Data type for a hook experiment supplementer.
///
struct hook_supplementer_t {
    char const *name;               ///< hook supplementer name
    char const *desc;               ///< hook supplementer description
    hook_data_format_t *formats;    ///< supported data format labels

    ///
    /// @brief Hook supplementer instruction printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_usage)(int argc, char * const *argv);

    ///
    /// @brief Hook supplementer command line argument printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Hook supplementer command line argument validation function.
    ///
    /// @param[in] argc      command line argument count
    /// @param[in] argv      command line arguments
    /// @param[in,out] argr  array that tracks which command line arguments have
    ///                      been processed
    ///
    /// @return 0 if the arguments are valid, non-zero otherwise
    ///
    int (*check_args)(
        int argc, char * const *argv, int *argr);

    ///
    /// @brief Hook supplementer data initialization function.
    ///
    /// @param[in] env     hook data envelope
    /// @param[in] argc    command line argument count
    /// @param[in] argv    command line arguments
    ///
    /// @return valid hook data envelope; NULL if the initialization failed
    ///
    void (*supplement)(
        struct hook_data_env *env, int argc, char * const *argv);
};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Data type for a hook solver state.
///
///  The state is valid from hook solver's prepare function call to hook
///  solver's finalize function call.
///
typedef void * hook_solver_state_t;

///
/// Data type for a hook experiment solver.
///
struct hook_solver {
    char *name;                   ///< hook solver name
    char *desc;                   ///< hook solver description
    hook_data_format_t *formats;  ///< supported data format labels

    ///
    /// @brief Hook solver instruction printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_usage)(int argc, char * const *argv);

    ///
    /// @brief Hook solver command line argument printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_args)(int argc, char * const *argv);

    ///
    /// @brief Hook solver command line argument validation function.
    ///
    /// @param[in] argc      command line argument count
    /// @param[in] argv      command line arguments
    /// @param[in,out] argr  array that tracks which command line arguments have
    ///                      been processed
    ///
    /// @return 0 if the arguments are valid, non-zero otherwise
    ///
    int (*check_args)(int argc, char * const *argv, int *argr);

    ///
    /// @brief Hook solver state preparation function.
    ///
    /// @param[in] argc     command line argument count
    /// @param[in] argv     command line arguments
    /// @param[in,out] env  solver input/output data envelope
    ///
    /// @return a hook solver state (usually the data in an internal format)
    ///
    hook_solver_state_t (*prepare)(
        int argc, char * const *argv, struct hook_data_env *env);

    ///
    /// @brief Hook solver state finalization function.
    ///
    ///  This function should cleanup hook solver's state, i.e., release
    ///  allocated resources, if any.
    ///
    /// @param[in,out] state  hook solver's state
    /// @param[in,out] env    solver input/output data envelope
    ///
    /// @return 0 if successful, non-zero otherwise
    ///
    int (*finalize)(
        hook_solver_state_t state, struct hook_data_env *env);

    ///
    /// @brief Hook solver execution function.
    ///
    /// @param[in,out] state  hook solver's state
    ///
    /// @return 0 if the call was successful, non-zero otherwise
    ///
    int (*run)(hook_solver_state_t state);

};

////////////////////////////////////////////////////////////////////////////////

///
/// @brief Hook mode enumerator.
///
///  Mode specifies how the experiment should react to various return values:
///
///     mode \ return  |  HOOK_WARNING  | HOOK_SOFT_FAIL | HOOK_HARD_FAIL
///  ------------------+----------------+----------------+----------------
///   HOOK_MODE_NORMAL |    warning     | delayed report | immediate fail
///   HOOK_MODE_STRICT |    warning     | immediate fail | immediate fail
///   HOOK_MODE_ALL    | immediate fail | immediate fail | immediate fail
///
///  Delayed reporting records the failure but the experiment continues;
///  failure is reported in the end. Immediate fail will end the experiment
///  immediately.
///
typedef enum {
    HOOK_MODE_NORMAL,   ///< normal mode
    HOOK_MODE_STRICT,   ///< strict mode
    HOOK_MODE_ALL       ///< "all" (paranoid) mode
} hook_mode_t;

///
/// @brief Data type for a hook descriptor.
///
///  Hook descriptor encapsulates the actual hook and the hook experiment
///  specific options.
///
struct hook_descr_t {
    int is_enabled;             ///< hook is enabled by default
    hook_mode_t default_mode;   ///< default mode
    struct hook_t const *hook;  ///< hook
};

///
/// @brief Data type for a hook experiment descriptor.
///
struct hook_experiment_descr {

    ///
    /// @brief Hook hook experiment instruction printout function.
    ///
    /// @param[in] argc  command line argument count
    /// @param[in] argv  command line arguments
    ///
    void (*print_usage)(int argc, char * const *argv);

    struct hook_initializer_t const **initializers;   ///< initializers
    struct hook_supplementer_t const **supplementers; ///< supplementers
    struct hook_solver const **solvers;               ///< solvers
    struct hook_descr_t const **hook_descrs;          ///< hook descriptors
};

///
/// @brief Prints hook experiment's instructions.
///
void hook_experiment_print_usage(
    int argc, char * const *argv, experiment_info_t const info);

///
/// @brief Prints hook experiment's command line arguments.
///
void hook_experiment_print_args(
    int argc, char * const *argv, experiment_info_t const info);

///
/// @brief Checks hook experiment's command line arguments.
///
int hook_experiment_check_args(
    int argc, char * const *argv, int *argr, experiment_info_t const info);

///
/// @brief Executes the hook experiment.
///
int hook_experiment_run(
    int argc, char * const *argv, experiment_info_t const info);

#endif
