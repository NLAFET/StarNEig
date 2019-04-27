///
/// @file
///
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
///
/// @section LICENSE
///
/// Copyright (c) 2019, Umeå Universitet
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
#include <starneig/sep_sm.h>
#include "cpu.h"
#include "../../mpi/utils.h"
#include "../../mpi/node_internal.h"
#include "../../common/node_internal.h"
#include "../../common/tiles.h"
#include "../../common/utils.h"
#include <starpu.h>

#ifdef STARNEIG_ENABLE_MPI
#include <starpu_mpi.h>
#include <starneig/node.h>
#endif

static struct starpu_codelet bound_cl = {
    .name = "bound",
    .cpu_funcs = {starneig_cpu_bound_DM},
    .nbuffers = STARPU_VARIABLE_NBUFFERS
};

void starneig_std_eigvecs_insert_bound(
    int prio,
    int rbegin, int rend, int cbegin, int cend, // S
    starpu_data_handle_t S_tile_norm,
    starneig_matrix_descr_t matrix_s, mpi_info_t mpi
)
{
    if (rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_W, S_tile_norm, helper, 0);

    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);

    struct packing_info packing_info;
    starneig_pack_window(STARPU_R, rbegin, rend, cbegin, cend,
        matrix_s, helper, &packing_info, 0);

    starpu_mpi_task_insert(
        starneig_mpi_get_comm(),
        &bound_cl,
        STARPU_EXECUTE_ON_NODE,
        starneig_get_elem_owner_matrix_descr(rbegin, cbegin, matrix_s),
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &packing_info, sizeof(packing_info),
        STARPU_DATA_MODE_ARRAY, helper->descrs, helper->count, 0);

    starneig_free_packing_helper(helper);
}


/*
void starneig_std_eigvecs_insert_backsolve(
    int prio,
    int rbegin, int rend, int cbegin, int cend, // S
    starpu_data_handle_t S,
    starpu_data_handle_t Snorm,
    starpu_data_handle_t X,
    starpu_data_handle_t scales,
    starpu_data_handle_t Xnorms,
    starpu_data_handle_t lambda_type,
    starpu_data_handle_t selected,
    starpu_data_handle_t info,
    double smlnum,
    starneig_matrix_descr_t matrix_s, mpi_info_t mpi)
{
    if (rend-rbegin < 1 || cend-cbegin < 1)
        return;

    struct packing_helper *helper = starneig_init_packing_helper();

    starneig_pack_handle(STARPU_R, S, helper, 0); // This should probably pack all handles.
    starneig_pack_handle(STARPU_R, Snorm, helper, 0);

    starneig_pack_cached_scratch_matrix(
        rend-rbegin, cend-cbegin, sizeof(double), helper);

    struct packing_info packing_info;
    starneig_pack_window(STARPU_RW, rbegin, rend, cbegin, cend, // Apparently has to be RW
        matrix_s, helper, &packing_info, 0);

    starpu_mpi_task_insert(
        starneig_mpi_get_comm(),
        &backsolve_cl,
        STARPU_EXECUTE_ON_NODE,
        starneig_get_elem_owner_matrix_descr(rbegin, cbegin, matrix_a),
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &smlnum, sizeof(smlnum),
    );

    starpu_mpi_task_insert(
        starneig_mpi_get_comm(),
        &backsolve_cl,
        STARPU_EXECUTE_ON_NODE, owner,
        STARPU_PRIORITY, prio,
        STARPU_VALUE, &smlnum, sizeof(smlnum),
    );

}
*/
