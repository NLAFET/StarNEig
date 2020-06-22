///
/// @file
///
/// @brief This file contains configuration structures and functions for the
/// expert interface functions.
///
/// @author Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
/// @author Angelika Schwarz (angies@cs.umu.se), Umeå University
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

#ifndef STARNEIG_EXPERT_H
#define STARNEIG_EXPERT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <starneig/configuration.h>

///
/// @defgroup starneig_ex_conf Expert configuration structures
///
/// @brief Configuration structures and functions for the expert interface
/// functions.
///
/// @{
///

///
/// @name Hessenberg reduction
/// @{
///

///
/// @brief Default tile size.
///
#define STARNEIG_HESSENBERG_DEFAULT_TILE_SIZE            -1

///
/// @brief Default panel width.
///
#define STARNEIG_HESSENBERG_DEFAULT_PANEL_WIDTH          -1

///
/// @brief Default parallel worker size.
///
#define STARNEIG_HESSENBERG_DEFAULT_PARALLEL_WORKER_SIZE -1

///
/// @brief Hessenberg reduction configuration structure.
///
struct starneig_hessenberg_conf {

    /// The matrices are divided into square tiles. This parameter defines the
    /// used tile size. If the parameter is set to
    /// @ref STARNEIG_HESSENBERG_DEFAULT_TILE_SIZE, then the implementation
    /// will determine a suitable tile size automatically.
    int tile_size;

    /// The reduction is performed one panel at a time. This parameter defines
    /// the used panel width. If the parameter is set to
    /// @ref STARNEIG_HESSENBERG_DEFAULT_PANEL_WIDTH, then the implementation
    /// will determine a suitable panel width automatically.
    int panel_width;

    /// The CPU variants of the panel reduction and trailing matrix update tasks
    /// are multithreaded. This parameter defines the number of cores allocated
    /// to these tasks. If the parameter is set to
    /// @ref STARNEIG_HESSENBERG_DEFAULT_PARALLEL_WORKER_SIZE, then the
    /// implementation will determine a suitable CPU core count automatically.
    int parallel_worker_size;
};

///
/// @brief Initializes a Hessenberg reduction configuration structure with
/// default parameters.
///
/// @param[out] conf
///         The Hessenberg reduction configuration structure.
///
void starneig_hessenberg_init_conf(struct starneig_hessenberg_conf *conf);

///
/// @}
///

///
/// @name Schur reduction
/// @{
///

///
/// @brief Default iteration limit.
///
#define STARNEIG_SCHUR_DEFAULT_INTERATION_LIMIT        -1

///
/// @brief Default tile size.
///
#define STARNEIG_SCHUR_DEFAULT_TILE_SIZE               -1

///
/// @brief Default sequential QR limit.
///
#define STARNEIG_SCHUR_DEFAULT_SMALL_LIMIT             -1

///
/// @brief Default AED window size.
///
#define STARNEIG_SCHUR_DEFAULT_AED_WINDOW_SIZE         -1

///
/// @brief Default nibble value.
///
#define STARNEIG_SCHUR_DEFAULT_AED_NIBBLE              -1

///
/// @brief Default soft sequential AED limit.
///
#define STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_SOFT_LIMIT -1

///
/// @brief Default hard sequential AED limit.
///
#define STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_HARD_LIMIT -1

///
/// @brief Default shift count.
///
#define STARNEIG_SCHUR_DEFAULT_SHIFT_COUNT             -1

///
/// @brief Default bulge chasing window size.
///
#define STARNEIG_SCHUR_DEFAULT_WINDOW_SIZE             -1

///
/// @brief Rounded bulge chasing window.
///
#define STARNEIG_SCHUR_ROUNDED_WINDOW_SIZE             -2

///
/// @brief Default number of shifts per bulge chasing window.
///
#define STARNEIG_SCHUR_DEFAULT_SHIFTS_PER_WINDOW       -1

///
/// @brief Default left-hand side update width.
///
#define STARNEIG_SCHUR_DEFAULT_UPDATE_WIDTH            -1

///
/// @brief Default right-hand side update height.
///
#define STARNEIG_SCHUR_DEFAULT_UPDATE_HEIGHT           -1

///
/// @brief Default deflation threshold.
///
#define STARNEIG_SCHUR_DEFAULT_THRESHOLD               -1

///
/// @brief Norm stable deflation threshold.
///
#define STARNEIG_SCHUR_NORM_STABLE_THRESHOLD           -2

///
/// @brief LAPACK-style deflation threshold.
///
#define STARNEIG_SCHUR_LAPACK_THRESHOLD                -3

///
/// @brief Schur reduction configuration structure.
///
struct starneig_schur_conf {

    /// The QR/QZ is an iterative algorithm. This parameter defines the
    /// maximum number of iterations the algorithm is allowed to perform.
    /// If the parameter is @ref STARNEIG_SCHUR_DEFAULT_INTERATION_LIMIT,
    /// then the implementation will determine a suitable iteration limit
    /// automatically.
    int iteration_limit;

    /// The matrices are divided into square tiles. This parameter defines the
    /// used tile size. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_TILE_SIZE, then the implementation
    /// will determine a suitable tile size automatically.
    int tile_size;

    /// As the QR/QZ algorithm progresses, the size of the active region
    /// shrinks. Once the size of the active region is small enough, then
    /// the remaining problem is solved in a sequential manner. This parameter
    /// defines the transition point where the implementation switches to a
    /// sequential QR algorithm. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_SMALL_LIMIT, then the implementation will
    /// determine a suitable switching point automatically.
    int small_limit;

    /// The implementation relies on a so-called Aggressive Early Deflation
    /// (AED) technique to accelerate the convergence of the algorithm. Each AED
    /// is performed inside a small diagonal window. This parameter defines
    /// used AED window size. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_AED_WINDOW_SIZE, then the implementation
    /// will determine a suitable AED window size automatically.
    int aed_window_size;

    /// The implementation relies on a so-called Aggressive Early Deflation
    /// (AED) technique to accelerate the convergence of the algorithm. Each AED
    /// is performed inside a small diagonal window. If the number deflated
    /// (converged) eigenvalues is larger than `(aed_nibble / 100)` \f$\times\f$
    /// `size of AED window`, then the next bulge chasing step is skipped. If
    /// the parameter is set to @ref STARNEIG_SCHUR_DEFAULT_AED_NIBBLE, then the
    /// implementation will determine a suitable value automatically.
    int aed_nibble;

    /// The implementation relies on a so-called Aggressive Early Deflation
    /// (AED) technique to accelerate the convergence of the algorithm. Each AED
    /// is performed inside a small diagonal window. An AED can be performed
    /// sequentially or in parallel. This parameter defines the transition point
    /// where the implementation allowed to switch to a sequential AED
    /// algorithm. The decision is made based on the size of the AED window.
    /// If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_SOFT_LIMIT, then the
    /// implementation will determine a suitable switching point
    /// automatically.
    int aed_parallel_soft_limit;

    /// The implementation relies on a so-called Aggressive Early Deflation
    /// (AED) technique to accelerate the convergence of the algorithm. Each AED
    /// is performed inside a small diagonal window. An AED can be performed
    /// sequentially or in parallel. This parameter defines the transition point
    /// where the implementation switches to a sequential AED algorithm. The
    /// decision is made based on the size of the AED window. If the parameter
    /// is set to @ref STARNEIG_SCHUR_DEFAULT_AED_PARALLEL_HARD_LIMIT, then the
    /// implementation will determine a suitable switching point
    /// automatically.
    int aed_parallel_hard_limit;

    /// The QR/QZ algorithm chases a set of \f$3 \times 3\f$ bulges across
    /// the diagonal of the Hessenberg(-triangular) decomposition. Two shifts
    /// (eigenvalue estimates) are required to generate each bulge. This
    /// parameter defines the number of shifts to use. If the parameter is set
    /// to @ref STARNEIG_SCHUR_DEFAULT_SHIFT_COUNT, then the implementation
    /// will determine a suitable shift count automatically.
    int shift_count;

    /// The QR/QZ algorithm chases a set of \f$3 \times 3\f$ bulges across
    /// the diagonal of the Hessenberg(-triangular) decomposition. The bulges
    /// are chased in batches. The related similarity transformations are
    /// initially restricted to inside a small diagonal window and the
    /// accumulated transformation  are applied only later as BLAS-3 updates.
    /// This parameter defines the used bulge chasing window size. If the
    /// parameter is set to @ref STARNEIG_SCHUR_ROUNDED_WINDOW_SIZE, then
    ///
    ///   - maximum window size is set to 2 * @ref tile_size and
    ///   - the windows are placed such that their lower right corners respect
    ///     the boundaries of the underlying data tiles.
    ///
    /// If the parameter is set to @ref STARNEIG_SCHUR_DEFAULT_WINDOW_SIZE, then
    /// the implementation will determine a suitable window size automatically.
    int window_size;

    /// The QR/QZ algorithm chases a set of \f$3 \times 3\f$ bulges across
    /// the diagonal of the Hessenberg(-triangular) decomposition. The bulges
    /// are chased in batches. This parameter defines the used batch size.
    /// If the parameter is set to @ref STARNEIG_SCHUR_DEFAULT_SHIFTS_PER_WINDOW
    /// then the implementation will determine a suitable batch size
    /// automatically.
    int shifts_per_window;

    /// The similarity similarity transformations are initially restricted to
    /// inside a small diagonal window and the accumulated transformation are
    /// applied only later as BLAS-3 updates. This parameter defines the width
    /// of each left-hand side update task. The value should be multiple of the
    /// tile size. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_UPDATE_WIDTH, then the implementation will
    /// determine a suitable width automatically.
    int update_width;

    /// The similarity similarity transformations are initially restricted to
    /// inside a small diagonal window and the accumulated transformation are
    /// applied only later as BLAS-3 updates. This parameter defines the height
    /// of each right-hand side update task. The value should be multiple of the
    /// tile size. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_UPDATE_HEIGHT, then the implementation will
    /// determine a suitable height automatically.
    int update_height;

    /// The QR/QZ algorithm is allowed to set tiny matrix entires to zero as
    /// long as their magnitudes are smaller that a given threshold. This
    /// parameter defines the threshold for the left-hand side matrix (\f$H\f$).
    /// If the parameter is set to @ref STARNEIG_SCHUR_DEFAULT_THRESHOLD, then
    /// the implementation will determine a suitable threshold automatically. If
    /// the parameter is set to @ref STARNEIG_SCHUR_NORM_STABLE_THRESHOLD, then
    /// the implementation will use the threshold \f$u |H|_F\f$, where \f$u\f$
    /// is the unit roundoff and \f$|H|_F\f$ is the Frobenius norm of the matrix
    /// \f$H\f$. If the parameter is set to
    /// @ref STARNEIG_SCHUR_LAPACK_THRESHOLD, then the implementation will use
    /// a deflation threshold that is compatible with LAPACK.
    double left_threshold;

    /// The QZ algorithm is allowed to set tiny matrix entires to zero as
    /// long as their magnitudes are smaller that a given threshold. This
    /// parameter defines the threshold for the right-hand side matrix
    /// (\f$R\f$) off-diagonal entires. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_THRESHOLD, then the implementation will
    /// determine a suitable threshold automatically. If the parameter is set to
    /// @ref STARNEIG_SCHUR_NORM_STABLE_THRESHOLD, then the implementation will
    /// use the threshold \f$u |R|_F\f$, where \f$u\f$ is the unit roundoff and
    /// \f$|H|_F\f$ is the Frobenius norm of the matrix \f$R\f$. If the
    /// parameter is set to @ref STARNEIG_SCHUR_LAPACK_THRESHOLD, then the
    /// implementation will use a deflation threshold that is compatible with
    /// LAPACK.
    double right_threshold;

    /// The QZ algorithm is allowed to set tiny matrix entires to zero as
    /// long as their magnitudes are smaller that a given threshold. This
    /// parameter defines the threshold for the right-hand side matrix
    /// (\f$R\f$) diagonal entries. If the parameter is set to
    /// @ref STARNEIG_SCHUR_DEFAULT_THRESHOLD, then the implementation will
    /// determine a suitable threshold automatically. If the parameter is set to
    /// @ref STARNEIG_SCHUR_NORM_STABLE_THRESHOLD, then the implementation will
    /// use the threshold \f$u |R|_F\f$, where \f$u\f$ is the unit roundoff and
    /// \f$|R|_F\f$ is the Frobenius norm of the matrix \f$R\f$.
    double inf_threshold;
};

///
/// @brief Initializes a Schur reduction configuration structure with default
/// parameters.
///
/// @param[out] conf
///         The Schur reduction configuration structure.
///
void starneig_schur_init_conf(struct starneig_schur_conf *conf);

///
/// @}
///

///
/// @name Eigenvalue reordering
/// @{
///

///
/// @brief Reordering plan enumerator.
///
///  Eigenvalues that fall within a diagonal computation *window* are reordered
///  such that all selected eigenvalues are moved to the upper left corner of
///  the window. The corresponding orthogonal transformations are accumulated
///  to separate accumulator matrix / matrices.
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///    +---------+       +---------+
///    |¤ x x x x|       |$ x x x x|       $ selected eigenvalue
///    |  $ x x x|       |  $ x x x|       ¤ deselected eigenvalue
///    |    ¤ x x| ==> Q |    ¤ x x| Q^T   x non-zero entry
///    |      ¤ x|       |      ¤ x|
///    |        $|       |        ¤|
///    +---------+       +---------+
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  A *window chain* comprises from multiple overlapping diagonal computation
///  windows that are intended to be processed in a particular order. More
///  precisely, the windows are placed such that the overlap between two windows
///  is big enough to accommodate all selected eigenvalues that fall within the
///  preceding windows. In this way, the windows can be processed in sequential
///  order, starting from the bottom window, such that the reordering that takes
///  place in one window always moves the preceding selected eigenvalues to the
///  lower right corner of the next window. In the end, all selected that fall
///  within the combined computation area of the chain are moved to the upper
///  left corner of the topmost window.
///
///  An example showing how an eigenvalue can be moved six entries upwards by
///  using three diagonal windows:
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///   +-------+         +-------+         +-------+         +-------+
///   | x x x |  ===>   | x x x |  ===>   | x x x |  ===>   | #<--+ |
///   |   x-x-+---+     |   x-x-+---+     |   x-x-+---+     |   ¤-|-+---+
///   |   | x x x |     |   | x x x |     |   | #<--+ |     |   | ¤ ¤ ¤ |
///   +---+---x-x-+---+ +---+---x-x-+---+ +---+---¤-|-+---+ +---+---¤-¤-+---+
///       |   | x x x |     |   | #<--+ |     |   | ¤ ¤ ¤ |     |   | ¤ ¤ ¤ |
///       +---+---x x |     +---+---¤ | |     +---+---¤ ¤ |     +---+---¤ ¤ |
///           |     # |         |     ¤ |         |     ¤ |         |     ¤ |
///           +-------+         +-------+         +-------+         +-------+
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  The number of selected eigenvalues that can be moved by a single window
///  chain is limited by the windows size. Thus, the whole reordering procedure
///  usually involves multiple chains that must be be processed in a particular
///  order. A *chain list* describes a list of chains that are intended to be
///  processed together. Window chains that belong to different chain lists are
///  processed separately.
///
///  A *plan* consists from one or more chain lists that are intended to be
///  processed in a particular order.
///
///  @ref STARNEIG_REORDER_ONE_PART_PLAN :
///
///  The first chain is placed in the upper left corner of the matrix and its
///  size is chosen such that it contains a desired number of selected
///  eigenvalues (@ref starneig_reorder_conf::values_per_chain parameter). The
///  next chain is places such that its upper left corner is located one entry
///  after the location where the last selected eigenvalue, that falls within
///  the first chain, would be after the reordering. The chain is sized such
///  that the part of the chain, that does not intersect the first chain,
///  contain the desired number of selected eigenvalues. This same procedure is
///  repeated until all selected eigenvalues have been accounted for. All chains
///  belong to the same chain lists and are intended to be processed
///  sequentially.
///
///  An example showing the placement of the chains in a case where each chain
///  wields two selected eigenvalues:
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///                                                         before:      after:
///   .-------+                                             .            0
///   | 0     | <-- chain 0                                 0 _          1_
///   |   .-----------+                                     . .   ====>  2
///   |   | . |       | <-- chain 1                         . . _        3_
///   +---|---1---------------+ . . . . . . . . . . . . . . 1_. .        4
///       |   | .     |       | <-- chain 2                   . . _      5_
///       |   |   2-------------------+                       2 . .      6
///       |   |   | . |       |       | <-- chain 3           . . . _    7_
///       +---|---|---3-----------------------+ . . . . . . . 3_. . .    8
///           |   |   | 4     |       |       | <-- chain 4     4 . .    9
///           |   |   |   .   |       |       |                 . . .    .
///           |   |   |     . |       |       |                 . . .    .
///           +---|---|-------5       |       | . . . . . . . . 5_. .    .
///               |   |         6     |       |                   6 .    .
///               |   |           .   |       |                   . .    .
///               |   |             . |       |                   . .    .
///               +---|---------------7       | . . . . . . . . . 7_.    .
///                   |                 .     |                     .    .
///                   |                   .   |                     .    .
///                   |                     8 |                     8    .
///                   +-----------------------9 . . . . . . . . . . 9    .
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  An example showing what happens when the first three chains are processed:
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///   .-------+                        >0<------+
///   | 0     |                         |>1<    |
///   |   .-----------+                 |   #-----------+
///   |   | . |       |                 |   | # |       |
///   +---|---1---------------+         +---|---#---------------+
///       |   | .     |       |             |   | .     |       |
///       |   |   2   |       |  ===>       |   |   2   |       |  ===>
///       |   |     . |       |             |   |     . |       |
///       +---|-------3       |             +---|-------3       |
///           |         4     |                 |         4     |
///           |           .   |                 |           .   |
///           |             . |                 |             . |
///           +---------------5                 +---------------5
///
///   0-------+                         0-------+
///   | 1     |                         | 1     |
///   |  >2<----------+                 |   2-----------+
///   |   |>3<|       |                 |   | 3 |       |
///   +---|---#---------------+         +---|-->4<--------------+
///       |   | #     |       |             |   |>5<    |       |
///       |   |   #   |       |   ===>      |   |   #   |       |
///       |   |     # |       |             |   |     # |       |
///       +---|-------#       |             +---|-------#       |
///           |         4     |                 |         #     |
///           |           .   |                 |           #   |
///           |             . |                 |             # |
///           +---------------5                 +---------------#
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  If necessary, each chain is re-sized to avoid splitting any
///  \f$2 \times 2\f$ tiles.
///
///  Windows are placed such that the first window is located in the lower
///  right corner of the computation area of the window chain. The last window
///  is correspondingly placed in the upper left corner of the computation area.
///
///  If necessary, each window is re-sized to avoid splitting any
///  \f$2 \times 2\f$ tiles.
///
///  @ref STARNEIG_REORDER_MULTI_PART_PLAN :
///
///  A multi-part reordering plan is derived from an one-part reordering plan
///  by splitting the chains into sub-chains as shown below:
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  Initial one-part plan:
///    Chain 0: aaaaaa
///    Chain 1:  bbbbbbbbbb               a,b,c,d,e diagonal computation window
///    Chain 2:   cccccccccccccc
///    Chain 3:    dddddddddddddddddd
///    Chain 4:     eeeeeeeeeeeeeeeeeeeeee
///
///  Resulting multi-part plan:
///    Chain 0: aaaaaa
///    Chain 1:  ......bbbb
///    Chain 2:   ..........cccc               chain list 0
///    Chain 3:    ..............dddd
///    Chain 4:     ..................eeee
///    -----------------------------------------------------
///    Chain 0:  bbbbbb....
///    Chain 1:   ......cccc....               chain list 1
///    Chain 2:    ..........dddd....
///    Chain 3:     ..............eeee....
///    -----------------------------------------------------
///    Chain 0:   cccccc........
///    Chain 1:    ......dddd........          chain list 2
///    Chain 2:     ..........eeee........
///    -----------------------------------------------------
///    Chain 0:    dddddd............          chain list 3
///    Chain 1:     ......eeee............
///    -----------------------------------------------------
///    Chain 0:     eeeeee................     chain list 4
///
///  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
///  Note that the chains that belong to the same chain list are independent
///  from each other and can therefore be processed in an arbitrary order.
///
typedef enum {
    STARNEIG_REORDER_DEFAULT_PLAN    = 1,     ///< Default plan.
    STARNEIG_REORDER_ONE_PART_PLAN   = 2,     ///< One part plan.
    STARNEIG_REORDER_MULTI_PART_PLAN = 3      ///< Multi part plan.
} starneig_reorder_plan_t;

///
/// @brief Task insertion blueprint.
///
///  A task insertion blueprint defines how a reordering plan is carried out.
///
typedef enum {

    /// Default blueprint.
    STARNEIG_REORDER_DEFAULT_BLUEPRINT = 1,

    /// One-pass forward dummy blueprint. Processes the window chains in order
    /// starting from the topmost chain. All update tasks are inserted right
    /// after each window reordering task.
    STARNEIG_REORDER_DUMMY_INSERT_A = 2,

    /// Two-pass backward dummy blueprint. Processes the window chains in two
    /// phases starting from the bottommost chain. The window reordering tasks
    /// and the right-hand side update tasks are inserted during the first
    /// phase. Other update tasks are inserted during the second phase.
    STARNEIG_REORDER_DUMMY_INSERT_B = 3,

    /// One-pass forward chain blueprint. Processes the window chains in order
    /// starting from the topmost chain. The window reordering tasks and high
    /// priority right-hand side update tasks are inserted first. Other update
    /// tasks are inserted after them.
    STARNEIG_REORDER_CHAIN_INSERT_A = 4,

    /// Two-pass forward chain blueprint. Processes the window chains in two
    /// phases starting from the topmost chain. The window reordering tasks and
    /// high priority right-hand side update tasks are inserted first. The left-
    /// hand side updates are inserted after them. Other updates are inserted
    /// during the second phase.
    STARNEIG_REORDER_CHAIN_INSERT_B = 5,

    /// One-pass backward chain blueprint. Processes the window chains in order
    /// starting from the bottommost chain. The window reordering tasks and high
    /// priority right-hand side update tasks are inserted first. Other update
    /// tasks are inserted later.
    STARNEIG_REORDER_CHAIN_INSERT_C = 6,

    /// Two-pass backward chain blueprint. Processes the window chains in two
    /// phases starting from the bottommost chain. The window reordering tasks
    /// and high priority right-hand side update tasks are inserted first.
    /// Update tasks that are related to the Schur matrix are inserted later.
    /// Update tasks that are related to the orthogonal matrices are inserted
    /// during the second phase.
    STARNEIG_REORDER_CHAIN_INSERT_D = 7,

    /// Two-pass delayed backward chain blueprint. Processes the window chains
    /// in order starting from the bottommost chain. The window reordering tasks
    /// and high priority right-hand side update tasks are inserted first.
    /// Update tasks that are related to the Schur matrix are inserted later.
    /// Update tasks that are related to the orthogonal matrices are inserted
    /// only after all chain list have been processed.
    STARNEIG_REORDER_CHAIN_INSERT_E = 8,

    /// Three-pass delayed backward chain blueprint. Processes the window chains
    /// in two phases starting from the bottommost chain. The window reordering
    /// tasks and high priority right-hand side update tasks are inserted during
    /// the first phase. Update tasks that are related to the Schur matrix are
    /// inserted during the second phase. Update tasks that are related to the
    /// orthogonal matrices are inserted only after all chain list have been
    /// processed.
    STARNEIG_REORDER_CHAIN_INSERT_F = 9

} starneig_reorder_blueprint_t;

///
/// @brief Default left-hand side update task width.
///
#define STARNEIG_REORDER_DEFAULT_UPDATE_WIDTH               -1

///
/// @brief Default right-hand side update task height.
///
#define STARNEIG_REORDER_DEFAULT_UPDATE_HEIGHT              -1

///
/// @brief Default tile size.
///
#define STARNEIG_REORDER_DEFAULT_TILE_SIZE                  -1

///
/// @brief Default number of selected eigenvalues per window.
///
#define STARNEIG_REORDER_DEFAULT_VALUES_PER_CHAIN           -1

///
/// @brief Default default window size.
///
#define STARNEIG_REORDER_DEFAULT_WINDOW_SIZE                -1

///
/// @brief Default rounded window size.
///
#define STARNEIG_REORDER_ROUNDED_WINDOW_SIZE                -2

///
/// @brief Default small window size.
///
#define STARNEIG_REORDER_DEFAULT_SMALL_WINDOW_SIZE          -1

///
/// @brief Default small window threshold.
///
#define STARNEIG_REORDER_DEFAULT_SMALL_WINDOW_THRESHOLD     -1

///
/// @brief Eigenvalue reordering configuration structure.
///
struct starneig_reorder_conf {

    /// This parameter plan defines the used reordering plan. If the parameter
    /// is set to @ref STARNEIG_REORDER_DEFAULT_PLAN, then the implementation
    /// will determine a suitable reordering plan automatically.
    starneig_reorder_plan_t plan;

    /// This parameter defines the used task insertion blueprint. If the
    /// parameter is set to @ref STARNEIG_REORDER_DEFAULT_BLUEPRINT, then the
    /// implementation will determine a suitable task insertion blueprint
    /// automatically.
    starneig_reorder_blueprint_t blueprint;

    /// The matrices are divided into square tiles. This parameter defines the
    /// used tile size. If the parameter is set to
    /// @ref STARNEIG_REORDER_DEFAULT_TILE_SIZE, then the implementation
    /// will determine a suitable tile size automatically.
    int tile_size;

    /// The selected eigenvalues are processed in batches and each batch is
    /// assigned a window chain. This parameter defines the number of selected
    /// eigenvalues processed by each window chain. If the parameter is set to
    /// @ref STARNEIG_REORDER_DEFAULT_VALUES_PER_CHAIN, then the implementation
    /// will determine a suitable value automatically.
    int values_per_chain;

    /// The similarity similarity transformations are initially restricted to
    /// inside a small diagonal window and the accumulated transformation are
    /// applied only later as BLAS-3 updates. This parameter defines the size
    /// of the window. If the parameter is set to
    /// @ref STARNEIG_REORDER_ROUNDED_WINDOW_SIZE, then
    ///
    ///   - maximum window size is set to 2 * @ref tile_size,
    ///   - the windows are placed such that their upper left corners respect
    ///     the boundaries of the underlying data tiles, and
    ///   - the parameter @ref values_per_chain is ignored.
    ///
    /// If the parameter is set to @ref STARNEIG_REORDER_DEFAULT_WINDOW_SIZE,
    /// then the implementation will determine a suitable window size
    /// automatically.
    int window_size;

    /// Larger diagonal window are processed using even smaller diagonal
    /// windows in a recursive manner. This parameter defines the used small
    /// window size. If the parameter is set to
    /// @ref STARNEIG_REORDER_DEFAULT_SMALL_WINDOW_SIZE, then the implementation
    /// will determine a suitable small window size automatically.
    int small_window_size;

    /// Larger diagonal window are processed using even smaller diagonal
    /// windows in a recursive manner. This parameter defines the largest
    /// diagonal window that is processed in a scalar manner. If the parameter
    /// is set to @ref STARNEIG_REORDER_DEFAULT_SMALL_WINDOW_THRESHOLD,
    /// then the implementation will determine a suitable threshold
    /// automatically.
    int small_window_threshold;

    /// The similarity similarity transformations are initially restricted to
    /// inside a small diagonal window and the accumulated transformation are
    /// applied only later as BLAS-3 updates. This parameter defines the width
    /// of each left-hand side update task. The value should be multiple of the
    /// tile size. If the parameter is set to
    /// @ref STARNEIG_REORDER_DEFAULT_UPDATE_WIDTH, then the implementation will
    /// determine a suitable width automatically.
    int update_width;

    /// The similarity similarity transformations are initially restricted to
    /// inside a small diagonal window and the accumulated transformation are
    /// applied only later as BLAS-3 updates. This parameter defines the height
    /// of each right-hand side update task. The value should be multiple of the
    /// tile size. If the parameter is set to
    /// @ref STARNEIG_REORDER_DEFAULT_UPDATE_HEIGHT, then the implementation
    /// will determine a suitable height automatically.
    int update_height;
};

///
/// @brief Initializes an eigenvalue reordering configuration structure with
/// default parameters.
///
/// @param[out] conf
///         The eigenvalue reordering configuration structure.
///
void starneig_reorder_init_conf(struct starneig_reorder_conf *conf);

///
/// @}
///

///
/// @name Eigenvectors
/// @{
///

///
/// @brief Default tile size.
///
#define STARNEIG_EIGENVECTORS_DEFAULT_TILE_SIZE         -1

///
/// @brief Eigenvector computation configuration structure.
///
struct starneig_eigenvectors_conf {

    /// The matrices are divided into tiles. This parameter defines the used
    /// tile size. If the parameter is set to
    /// @ref STARNEIG_EIGENVECTORS_DEFAULT_TILE_SIZE, then the implementation
    /// will determine a suitable tile size automatically.
    int tile_size;
};


///
/// @brief Initializes an eigenvectors configuration structure with default
/// parameters.
///
/// @param[out] conf
///         The eigenvectors configuration structure.
///
void starneig_eigenvectors_init_conf(struct starneig_eigenvectors_conf *conf);

///
/// @}
///

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // STARNEIG_EXPERT_H
