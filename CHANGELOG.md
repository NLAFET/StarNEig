### v0.1.0:
 - First stable release of the library.

### v0.1-beta.6:
 - Fix `starneig_node_enable_pinning()` and `starneig_node_disable_pinning()`
   functions.
 - Built `pdgghrd` as a separate library.
 - Deprecate several precompiler defines and interface functions. Add
   `<starneig/distr_helpers.h>` header file.

### v0.1-beta.5:
 - Improve the performance of the Hessenberg reduction phase by limiting
   the number of submitted tasks. This should reduce the task scheduling
   overhead.
 - Allocate pinned memory by default when CUDA support is enabled. Add
   `starneig_enable_pinning()` and `starneig_disable_pinning()`.

### v0.1-beta.4:
 - Fix a problem where infinite eigenvalues were detected too late.
 - Add an option to choose between the norm stable deflation condition
   (`STARNEIG_SCHUR_NORM_STABLE_THRESHOLD`) and and the LAPACK style deflation
   condition (`STARNEIG_SCHUR_LAPACK_THRESHOLD`).

### v0.1-beta.3:
 - Re-implemented Hessenberg reduction.

### v0.1-beta.2:
 - Fix an installation-related bug.
 - Fix a MPI-related compile error.
 - Remove unused code.

### v0.1-beta.1:
 - First beta release of the library.
