## Changelog

### v0.1.7
 - Fix a compile error with GCC 10

### v0.1.6
 - Minor improvements to the Hessenberg reduction step

### v0.1.5
 - Disable OpenCL workers during initialization.
 - Add `STARNEIG_USE_ALL`.

### v0.1.4
 - Add deb packages.

### v0.1.3
 - Restore older Hessenberg reduction implementation from v0.1-beta2
 - Rename `aed_shift_count` parameter as `shift_count`. Rename the default value
   `STARNEIG_SCHUR_DEFAULT_AED_SHIFT_COUNT` as
   `STARNEIG_SCHUR_DEFAULT_SHIFT_COUNT`.
 - Improved performance models.

### v0.1.2
 - Improved performance models.
 - Updates to the documentation.
 - Rename `STARPU_LIBRARIES_BASE` and `STARPU_LIBRARIES_MPI` environmental
   variables as `STARPU_LIBRARIES` and `STARPU_MPI_LIBRARIES`, respectively.

### v0.1.1:
 - Fix a bug that may cause the code to hang in distributed memory  Schur
   reduction.

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
