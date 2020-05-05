# Initialization and shutdown

The initialization and shutdown interface functions can be found from the
@ref starneig/node.h header file. The library provides separate header files for
shared memory (@ref starneig/sep_sm.h, @ref starneig/gep_sm.h) and distributed
memory (@ref starneig/sep_dm.h, @ref starneig/gep_dm.h).  However, a user may
simply include all header files as follows:
@code{.c}
#include <starneig/starneig.h>
@endcode

Certain header files and interface functions exist only when the library
is compiled with MPI and ScaLAPACK / BLACS support. The configuration of the
installed library can be found from the @ref starneig/configuration.h header
file. See module @ref starneig_conf for further information.

Each node must call the starneig_node_init() interface function to initialize
the library and the starneig_node_finalize() interface function to shutdown the
library:
@code{.c}
starneig_node_init(cores, gpus, flags);

...

starneig_node_finalize();
@endcode

The starneig_node_init() interface function initializes StarPU (and cuBLAS) and
pauses all worker threads. The `cores` argument specifies the total number of
used CPU cores. In distributed memory mode, one of these CPU cores is
automatically allocated for the StarPU-MPI communication thread. The `gpus`
argument specifies the total number of used GPUs. One or more CPU cores are
automatically allocated for GPU devices. The `flags` (@ref starneig_flag_t)
argument can provide additional configuration information.

A node can also be configured with default values:
@code{.c}
starneig_node_init(-1, -1, STARNEIG_DEFAULT);
@endcode
This tells the library to use all available CPU cores and GPUs. See module
@ref starneig_node for further information.

Most interface functions return one of the following values:

 - @ref STARNEIG_SUCCESS (0): The interface function was executed successfully.
 - A negative number `-i`: The `i`'th interface function argument was invalid.
 - A positive number `i`: The interface function encountered an error or a
   warning was raised. See module @ref starneig_error for further information.

All return values (@ref starneig_error_t) are defined in the
@ref starneig/error.h header file.

@remark StarNEig supports OpenBLAS, MKL and GotoBLAS. For optimal performance,
a multi-threaded variant of one of the listed BLAS libraries must be provided.
StarNEig will automatically set the BLAS library to single-threaded more when
necessary. If a different BLAS library is provided, then the user is responsible
for setting the BLAS library to *single-threaded* mode. However, the use of a
non-supported BLAS library can still impact the performance negatively.

@remark The library may call the `exit()` and `abort()` functions if an
interface function encounters a fatal error from which it cannot recover.

@remark The StarPU performance models must be calibrated before the software can
function efficiently on heterogeneous platforms (CPUs + GPUs). The calibration
is triggered automatically if the models are not calibrated well enough for a
given problem size. This may impact the execution time negatively during the
first run. Please see the StarPU handbook for further information:
http://starpu.gforge.inria.fr/doc/html/Scheduling.html
