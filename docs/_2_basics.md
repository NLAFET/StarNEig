# Basic usage

## Header files

The principal interface functions can be found from the @ref starneig/node.h
header file. The library provides separate header files for shared memory
(@ref starneig/sep_sm.h, @ref starneig/gep_sm.h) and distributed memory
(@ref starneig/sep_dm.h, @ref starneig/gep_dm.h). In most cases, a user can
simply include all header files as follows:
@code{.c}
#include <starneig/starneig.h>
@endcode

Certain header files and interface functions exist only when the library
is compiled with MPI and ScaLAPACK / BLACS support. The configuration of the
installed library can be found from the @ref starneig/configuration.h header
file. See module @ref starneig_conf for further information.

## Library initialization

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
starneig_node_init(STARNEIG_USE_ALL, STARNEIG_USE_ALL, STARNEIG_DEFAULT);
@endcode
This tells the library to use all available CPU cores and GPUs.

See module @ref starneig_node for further information.

## Error handling

Most interface functions return one of the following values:

 - @ref STARNEIG_SUCCESS (0): The interface function was executed successfully.
 - A negative number `-i`: The `i`'th interface function argument was invalid.
 - A positive number `i`: The interface function encountered an error or a
   warning was raised. See module @ref starneig_error for further information.

All return values (@ref starneig_error_t) are defined in the
@ref starneig/error.h header file.

@remark The library may call the `exit()` and `abort()` functions if an
interface function encounters a fatal error from which it cannot recover.

## Performance models

The StarPU performance models must be calibrated before the software can
function efficiently on heterogeneous platforms (CPUs + GPUs). The calibration
is triggered automatically if the models are not calibrated well enough for a
given problem size. This may impact the execution time negatively during the
first run. Please see the StarPU handbook for further information:
http://starpu.gforge.inria.fr/doc/html/Scheduling.html

## Compilation and linking

During compilation, the `starneig` library library must be linked with the
user's software:
```
$ gcc -o my_program my_program.c -lstarneig
```

StarNEig provides a pkg-config file for easing the compilation. A user may
integrate it to their `Makefile`:
```
CFLAGS  += $$(pkg-config --cflags starneig)
LDLIBS  += $$(pkg-config --libs starneig)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

my_program: my_program.c
	$(CC) -o $@ $^ $(LDLIBS)
```
Or their `CMakeLists.txt` file:
```
find_package(PkgConfig REQUIRED)
pkg_search_module(STARNEIG REQUIRED starneig)

include_directories (${STARNEIG_INCLUDE_DIRS})
link_directories (${STARNEIG_LIBRARY_DIRS})
set (CMAKE_C_FLAGS "${STARNEIG_C_FLAGS} ${CMAKE_C_FLAGS}")

add_executable (my_program my_program.c)
target_link_libraries (my_program ${STARNEIG_LIBRARIES})
```
