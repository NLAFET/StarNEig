# Installation

Prebuild binary packages (Ubuntu 18.04 and Ubuntu 20.04) are available under
[releases](https://github.com/NLAFET/StarNEig/releases) and can be installed
with the following command:

```
$ sudo apt install ./StarNEig-v0.xx.yy-ubuntu-vv.uu.deb
```

The binary packages rely on mainstream StarPU packages and do not necessary
provide full functionality.

For full functionality, it is recommended that StarNEig (and StarPU) are
compiled from the source code, see below and/or the StarNEig User's Guide.
Please consider using one of the tested
[release](https://github.com/NLAFET/StarNEig/releases) versions.

## Dependencies

Library dependencies:

 - Linux
 - CMake 3.3 or newer
 - Portable Hardware Locality (hwloc)
 - Starpu 1.2 or 1.3
    - Newer versions require the user set the `STARPU_LIBRARIES`,
      `STARPU_MPI_LIBRARIES` and `STARPU_INCLUDE_PATH` environmental variables.
 - OpenBLAS, MKL, GotoBLAS or single-threaded BLAS library
 - LAPACK
 - MPI (optional)
 - CUDA (optional)
 - ScaLAPACK + BLACS (optional)

Test program and example code dependencies:

 - pkg-config
 - GNU Scientific Library (optional)
 - MAGMA (optional)

@attention StarNEig supports OpenBLAS, MKL and GotoBLAS. For optimal
performance, a multi-threaded variant of one of the listed BLAS libraries must
be provided. StarNEig will automatically set the BLAS library to single-threaded
more when necessary. If a different BLAS library is provided, then the user is
responsible for setting the BLAS library to *single-threaded* mode. However, the
use of a non-supported BLAS library can still impact the performance negatively.

### StarPU 1.3.4 installation

 1. Download StarPU 1.3.4 (or newer) from http://starpu.gforge.inria.fr/files/
 2. Unzip the package and create/enter directory `starpu-1.3.4/build`
 3. Configure: `$ ../configure`
 4. Compile: `$ make`
 5. Install: `$ sudo make install`

The default installation path is `/usr/local` but this can be changed during the
configuration phase (`$ ../configure --prefix=...`). It is something necessary
to append the `CPATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` environmental
variables by adding the following to `~/.profile`:
```
export CPATH=$CPATH:/usr/local/include/
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

See the StarPU handbook for further instructions:
http://starpu.gforge.inria.fr/doc/html/BuildingAndInstallingStarPU.html

## Configuration

It is recommended that a user builds the library in a separate build directory:
```
$ cd path_to_the_top_directory/
$ mkdir build
$ cd build
```

The library is configured with the `cmake` command. In most cases, it is not
necessary to give this command any additional arguments:
```
$ cmake ../
...
-- Configuring done
-- Generating done
-- Build files have been written to: /.../build
```

However, the library can be customized with various options. For example, the
example codes and documentation generation can be enabled by setting the
`STARNEIG_ENABLE_EXAMPLES` and `STARNEIG_ENABLE_DOCS` options:
```
$ cmake -DSTARNEIG_ENABLE_EXAMPLES=ON -DSTARNEIG_ENABLE_DOCS=ON ../
```
The *installation path* can be changed during the configuration phase:
```
$ cmake -DCMAKE_INSTALL_PREFIX=/path/to/somewhere/ ../
```

@remark The library can be compiled separately from the other software
components:
```
$ mkdir build-src
$ cd build-src/
$ cmake ../src/
$ make
```

@remark It may sometimes be necessary to compile CUDA source files with a
different compiler than what `cmake` uses by default. For example, some CUDA
version do not support GCC compilers that are newer than GCC 5 release series.
In that case `cmake` can be configured to use a GCC 5 release series compiler:
```
$ cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-5 -DCUDA_PROPAGATE_HOST_FLAGS=OFF ../
```

List of StarNEig library specific configuration options:

 - `STARNEIG_ENABLE_OPTIMIZATION`: Enables extra compiler optimizations (`ON` by
   default).
    - Can break portability.

 - `STARNEIG_ENABLE_EXAMPLES`: Enables examples (`OFF` by default).
 - `STARNEIG_ENABLE_DOCS`: Enables documentation generation (`OFF` by default).
 - `STARNEIG_ENABLE_TESTS`: : Enables test program (`ON` by default).
 - `STARNEIG_ENABLE_FULL_TESTS`: Enables additional tests (`OFF` by default).
 - `STARNEIG_ENABLE_REFERENCE`: : Enables reference MPI implementations (`OFF`
   by default).
    - A user must initialize and update the included GIT submodules.

 - `STARNEIG_DISABLE_MPI`: Explicitly disables the MPI support even when the
   system would support it (`OFF` by default).
    - Must be set before `cmake` is run for the first time.
 - `STARNEIG_DISABLE_CUDA`: Explicitly disables the CUDA support even when the
   system would support it (`OFF` by default).
    - Must be set before `cmake` is run for the first time.
 - `STARNEIG_DISABLE_BLACS`: Explicitly disables the ScaLAPACK/BLACS support
   even when the system would support it (`OFF` by default).
    - Must be set before `cmake` is run for the first time.

 - `STARNEIG_ENABLE_MESSAGES`: Enable basic verbose messages (`ON` by default).
 - `STARNEIG_ENABLE_VERBOSE`: Enable additional verbose messages (`OFF` by
   default).
 - `STARNEIG_ENABLE_EVENTS`: Enable event traces (`OFF` by default).
 - `STARNEIG_ENABLE_EVENT_PARSER`: Enable event parser (`OFF` by default).
 - `STARNEIG_ENABLE_SANITY_CHECKS`: Enables additional satiny checks. (`OFF` by
   default).
    - These checks are very expensive and should not be enabled unless
      absolutely necessary.

 - `STARNEIG_ENABLE_PRUNING`: Enable task graph pruning (`ON` by default).
 - `STARNEIG_ENABLE_MRM`: Enable multiple linear regression performance models
   (`OFF` by default).
 - `STARNEIG_ENABLE_CUDA_REORDER_WINDOW`: Enable CUDA-based reorder_window
   codelet (`OFF` by default).
 - `STARNEIG_ENABLE_INTEGER_SCALING`: Enable integer-based scaling factors
   (`ON` by default).

The following **environmental variables** can be used to configure the used
libraries:

 - `BLAS_LIBRARIES`: BLAS library.
 - `LAPACK_LIBRARIES`: LAPACK library.
 - `HWLOC_LIBRARIES`: Portable Hardware Locality (hwloc) library.
 - `MPI_LIBRARIES`: C MPI library.
 - `MPI_Fortran_LIBRARIES`: Fortran MPI library.
 - `SCALAPACK_LIBRARIES`: ScaLAPACK library.
 - `BLACS_LIBRARIES`: BLACS library.
 - `STARPU_LIBRARIES`: StarPU library.
 - `STARPU_MPI_LIBRARIES`: StarPU-MPI library.
 - `GSL_LIBRARIES`: GNU Scientific Library.
 - `MAGMA_LIBRARIES`: MAGMA library.
 - `MISC_LIBRARIES`: Miscellaneous libraries.

For example, if a user has a custom build ATLAS BLAS library and a matching
LAPACK library that are not detected by the build system, then the user might
define `BLAS_LIBRARIES=/usr/local/atlas/lib/libsatlas.so` and
`LAPACK_LIBRARIES=/usr/local/atlas/lib/liblapack.so` before calling `cmake`.

The following environmental variables can be used to configure include paths for
the used libraries:

 - `OMP_INCLUDE_PATH`: OpenMP include path.
 - `BLAS_INCLUDE_PATH`: BLAS include path.
 - `MKL_INCLUDE_PATH`: MKL include path.
 - `HWLOC_INCLUDE_PATH`: Portable Hardware Locality (hwloc) include path.
 - `MPI_INCLUDE_PATH`: MPI include path.
 - `STARPU_INCLUDE_PATH`: StarPU include path.
 - `GSL_INCLUDE_PATH`: GNU Scientific Library include path.
 - `MAGMA_INCLUDE_PATH`: MAGMA include path.
 - `MISC_INCLUDE_PATH`: Miscellaneous include paths.

## Compile

The library (and other software components) are compiled with the `make`
command:
```
$ make
...
[ 99%] Building C object test/CMakeFiles/starneig-test.dir/3rdparty/matrixmarket/mmio.c.o
[100%] Linking C executable ../starneig-test
[100%] Built target starneig-test
```

## Test

The automated tests can be executed as follows:
```
$ make test
Running tests...
Test project /.../build
      Start  1: simple-hessenberg
 1/18 Test  #1: simple-hessenberg ...................   Passed   15.19 sec
      Start  2: simple-hessenberg-mpi
 2/18 Test  #2: simple-hessenberg-mpi ...............   Passed   49.59 sec
...
      Start 17: simple-full-chain-generalized
17/18 Test #17: simple-full-chain-generalized .......   Passed  180.50 sec
      Start 18: simple-full-chain-generalized-mpi
18/18 Test #18: simple-full-chain-generalized-mpi ...   Passed  195.39 sec

100% tests passed, 0 tests failed out of 18

Total Test time (real) = 1219.47 sec
```

The `STARNEIG_ENABLE_FULL_TESTS` `cmake` option can be used to enable additional
tests.

## Install

The library and the related header files are installed by executing:
```
$ sudo make install
```
