#
# Author: Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
#
# Copyright (c) 2019-2020, Umeå Universitet
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

project (StarNEig LANGUAGES C)
cmake_minimum_required (VERSION 3.3)

#
# options
#

option (STARNEIG_ENABLE_FULL_TESTS "Enable additional tests" OFF)
option (STARNEIG_ENABLE_REFERENCE "Enable reference solvers" OFF)

#
# includes
#

include (CheckLibraryExists)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckCCompilerFlag)

#
# compiler flag checks
#

function (enable_fortran_compiler_flag_if_supported flag)
    string(REPLACE "=" "_" flag_name ${flag})
    string (FIND "${CMAKE_Fortran_FLAGS}" "${flag}" flag_already_set)
    if (flag_already_set EQUAL -1)
        check_fortran_compiler_flag (
            "${flag}" Fortran_FLAG_DETECTED${flag_name})
        if (Fortran_FLAG_DETECTED${flag_name})
            set (CMAKE_Fortran_FLAGS
                "${flag} ${CMAKE_Fortran_FLAGS}" PARENT_SCOPE)
        endif ()
    endif ()
endfunction ()

#
# header file finder
#

function (find_header_file name header doc action)
    if (NOT DEFINED ${name} OR ${name} STREQUAL ${name}-NOTFOUND)
        if (DEFINED ENV{${name}})
            message (STATUS "Reading ${name} from the enviroment...")
            set (${name} $ENV{${name}})
        else ()
            message (STATUS "Looking for ${header} header file ...")
            find_path (
                ${name} ${header} PATHS ENV CPATH ENV C_INCLUDE_PATH DOC ${doc})
            if (NOT ${name} STREQUAL ${name}-NOTFOUND)
                message (STATUS "Found ${header} header file from ${${name}}")
            else ()
                message (${action} "Cannot find ${header} header file")
            endif ()
        endif ()
    endif ()
    if (NOT ${name} STREQUAL ${name}-NOTFOUND)
        set (${name} ${${name}} PARENT_SCOPE)
        set (CMAKE_REQUIRED_INCLUDES
            ${${name}} ${CMAKE_REQUIRED_INCLUDES} PARENT_SCOPE)
    endif ()
endfunction ()

#
# library finder
#

function (find_library_file name flag libname func doc action)
    if (NOT ${flag})
        if (DEFINED ENV{${name}})
            message (STATUS "Reading ${name} from the enviroment...")
            set (${name} $ENV{${name}})
        else ()
            message (STATUS "Looking for ${libname} library ...")
            find_library (${name} ${libname} PATHS ENV LIBRARY_PATH DOC ${doc})
        endif ()
        if (NOT ${name} STREQUAL ${name}-NOTFOUND)
            CHECK_LIBRARY_EXISTS (${${name}} ${func} "" ${flag})
        else ()
            set (${flag} FALSE)
        endif ()
    endif ()
    if (${flag})
        set (${name} ${${name}} PARENT_SCOPE)
        set (${flag} ${${flag}} PARENT_SCOPE)
        set (CMAKE_REQUIRED_LIBRARIES
            ${${name}} ${CMAKE_REQUIRED_LIBRARIES} PARENT_SCOPE)
    else ()
        message (${action} "Cannot find ${libname} library")
    endif ()
endfunction ()

#
# set compiler flags
#

#enable_c_compiler_flag_if_supported ("-std=gnu99")
enable_c_compiler_flag_if_supported ("-Wall")
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_c_compiler_flag_if_supported ("-O2")
    if (STARNEIG_ENABLE_OPTIMIZATION)
        enable_c_compiler_flag_if_supported ("-march=native")
        enable_c_compiler_flag_if_supported ("--tree-vectorize")
    else ()
        enable_c_compiler_flag_if_supported (-mtune=generic)
    endif ()
endif ()

#
# OpenMP
#

find_package (OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
endif()

#
# StarNEig library
#

find_package(PkgConfig REQUIRED)
pkg_search_module(STARNEIG REQUIRED starneig)

set (CMAKE_REQUIRED_INCLUDES ${STARNEIG_INCLUDE_DIRS} ${CMAKE_REQUIRED_INCLUDES})
set (CMAKE_REQUIRED_LIBRARIES ${STARNEIG_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})

# the StarNEig library must be compiled before it can be added to
# ${CMAKE_REQUIRED_LIBRARIES}

list (FIND CMAKE_REQUIRED_LIBRARIES starneig-pdgghrd BUILTIN_PDGGHRD)
if (BUILTIN_PDGGHRD GREATER -1)
    list (REMOVE_ITEM CMAKE_REQUIRED_LIBRARIES starneig-pdgghrd)
    set (ADD_BACK_LIBS starneig-pdgghrd ${starneig})
endif ()

list (REMOVE_ITEM CMAKE_REQUIRED_LIBRARIES starneig)
set (ADD_BACK_LIBS starneig ${starneig})

include_directories (${STARNEIG_INCLUDE_DIRS})
link_directories (${STARNEIG_LIBRARY_DIRS})
set (CMAKE_C_FLAGS "${STARNEIG_C_FLAGS} ${CMAKE_C_FLAGS}")

CHECK_SYMBOL_EXISTS (
    STARNEIG_ENABLE_MPI starneig/configuration.h STARNEIG_ENABLE_MPI)
CHECK_SYMBOL_EXISTS (
    STARNEIG_ENABLE_BLACS starneig/configuration.h STARNEIG_ENABLE_BLACS)
CHECK_SYMBOL_EXISTS (
    STARNEIG_ENABLE_CUDA starneig/configuration.h STARNEIG_ENABLE_CUDA)
CHECK_SYMBOL_EXISTS (
    STARNEIG_SEP_DM_HESSENBERG starneig/configuration.h
    STARNEIG_SEP_DM_HESSENBERG)
CHECK_SYMBOL_EXISTS (
    STARNEIG_GEP_DM_HESSENBERGTRIANGULAR starneig/configuration.h
    STARNEIG_GEP_DM_HESSENBERGTRIANGULAR)
CHECK_SYMBOL_EXISTS (
    STARNEIG_SEP_DM_REDUCE starneig/configuration.h STARNEIG_SEP_DM_REDUCE)
CHECK_SYMBOL_EXISTS (
    STARNEIG_GEP_DM_REDUCE starneig/configuration.h STARNEIG_GEP_DM_REDUCE)

#
# parallel BLAS
#

CHECK_FUNCTION_EXISTS (
    "mkl_set_num_threads_local" MKL_SET_NUM_THREADS_LOCAL_FOUND)
if (NOT MKL_SET_NUM_THREADS_LOCAL_FOUND)
    CHECK_FUNCTION_EXISTS (
        "openblas_set_num_threads" OPENBLAS_SET_NUM_THREADS_FOUND)
    if (NOT OPENBLAS_SET_NUM_THREADS_FOUND)
        CHECK_FUNCTION_EXISTS (
            "goto_set_num_threads" GOTO_SET_NUM_THREADS_FOUND)
    endif ()
endif ()

#
# ScaLAPACK subroutines
#

if (STARNEIG_ENABLE_BLACS)
    CHECK_FUNCTION_EXISTS ("pdgehrd_" PDGEHRD_FOUND)
    CHECK_FUNCTION_EXISTS ("pdormhr_" PDORMHR_FOUND)
    CHECK_FUNCTION_EXISTS ("pdlaset_" PDLASET_FOUND)
    CHECK_FUNCTION_EXISTS ("pdlahqr_" PDLAHQR_FOUND)
    CHECK_FUNCTION_EXISTS ("pdhseqr_" PDHSEQR_FOUND)
    CHECK_FUNCTION_EXISTS ("pdtrsen_" PDTRSEN_FOUND)
    CHECK_FUNCTION_EXISTS ("pdhgeqz_" PDHGEQZ_FOUND)
    CHECK_FUNCTION_EXISTS ("pdtgsen_" PDTGSEN_FOUND)
endif ()

#
# GSL
#

find_library_file (
    GSL_LIBRARIES GSL_FOUND gsl "gsl_rng_env_setup" "GSL library" STATUS)
find_header_file (GSL_INCLUDE_PATH gsl/gsl_rng.h "GSL include path" STATUS)

#
# MAGMA
#

if (STARNEIG_ENABLE_CUDA)
    find_library_file (
        MAGMA_LIBRARIES MAGMA_FOUND magma "magma_init" "MAGMA library" STATUS)
    find_header_file (MAGMA_INCLUDE_PATH magma.h "MAGMA include path" STATUS)
endif ()

#
# source files
#

file (GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/misc/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/reorder/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigenvectors/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/hessenberg/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/schur/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/matrixmarket/*.c")

# remove StarNEig pencil if MPI support is not present
if (NOT STARNEIG_ENABLE_MPI)
    list (REMOVE_ITEM SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/common/starneig_pencil.c)
endif ()

#
# custom pdhseqr and pdhgeqz
#

if (STARNEIG_ENABLE_BLACS AND STARNEIG_ENABLE_REFERENCE)
    set (CUSTOM_PDHSEQR TRUE)
    add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/custom_pdhseqr/)
    set (CMAKE_REQUIRED_LIBRARIES custom_pdhseqr ${CMAKE_REQUIRED_LIBRARIES})

    if (NOT PDHGEQZ_FOUND)
        set (PDHGEQZ_FOUND TRUE)
        add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pdhgeqz/)
        set (CMAKE_REQUIRED_LIBRARIES pdhgeqz ${CMAKE_REQUIRED_LIBRARIES})
    endif ()

    if (NOT PDTGSEN_FOUND)
        set (PDTGSEN_FOUND TRUE)
        add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pdtgsen/)
        set (CMAKE_REQUIRED_LIBRARIES pdtgsen ${CMAKE_REQUIRED_LIBRARIES})
    endif ()
endif ()

#
# configuration file
#

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/starneig_test_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/starneig_test_config.h")

#
# test binary
#

include_directories (${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_REQUIRED_INCLUDES})

add_executable (starneig-test ${SOURCES})
set_target_properties (starneig-test PROPERTIES LINKER_LANGUAGE C)
target_link_libraries (starneig-test
    ${ADD_BACK_LIBS} ${CMAKE_REQUIRED_LIBRARIES})

#
# simplified tests for the standard case
#

enable_testing ()

foreach (alg hessenberg schur reorder eigenvectors full-chain)

    set (extra_args)
    if (alg STREQUAL "reorder")
        set (extra_args "--fortify")
    endif ()

    add_test(
        NAME simple-${alg}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment ${alg}
            --n 5000 --solver starneig-simple --keep-going ${extra_args})

    if (STARNEIG_ENABLE_MPI AND NOT alg STREQUAL "eigenvectors" AND
    NOT (NOT STARNEIG_SEP_DM_HESSENBERG AND alg STREQUAL "hessenberg") AND
    NOT (NOT STARNEIG_SEP_DM_REDUCE AND alg STREQUAL "full-chain"))
        add_test(
            NAME simple-${alg}-mpi
            COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/starneig-test --mpi
                --experiment ${alg} --n 5000 --solver starneig-simple
                --cores 1 --gpus 0 --test-workers 1 --blas-threads 1
                --keep-going ${extra_args})
        set_property (TEST simple-${alg}-mpi
            PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
    endif ()
endforeach ()

#
# simplified tests for the generalized case
#

foreach (alg hessenberg schur reorder eigenvectors full-chain)

    set (extra_args)
    if (alg STREQUAL "reorder")
        set (extra_args --fortify)
    endif ()

    add_test(
        NAME simple-${alg}-generalized
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment ${alg}
            --n 5000 --generalized --solver starneig-simple --keep-going
            ${extra_args})

    if (STARNEIG_ENABLE_MPI AND NOT alg STREQUAL "eigenvectors" AND
    NOT (NOT STARNEIG_GEP_DM_HESSENBERGTRIANGULAR AND alg STREQUAL "hessenberg") AND
    NOT (NOT STARNEIG_GEP_DM_REDUCE AND alg STREQUAL "full-chain"))
        add_test(
            NAME simple-${alg}-generalized-mpi
            COMMAND mpirun -n 4 ${EXECUTABLE_OUTPUT_PATH}/starneig-test --mpi
                --experiment ${alg} --n 5000 --generalized
                --solver starneig-simple --cores 1 --gpus 0 --test-workers 1
                --blas-threads 1 --keep-going ${extra_args})
        set_property (TEST simple-${alg}-generalized-mpi
            PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
    endif ()
endforeach ()

if (STARNEIG_ENABLE_FULL_TESTS)

#
# parameters additional tests
#

set (BLUEPRINTS dsa dsb csa csb csc csd cse csf)
set (TILE_SIZES 48 549 611 883 448 340 526 197)
set (PANEL_WIDTHS 45 314 400 410 170 35 303)
set (AED_SIZES default 50 500 1000 2000)
set (RANKS 1 2 3 4)

#
# Hessenberg tests
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME hessenberg-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment hessenberg
            --n 4000 --tile-size ${ts})
endforeach ()

foreach (pwidth ${PANEL_WIDTHS})
    add_test(
        NAME hessenberg-panel-${pwidth}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment hessenberg
            --n 4000 --panel-width ${pwidth})
endforeach ()

if (STARNEIG_ENABLE_FULL_TESTS)
    foreach (n 47 88 333 554 3569)
        math(EXPR begin ${n}/4)
        math(EXPR end 3*${n}/4)
        add_test(
            NAME hessenberg-partial-${n}
            COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                --experiment partial-hessenberg --n ${n} --begin ${begin}
                --end ${end} --gpus 0)
        if (STARNEIG_ENABLE_CUDA)
            add_test(
                NAME hessenberg-partial-gpu-${n}
                COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --experiment partial-hessenberg --n ${n} --begin ${begin}
                    --end ${end})
        endif ()
    endforeach ()
endif ()

#
# Schur tests (standard)
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME schur-standard-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
        --n 4000 --tile-size ${ts})
endforeach ()

foreach (aed_size ${AED_SIZES})
    add_test(
        NAME schur-standard-sequential-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --n 4000 --aed-window-size ${aed_size}
            --aed-parallel-soft-limit 9999 --aed-parallel-hard-limit 9999)

    add_test(
        NAME schur-standard-decouple-sequential-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --n 4000 --aed-window-size ${aed_size}
            --aed-parallel-soft-limit 9999 --aed-parallel-hard-limit 9999
            --decouple 3)

    add_test(
        NAME schur-standard-parallel-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --n 4000 --aed-window-size ${aed_size} --aed-parallel-soft-limit 1
            --aed-parallel-hard-limit 1)

    add_test(
        NAME schur-standard-decouple-parallel-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --n 4000 --aed-window-size ${aed_size} --aed-parallel-soft-limit 1
            --aed-parallel-hard-limit 1 --decouple 3)
endforeach ()

if (STARNEIG_ENABLE_MPI)
    foreach (ranks ${RANKS})
        foreach (aed_size ${AED_SIZES})
            add_test(
                NAME schur-mpi-standard-${ranks}-sequential-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 9999
                    --aed-parallel-hard-limit 99999 --cores 1 --gpus 0
                    --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-standard-${ranks}-sequential-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)

            add_test(
                NAME schur-mpi-standard-${ranks}-decouple-sequential-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 9999
                    --aed-parallel-hard-limit 99999 --decouple 3
                    --cores 1 --gpus 0 --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-standard-${ranks}-decouple-sequential-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)

            add_test(
                NAME schur-mpi-standard-${ranks}-parallel-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 1
                    --aed-parallel-hard-limit 1 --cores 1 --gpus 0
                    --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-standard-${ranks}-parallel-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)

            add_test(
                NAME schur-mpi-standard-${ranks}-decouple-parallel-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 1
                    --aed-parallel-hard-limit 1 --decouple 3 --cores 1 --gpus 0
                    --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-standard-${ranks}-decouple-parallel-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
        endforeach ()
    endforeach ()
endif ()

#
# Schur tests (generalized)
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME schur-generalized-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
        --generalized --n 4000 --tile-size ${ts})
endforeach ()

foreach (aed_size ${AED_SIZES})
    add_test(
        NAME schur-generalized-sequential-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --generalized --n 4000 --aed-window-size ${aed_size}
            --aed-parallel-soft-limit 9999 --aed-parallel-hard-limit 9999)

    add_test(
        NAME schur-generalized-decouple-sequential-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --generalized --n 4000 --aed-window-size ${aed_size}
            --aed-parallel-soft-limit 9999 --aed-parallel-hard-limit 9999
            --decouple 3 --set-to-inf 100)

    add_test(
        NAME schur-generalized-parallel-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --generalized --n 4000 --aed-window-size ${aed_size}
            --aed-parallel-soft-limit 1 --aed-parallel-hard-limit 1)

    add_test(
        NAME schur-generalized-decouple-parallel-aed-${aed_size}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment schur
            --generalized --n 4000 --aed-window-size ${aed_size}
            --aed-parallel-soft-limit 1 --aed-parallel-hard-limit 1
            --decouple 3 --set-to-inf 100)
endforeach ()

if (STARNEIG_ENABLE_MPI)
    foreach (ranks ${RANKS})
        foreach (aed_size ${AED_SIZES})
            add_test(
                NAME schur-mpi-generalized-${ranks}-sequential-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --generalized --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 9999
                    --aed-parallel-hard-limit 99999 --cores 1 --gpus 0
                    --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-generalized-${ranks}-sequential-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)

            add_test(
                NAME schur-mpi-generalized-${ranks}-decouple-sequential-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --generalized --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 9999
                    --aed-parallel-hard-limit 99999 --decouple 3
                    --set-to-inf 100 --cores 1 --gpus 0 --test-workers 1
                    --blas-threads 1)
            set_property (
                TEST schur-mpi-generalized-${ranks}-decouple-sequential-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)

            add_test(
                NAME schur-mpi-generalized-${ranks}-parallel-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --generalized --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 1
                    --aed-parallel-hard-limit 1 --cores 1 --gpus 0
                    --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-generalized-${ranks}-parallel-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)

            add_test(
                NAME schur-mpi-generalized-${ranks}-decouple-parallel-aed-${aed_size}
                COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                    --mpi --experiment schur --generalized --n 4000
                    --aed-window-size ${aed_size} --aed-parallel-soft-limit 1
                    --aed-parallel-hard-limit 1 --decouple 3 --set-to-inf 100
                    --cores 1 --gpus 0 --test-workers 1 --blas-threads 1)
            set_property (
                TEST schur-mpi-generalized-${ranks}-decouple-parallel-aed-${aed_size}
                    PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
        endforeach ()
    endforeach ()
endif ()

#
# reordering tests (standard)
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME reorder-standard-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment reorder
            --n 4000 --fortify --tile-size ${ts})
endforeach ()

foreach (blueprint ${BLUEPRINTS})
    add_test(
        NAME reorder-standard-blueprint-${blueprint}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment reorder
        --n 4000 --blueprint ${blueprint} --fortify)
endforeach ()

if (STARNEIG_ENABLE_MPI)
    foreach (ranks ${RANKS})
        add_test(
            NAME reorder-mpi-${ranks}-standard
            COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                --mpi --experiment reorder --n 4000 --fortify
                --cores 1 --gpus 0 --test-workers 1 --blas-threads 1)
        set_property (TEST reorder-mpi-${ranks}-standard
            PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
    endforeach ()

    foreach (blueprint ${BLUEPRINTS})
        add_test(
            NAME reorder-mpi-standard-blueprint-${blueprint}
            COMMAND mpirun ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                --mpi --experiment reorder --n 4000 --fortify
                --blueprint ${blueprint} --cores 1 --gpus 0
                --test-workers 1 --blas-threads 1)
        set_property (TEST reorder-mpi-standard-blueprint-${blueprint}
            PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
    endforeach ()
endif ()

#
# reordering tests (generalized)
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME reorder-generalized-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment reorder
            --generalized --n 4000 --fortify --tile-size ${ts})
endforeach ()

foreach (blueprint ${BLUEPRINTS})
    add_test(
        NAME reorder-generalized-blueprint-${blueprint}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test --experiment reorder
        --generalized --n 4000 --blueprint ${blueprint} --fortify)
endforeach ()

if (STARNEIG_ENABLE_MPI)
    foreach (ranks ${RANKS})
        add_test(
            NAME reorder-mpi-${ranks}-generalized
            COMMAND mpirun -n ${ranks} ${EXECUTABLE_OUTPUT_PATH}/starneig-test
                --mpi --experiment reorder --n 4000 --generalized --fortify
                --cores 1 --gpus 0 --test-workers 1 --blas-threads 1)
        set_property (TEST reorder-mpi-${ranks}-generalized
            PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
    endforeach ()

    foreach (blueprint ${BLUEPRINTS})
        add_test(
            NAME reorder-mpi-generalized-blueprint-${blueprint}
            COMMAND mpirun ${EXECUTABLE_OUTPUT_PATH}/starneig-test --mpi
                --experiment reorder --n 4000 --generalized --fortify
                --blueprint ${blueprint} --cores 1 --gpus 0
                --test-workers 1 --blas-threads 1)
        set_property (TEST reorder-mpi-generalized-blueprint-${blueprint}
            PROPERTY ENVIRONMENT STARPU_WORKERS_NOBIND=1)
    endforeach ()
endif ()

#
# eigenvectors tests (standarg)
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME eigenvectors-standard-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
        --experiment eigenvectors --n 4000 --tile-size ${ts} --keep-going)
endforeach ()

add_test(
    NAME eigenvectors-standard-zeros
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
        --experiment eigenvectors --n 4000 --zero-ratio 0.5  --keep-going)

#
# eigenvectors tests (generalized)
#

foreach (ts ${TILE_SIZES})
    add_test(
        NAME eigenvectors-generalized-tile-size-${ts}
        COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
            --experiment eigenvectors --generalized --n 4000 --tile-size ${ts}
            --keep-going)
endforeach ()

add_test(
    NAME eigenvectors-generalized-zeros
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
        --experiment eigenvectors --generalized --n 4000 --zero-ratio 0.5
        --keep-going)

add_test(
    NAME eigenvectors-generalized-infs
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/starneig-test
        --experiment eigenvectors --generalized --n 4000 --inf-ratio 0.5
        --keep-going)

endif (STARNEIG_ENABLE_FULL_TESTS)
