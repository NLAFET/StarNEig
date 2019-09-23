#
# Author: Mirko Myllykoski (mirkom@cs.umu.se), Umeå University
#
# Copyright (c) 2019, Umeå Universitet
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

cmake_minimum_required (VERSION 3.3)

#
# supporter StarPU versions
#

set (SUPPORTED_STARPU 1.3 1.2)

#
# options
#

option (STARNEIG_ENABLE_OPTIMIZATION "Enable compiler optimizations" ON)

option (STARNEIG_DISABLE_MPI "Disable MPI support" OFF)
option (STARNEIG_DISABLE_CUDA "Disable CUDA support" OFF)
option (STARNEIG_DISABLE_BLACS "Disable BLACS support" OFF)

option (STARNEIG_ENABLE_VERBOSE "Enable all verbose messages" OFF)
option (STARNEIG_ENABLE_MESSAGES "Enable some verbose messages" ON)
option (STARNEIG_ENABLE_SANITY_CHECKS "Enable additional sanity checks" OFF)

option (STARNEIG_ENABLE_PRUNING
    "Enable task graph pruning" ON)
option (STARNEIG_ENABLE_MRM
    "Enable multiple linear regression performance models" OFF)
option (STARNEIG_ENABLE_CUDA_REORDER_WINDOW
    "Enable CUDA-based reorder_window codelet" OFF)
option (STARNEIG_ENABLE_INTEGER_SCALING
    "Enable integer-based scaling factors" ON)

#
# languages
#

enable_language (C Fortran)

#
# includes
#

include (GNUInstallDirs)
include (CheckLibraryExists)
include (CheckFunctionExists)
include (CheckSymbolExists)
include (CheckCCompilerFlag)
include (CheckFortranCompilerFlag)

#
# compiler flag checks
#

function (enable_c_compiler_flag_if_supported flag)
    string(REPLACE "=" "_" flag_name ${flag})
    string (FIND "${CMAKE_C_FLAGS}" "${flag}" flag_already_set)
    if (flag_already_set EQUAL -1)
        check_c_compiler_flag ("${flag}" C_FLAG_DETECTED${flag_name})
        if (C_FLAG_DETECTED${flag_name})
            set (CMAKE_C_FLAGS "${flag} ${CMAKE_C_FLAGS}" PARENT_SCOPE)
        endif ()
    endif ()
endfunction ()

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
            set (${name} ${${name}} PARENT_SCOPE)
        endif ()
    endif ()
    if (NOT ${name} STREQUAL ${name}-NOTFOUND)
        set (CMAKE_REQUIRED_INCLUDES
            ${${name}} ${CMAKE_REQUIRED_INCLUDES} PARENT_SCOPE)
    endif ()
endfunction ()

#
# library finder
#

function (find_library_file name flag libname func doc action)
    if (NOT DEFINED ${name} OR NOT DEFINED ${flag} OR
    ${name} STREQUAL ${name}-NOTFOUND)
        if (DEFINED ENV{${name}})
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
        set (${name} ${${name}} PARENT_SCOPE)
        set (${flag} ${${flag}} PARENT_SCOPE)
    endif ()
    if (${flag})
        set (CMAKE_REQUIRED_LIBRARIES
            ${${name}} ${CMAKE_REQUIRED_LIBRARIES} PARENT_SCOPE)
    else ()
        message (${action} "Cannot find ${libname} library")
    endif ()
endfunction ()

#
# symbol visibility
#

cmake_policy (SET CMP0063 NEW)
set (CMAKE_C_VISIBILITY_PRESET hidden)

#
# library version
#

file (STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../version.txt" STARNEIG_VERSION)

#
# set compiler flags
#

#enable_c_compiler_flag_if_supported ("-std=gnu99")
enable_c_compiler_flag_if_supported ("-Wall")
if (STARNEIG_ENABLE_OPTIMIZATION AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        enable_c_compiler_flag_if_supported ("-march=native")
        enable_c_compiler_flag_if_supported ("-O2")
        enable_c_compiler_flag_if_supported ("--tree-vectorize")
        enable_fortran_compiler_flag_if_supported ("-march=native")
        enable_fortran_compiler_flag_if_supported ("-O2")
        enable_fortran_compiler_flag_if_supported ("--tree-vectorize")
    endif ()
endif ()

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    find_library_file (
        MATH_LIBRARIES MATH_FOUND m "sqrt" "Standard math library" FATAL_ERROR)
endif ()

#
# pthreads library
#

set (CMAKE_THREAD_PREFER_PTHREAD ON)
set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
set (CMAKE_REQUIRED_LIBRARIES
    ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_REQUIRED_LIBRARIES})

#
# OpenMP
#

find_package (OpenMP REQUIRED)
set (CMAKE_REQUIRED_LIBRARIES
    ${OpenMP_C_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
set (CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
#set (CMAKE_EXE_LINKER_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
find_header_file (OMP_INCLUDE_PATH omp.h "OpenMP include path" STATUS)

#
# fortran runtime
#

get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
    set (CMAKE_REQUIRED_LIBRARIES gfortran ${CMAKE_REQUIRED_LIBRARIES})
endif ()

#
# BLAS
#

if (DEFINED ENV{BLAS_LIBRARIES})
    find_library_file (BLAS_LIBRARIES BLAS_FOUND
        blas "dgemm_" "BLAS library" FATAL_ERROR)
else ()
    find_package (BLAS REQUIRED)
    set (CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
endif ()

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

find_header_file (BLAS_INCLUDE_PATH cblas.h "BLAS include path" FATAL_ERROR)

if (MKL_SET_NUM_THREADS_LOCAL_FOUND)
    find_header_file (MKL_INCLUDE_PATH mkl.h "MKL include path" FATAL_ERROR)
endif ()

#
# LAPACK
#

if (DEFINED ENV{LAPACK_LIBRARIES} OR DEFINED ENV{LAPACK_LIBRARIES})
    find_library_file (LAPACK_LIBRARIES LAPACK_FOUND
        lapack "dtrsen_" "LAPACK library" FATAL_ERROR FATAL_ERROR)
else ()
    find_package (LAPACK REQUIRED)
    set (CMAKE_REQUIRED_LIBRARIES
        ${LAPACK_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
endif ()

#
# StarPU include paths
#

if (NOT DEFINED STARPU_INCLUDE_PATH)
    if (DEFINED ENV{STARPU_INCLUDE_PATH})
        set (STARPU_INCLUDE_PATH $ENV{STARPU_INCLUDE_PATH}
            CACHE PATH "StarPU include path")
    else ()
        foreach (VERSION ${SUPPORTED_STARPU})
            message (STATUS "Looking for StarPU ${VERSION} header files...")
            find_path (SEARCH_PATH starpu/${VERSION}/starpu.h
                PATHS ENV CPATH ENV C_INCLUDE_PATH)
            if (NOT SEARCH_PATH STREQUAL SEARCH_PATH-NOTFOUND)
                set (STARPU_VERSION ${VERSION})
                set (STARPU_INCLUDE_PATH
                    "${SEARCH_PATH}/starpu/${STARPU_VERSION}"
                    CACHE PATH "StarPU include path")
                message (STATUS
                    "Found StarPU ${STARPU_VERSION} header files from "
                    "${STARPU_INCLUDE_PATH}")
                break ()
            endif ()
        endforeach ()
        unset (SEARCH_PATH CACHE)

        if (NOT STARPU_VERSION)
            message (FATAL_ERROR "StarPU header files not found")
        endif ()
    endif ()
endif ()
set (CMAKE_REQUIRED_INCLUDES ${STARPU_INCLUDE_PATH} ${CMAKE_REQUIRED_INCLUDES})

#
# Hardware Locality (hwloc)
#

find_library_file (HWLOC_LIBRARIES HWLOC_FOUND
    hwloc "hwloc_get_api_version" "hwloc library" FATAL_ERROR)
find_header_file (HWLOC_INCLUDE_PATH hwloc.h "hwloc include path" FATAL_ERROR)

#
# MPI
#

if (NOT STARNEIG_DISABLE_MPI)
    CHECK_SYMBOL_EXISTS (STARPU_USE_MPI starpu.h STARPU_USE_MPI)
endif ()
if (STARPU_USE_MPI)
    if (DEFINED ENV{MPI_LIBRARIES})
        find_library_file (
            MPI_LIBRARIES MPI_C_FOUND mpi "MPI_Init" "MPI library" FATAL_ERROR)
        find_library_file (
            MPI_Fortran_LIBRARIES MPI_Fortran_FOUND mpi "mpi_init_"
            "MPI Fortran library" STATUS)
        find_header_file (MPI_INCLUDE_PATH mpi.h "MPI include path" FATAL_ERROR)
    else ()
        find_package (MPI REQUIRED)

        set (CMAKE_REQUIRED_INCLUDES
            ${MPI_C_INCLUDE_PATH} ${CMAKE_REQUIRED_INCLUDES})

        set (CMAKE_REQUIRED_LIBRARIES
            ${MPI_C_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})

        if (MPI_Fortran_FOUND)
            set (CMAKE_REQUIRED_LIBRARIES
                ${MPI_Fortran_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
        endif ()
    endif ()
endif ()

#
# ScaLAPACK
#

if (NOT STARNEIG_DISABLE_BLACS AND MPI_Fortran_FOUND)
    set (SUPPORTED_SCALAPACK scalapack scalapack-openmpi libscalapack-mpi-dev)
    foreach (lib ${SUPPORTED_SCALAPACK})
        find_library_file (SCALAPACK_LIBRARIES SCALAPACK_FOUND
            ${lib} "pdgesv_" "ScaLAPACK library" STATUS)
        if (SCALAPACK_FOUND)
            break ()
        endif ()
    endforeach ()
endif ()

#
# BLACS
#

if (NOT STARNEIG_DISABLE_BLACS AND SCALAPACK_FOUND)
    CHECK_FUNCTION_EXISTS ("blacs_pinfo_" BLACS_FOUND)
    if (NOT BLACS_FOUND)
        find_library_file (BLACS_LIBRARIES BLACS_FOUND
            blacs-openmpi "blacs_pinfo_" "BLACS library" STATUS)
    endif ()
endif ()

#
# CUDA
#

if (NOT STARNEIG_DISABLE_CUDA)
    CHECK_SYMBOL_EXISTS (STARPU_USE_CUDA starpu.h STARPU_USE_CUDA)
endif ()
if (STARPU_USE_CUDA)
    enable_language (CXX)
    set (LINKER_LANGUAGE CXX)
    if (NOT (CMAKE_MAJOR_VERSION LESS 4 AND CMAKE_MINOR_VERSION LESS 8))
        cmake_minimum_required (VERSION 3.8)
        enable_language (CUDA)
        set (CMAKE_REQUIRED_INCLUDES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${CMAKE_REQUIRED_INCLUDES})
        set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}
            ${CMAKE_REQUIRED_LIBRARIES})
    else ()
        find_package (CUDA REQUIRED)

        if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
        CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
            set (OLD_CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER})
            unset (CUDA_HOST_COMPILER CACHE)

            if (CUDA_VERSION_MAJOR VERSION_LESS 9 AND
            CMAKE_C_COMPILER_VERSION VERSION_GREATER 5.99)
                message (STATUS
                    "Trying to switch CUDA compiler to gcc 5 series ...")
                find_program (CUDA_HOST_COMPILER gcc-5)
            elseif (CUDA_VERSION_MAJOR VERSION_LESS 10 AND
            CMAKE_C_COMPILER_VERSION VERSION_GREATER 6.99)
                message (STATUS
                    "Trying to switch CUDA compiler to gcc 6 series ...")
                find_program (CUDA_HOST_COMPILER gcc-6 gcc-5)
            elseif (CUDA_VERSION_MAJOR VERSION_LESS 11 AND
            CMAKE_C_COMPILER_VERSION VERSION_GREATER 7.99)
                message (STATUS
                    "Trying to switch CUDA compiler to gcc 7 series ...")
                find_program (CUDA_HOST_COMPILER gcc-7 gcc-6 gcc-5)
            endif ()

            if (NOT CUDA_HOST_COMPILER STREQUAL OLD_CUDA_HOST_COMPILER)
                message (STATUS "Switched to ${CUDA_HOST_COMPILER}")
                set (CUDA_PROPAGATE_HOST_FLAGS OFF)
            else ()
                message (WARNING "Could not switch to gcc 5/6 series")
                set (CUDA_HOST_COMPILER ${OLD_CUDA_HOST_COMPILER})
            endif ()
        endif ()

        set (CMAKE_REQUIRED_INCLUDES ${CUDA_INCLUDE_DIRS}
            ${CMAKE_REQUIRED_INCLUDES})
        set (CMAKE_REQUIRED_LIBRARIES ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES}
            ${CMAKE_REQUIRED_LIBRARIES})

        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
        CUDA_INCLUDE_DIRECTORIES (
            ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/include/"
            "${CMAKE_CURRENT_SOURCE_DIR}/include" ${CMAKE_REQUIRED_INCLUDES})
    endif ()
else ()
    set (LINKER_LANGUAGE C)
endif ()

#
# StarPU
#

find_library_file (STARPU_LIBRARIES_BASE STARPU_BASE_FOUND
    starpu-${STARPU_VERSION} "starpu_init" "StarPU base library" FATAL_ERROR)

#
# StarPU-MPI
#

if (STARPU_USE_MPI)
    find_library_file (STARPU_LIBRARIES_MPI STARPU_MPI_FOUND
        starpumpi-${STARPU_VERSION} "starpu_mpi_init" "StarPU-MPI library"
        FATAL_ERROR)
endif ()

#
# Miscellaneous libraries and headers
#

set (CMAKE_REQUIRED_INCLUDES ${MISC_LIBRARIES} ${CMAKE_REQUIRED_INCLUDES})
set (CMAKE_REQUIRED_LIBRARIES ${MISC_INCLUDE_PATH} ${CMAKE_REQUIRED_LIBRARIES})

#
# header files
#

set (STARNEIG_ENABLE_MPI ${STARPU_USE_MPI})
set (STARNEIG_ENABLE_CUDA ${STARPU_USE_CUDA})
if (STARPU_USE_MPI AND BLACS_FOUND)
    set (STARNEIG_ENABLE_BLACS TRUE)
    if (SCALAPACK_FOUND)
        set (STARNEIG_ENABLE_SCALAPACK TRUE)
        set (STARNEIG_SEP_DM_HESSENBERG TRUE)
        set (STARNEIG_GEP_DM_HESSENBERGTRIANGULAR TRUE)
        set (STARNEIG_SEP_DM_REDUCE TRUE)
        set (STARNEIG_GEP_DM_REDUCE TRUE)
    endif ()
endif ()

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/configuration.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/starneig/configuration.h" @ONLY)

if (NOT STARNEIG_ENABLE_MPI)
    set (COMMENT_DM "// ")
endif ()

if (NOT STARNEIG_ENABLE_BLACS)
    set (COMMENT_BLACS "// ")
endif ()

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/starneig.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/starneig/starneig.h" @ONLY)

set (PUBLIC_HEADER_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/starneig/configuration.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/error.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/expert.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/gep_sm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/node.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/sep_sm.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/starneig/starneig.h)

if (STARNEIG_ENABLE_MPI)
    set (PUBLIC_HEADER_FILES ${PUBLIC_HEADER_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/distr_matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/gep_dm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/sep_dm.h)
endif ()

if (STARNEIG_ENABLE_BLACS)
    set (PUBLIC_HEADER_FILES ${PUBLIC_HEADER_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/blacs_helpers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/starneig/blacs_matrix.h)
endif ()

#
# configuration file
#

CHECK_FUNCTION_EXISTS (aligned_alloc ALIGNED_ALLOC_FOUND)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/starneig_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/starneig_config.h")

#
# source files
#

file (GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/hessenberg/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/reorder/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/schur/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigenvectors/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/common.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/lapack.c")

# compile ScaLAPACK wrappers only when ScaLAPACK and BLACS support are present
if (STARNEIG_ENABLE_SCALAPACK)
    set (SOURCES ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/wrappers/scalapack.c)
endif ()

# MPI support
if (STARNEIG_ENABLE_MPI)
    file (GLOB_RECURSE MPI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mpi/*.c")

    # compile BLACS support only when BLACS is found
    if (NOT STARNEIG_ENABLE_BLACS)
        list (REMOVE_ITEM MPI_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/mpi/blacs_helpers.c)
        list (REMOVE_ITEM MPI_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/mpi/blacs_matrix.c)
    endif ()

    set (SOURCES ${SOURCES} ${MPI_SOURCES})
endif ()

# built-in pdgghrd
if (STARNEIG_GEP_DM_HESSENBERGTRIANGULAR)
    CHECK_FUNCTION_EXISTS ("pdgghrd_" PDGGHRD_FOUND)
    if (NOT PDGGHRD_FOUND)
        file (GLOB_RECURSE PDGGHRD_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/pdgghrd/*.f"
            "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/pdgghrd/*.f90")
        set (SOURCES ${SOURCES} ${PDGGHRD_SOURCES})
    endif ()
endif ()

#
# CUDA source files
#

if (STARPU_USE_CUDA)
    file (GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
    if (NOT (CMAKE_MAJOR_VERSION LESS 4 AND CMAKE_MINOR_VERSION LESS 8))
        set (SOURCES ${SOURCES} ${CUDA_SOURCES})
    else ()
        cuda_compile (CUDA_FILES SHARED ${CUDA_SOURCES})
        set (SOURCES ${SOURCES} ${CUDA_FILES})
    endif ()
endif ()

#
# library
#

add_library (starneig SHARED ${SOURCES})

list (REMOVE_DUPLICATES CMAKE_REQUIRED_INCLUDES)
target_include_directories (starneig
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include/
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_REQUIRED_INCLUDES})

target_link_libraries (starneig PRIVATE ${CMAKE_REQUIRED_LIBRARIES})

set_target_properties (starneig PROPERTIES
LINKER_LANGUAGE ${LINKER_LANGUAGE} VERSION ${STARNEIG_VERSION})

#
# pkg-config
#

set (PKGCONFIG_DEST_DIR "${CMAKE_INSTALL_PREFIX}")

set (PKGCONFIG_LIBS "-lstarneig")

foreach (LIB ${CMAKE_REQUIRED_LIBRARIES})
    string (REGEX MATCH "^-.*" NOT_LIB ${LIB})
    if (NOT_LIB)
        set (PKGCONFIG_PRIVATE_LIBS "${PKGCONFIG_PRIVATE_LIBS} ${LIB}")
    else ()
        set (PKGCONFIG_PRIVATE_LIBS "${PKGCONFIG_PRIVATE_LIBS} -l${LIB}")
    endif ()
endforeach ()

set (PKGCONFIG_PRIVATE_LIBS
    "${PKGCONFIG_PRIVATE_LIBS} ${CMAKE_EXE_LINKER_FLAGS}")

foreach (DIR ${CMAKE_REQUIRED_INCLUDES})
    set (PKGCONFIG_INCLUDES "${PKGCONFIG_INCLUDES} -I${DIR}")
endforeach ()

#set (PKGCONFIG_FLAGS "${CMAKE_C_FLAGS}")

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/starneig.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/starneig.pc" @ONLY)

#
# installation
#

install (TARGETS starneig LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install (
    FILES ${PUBLIC_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/starneig/)

set (INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/starneig.pc
    DESTINATION "${INSTALL_PKGCONFIG_DIR}")
