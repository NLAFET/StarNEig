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

project (StarNEig LANGUAGES Fortran)
cmake_minimum_required (VERSION 3.3)

include (CheckFortranCompilerFlag)

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
# library version
#

file (STRINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../version.txt" STARNEIG_VERSION)

enable_fortran_compiler_flag_if_supported ("-Wall")
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    enable_fortran_compiler_flag_if_supported ("-Wno-maybe-uninitialized")
#    enable_fortran_compiler_flag_if_supported ("-Wno-unused-label")
#    enable_fortran_compiler_flag_if_supported ("-Wno-unused-variable")
#    enable_fortran_compiler_flag_if_supported ("-Wno-unused-dummy-argument")
#    enable_fortran_compiler_flag_if_supported ("-Wno-character-truncation")
#    enable_fortran_compiler_flag_if_supported ("-Wno-conversion")
#    enable_fortran_compiler_flag_if_supported ("-Wno-tabs")

    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        enable_fortran_compiler_flag_if_supported ("-O2")
        if (STARNEIG_ENABLE_OPTIMIZATION)
            enable_fortran_compiler_flag_if_supported ("-march=native")
            enable_fortran_compiler_flag_if_supported ("--tree-vectorize")
        else ()
            enable_fortran_compiler_flag_if_supported (-mtune=generic)
        endif ()
    endif ()
endif ()

file (GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.f90")

add_library (starneig-pdgghrd SHARED ${SOURCES})

set_target_properties (starneig-pdgghrd PROPERTIES
    LINKER_LANGUAGE Fortran VERSION ${STARNEIG_VERSION})

install (TARGETS starneig-pdgghrd LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
