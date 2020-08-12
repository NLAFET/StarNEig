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

include (CheckSymbolExists)


#
# StarNEig library
#

find_package(PkgConfig REQUIRED)
pkg_search_module(STARNEIG REQUIRED starneig)

include_directories (${STARNEIG_INCLUDE_DIRS})
link_directories (${STARNEIG_LIBRARY_DIRS})
set (CMAKE_C_FLAGS "${STARNEIG_C_FLAGS} ${CMAKE_C_FLAGS}")

#
# shared memory examples
#

add_executable (sep-sm-full-chain sep_sm_full_chain.c validate.c)
target_link_libraries (sep-sm-full-chain ${STARNEIG_LIBRARIES})

add_executable (gep-sm-full-chain gep_sm_full_chain.c validate.c)
target_link_libraries (gep-sm-full-chain ${STARNEIG_LIBRARIES})

add_executable (sep-sm-eigenvectors sep_sm_eigenvectors.c validate.c)
target_link_libraries (sep-sm-eigenvectors ${STARNEIG_LIBRARIES})

add_executable (gep-sm-eigenvectors gep_sm_eigenvectors.c validate.c)
target_link_libraries (gep-sm-eigenvectors ${STARNEIG_LIBRARIES})

#
# distributed memory examples
#

set (CMAKE_REQUIRED_INCLUDES
    ${STARNEIG_INCLUDE_DIRS} ${CMAKE_REQUIRED_INCLUDES})

CHECK_SYMBOL_EXISTS (
    STARNEIG_ENABLE_MPI starneig/configuration.h STARNEIG_ENABLE_MPI)

CHECK_SYMBOL_EXISTS (STARNEIG_SEP_DM_HESSENBERG
    starneig/configuration.h STARNEIG_SEP_DM_HESSENBERG)

CHECK_SYMBOL_EXISTS (STARNEIG_GEP_DM_HESSENBERGTRIANGULAR
    starneig/configuration.h STARNEIG_GEP_DM_HESSENBERGTRIANGULAR)

if (STARNEIG_ENABLE_MPI)

    if (STARNEIG_SEP_DM_HESSENBERG)
        add_executable (sep-dm-full-chain sep_dm_full_chain.c validate.c)
        target_link_libraries (sep-dm-full-chain ${STARNEIG_LIBRARIES})
    endif ()

    if (STARNEIG_GEP_DM_HESSENBERGTRIANGULAR)
        add_executable (gep-dm-full-chain gep_dm_full_chain.c validate.c)
        target_link_libraries (gep-dm-full-chain ${STARNEIG_LIBRARIES})
    endif ()

endif ()
