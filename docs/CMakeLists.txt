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

project (StarNEig LANGUAGES NONE)
cmake_minimum_required (VERSION 3.3)

#
# file appending function
#

function (join_files input output)
    file (READ ${input} LICENSE)
    file (APPEND ${output} ${LICENSE})
    file (APPEND ${output} "\n")
endfunction ()

#
# setup Doxygen
#

find_package (Doxygen REQUIRED)
find_package (LATEX COMPONENTS PDFLATEX MAKEINDEX REQUIRED)

#
# library version
#

file (STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../version.txt" STARNEIG_VERSION)

#
# prepare header files
#

file (GLOB_RECURSE HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/include/starneig/*.h")

foreach (FILE_PATH ${HEADERS})
    get_filename_component (FILE_NAME ${FILE_PATH} NAME)
    configure_file (
        ${FILE_PATH} "${CMAKE_CURRENT_BINARY_DIR}/include/starneig/${FILE_NAME}"
        COPYONLY)
endforeach ()

set (STARNEIG_ENABLE_MPI YES)
set (STARNEIG_ENABLE_CUDA YES)
set (STARNEIG_ENABLE_BLACS YES)
set (STARNEIG_SEP_DM_HESSENBERG YES)
set (STARNEIG_GEP_DM_HESSENBERGTRIANGULAR YES)
set (STARNEIG_SEP_DM_REDUCE YES)
set (STARNEIG_GEP_DM_REDUCE YES)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/include/starneig/configuration.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/starneig/configuration.h" @ONLY)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/include/starneig/starneig.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/starneig/starneig.h" @ONLY)

#
# prepare dynamic documentation
#

file (WRITE
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_0_changelog.md"
    "# Known problems and changelog\n\n")
join_files (
    "${CMAKE_CURRENT_SOURCE_DIR}/../KNOWN_PROBLEMS.md"
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_0_changelog.md")
join_files (
    "${CMAKE_CURRENT_SOURCE_DIR}/../CHANGELOG.md"
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_0_changelog.md")

file (WRITE
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_1_authors.md"
    "# License, authors and literature\n\n")
join_files (
    "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_1_authors.md")
join_files (
    "${CMAKE_CURRENT_SOURCE_DIR}/../AUTHORS.md"
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_1_authors.md")
join_files (
    "${CMAKE_CURRENT_SOURCE_DIR}/../PUBLICATIONS.md"
    "${CMAKE_CURRENT_BINARY_DIR}/docs_generated/_1_authors.md")

#
# prepare Doxygen file
#

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)

#
# prepare targets
#

add_custom_target (doc_doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating Doxygen documentation"
    VERBATIM)

add_custom_target (doc_latex ALL
    COMMAND make
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/latex"
    DEPENDS doc_doxygen
    COMMENT "Generating Latex documentation"
    VERBATIM)

#
# copy output files
#

add_custom_command (TARGET doc_latex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf"
        "${CMAKE_CURRENT_BINARY_DIR}/starneig_manual.pdf")
