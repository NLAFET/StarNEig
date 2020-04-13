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

project (StarNEig)
cmake_minimum_required (VERSION 3.3)
enable_testing ()

option (STARNEIG_ENABLE_DOCS "Enable documentation" OFF)
option (STARNEIG_ENABLE_TESTS "Enable test binary" ON)
option (STARNEIG_ENABLE_EXAMPLES "Enable examples" OFF)
option (STARNEIG_ENABLE_EVENT_PARSER "Enable event parser" OFF)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory (src)
include_directories (${CMAKE_SOURCE_DIR}/src/include/)
include_directories (${CMAKE_BINARY_DIR}/src/include/)
set (CMAKE_REQUIRED_INCLUDES
    ${CMAKE_SOURCE_DIR}/src/include/ ${CMAKE_REQUIRED_INCLUDES})
set (CMAKE_REQUIRED_INCLUDES
    ${CMAKE_BINARY_DIR}/src/include/ ${CMAKE_REQUIRED_INCLUDES})
set (ENV{PKG_CONFIG_PATH}
    "$ENV{PKG_CONFIG_PATH}:${CMAKE_CURRENT_BINARY_DIR}/src/")

if (STARNEIG_ENABLE_DOCS)
    add_subdirectory (docs)
endif ()

if (STARNEIG_ENABLE_TESTS)
    add_subdirectory (test)
endif ()

if (STARNEIG_ENABLE_EXAMPLES)
    add_subdirectory (examples)
endif ()

if (STARNEIG_ENABLE_EVENT_PARSER)
    add_subdirectory (misc/event_parser)
endif ()
