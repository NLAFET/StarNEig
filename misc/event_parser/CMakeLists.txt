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

cmake_minimum_required (VERSION 3.3)
enable_language (CXX)

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
# pthreads library
#

set (CMAKE_THREAD_PREFER_PTHREAD ON)
set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
set (CMAKE_REQUIRED_LIBRARIES
    ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_REQUIRED_LIBRARIES})

#
# X11 library
#

find_package(X11 REQUIRED)
set (CMAKE_REQUIRED_LIBRARIES ${X11_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})

#
# CImg header
#

find_header_file (CIMG_INCLUDE_PATH CImg.h "CImg include path" ERROR)

add_executable (parse parse.cpp)
target_link_libraries (parse ${CMAKE_REQUIRED_LIBRARIES})
