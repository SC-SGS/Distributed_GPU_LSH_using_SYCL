# Authors: Marcel Breyer
# License: This file is released under the MIT license. See the LICENSE file in the project root for full information.
########################################################################################################################

cmake_minimum_required(VERSION 3.25)

project("Distributed k-nearest Neighbors using Locality Sensitive Hashing and SYCL"
        VERSION 3.0.0
        LANGUAGES CXX
        DESCRIPTION "Master Thesis Marcel Breyer")

########################################################################################################################
#                                             set default cmake build type                                             #
########################################################################################################################
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

########################################################################################################################
#                                                    create library                                                    #
########################################################################################################################
# set all library source files
set(SYCL_LSH_LIBRARY_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/detail/shape.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/detail/utility.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/exceptions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/source_location.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/hash_functions/entropy_based.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/hash_functions/hash_function_types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/hash_functions/mixed_hash_functions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/hash_functions/random_projections.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/detail/utility.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/file_parser/file.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/file_parser/file_parser_types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/communicator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/environment.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/timer.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/data_attributes.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/data_set.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/nearest_neighbors.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/nearest_neighbors_report.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/memory_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/options.cpp
)

# create library
set(SYCL_LSH_LIBRARY_NAME sycl-lsh)
add_library(${SYCL_LSH_LIBRARY_NAME} STATIC ${SYCL_LSH_LIBRARY_SOURCES})

########################################################################################################################
#                                                  set compiler flags                                                  #
########################################################################################################################
# set C++ standard
target_compile_features(${SYCL_LSH_LIBRARY_NAME} PUBLIC cxx_std_17)
# set include directory
target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
# additional library compile options
target_compile_options(
        ${SYCL_LSH_LIBRARY_NAME}
        PUBLIC $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,IntelLLVM>:-Wall
        -Wextra
        -fno-common
        -Wshadow
        -Wcast-qual
        -Wnull-dereference
        -Wnon-virtual-dtor
        -Wextra-semi
        -Wunreachable-code
        -Wuninitialized
        -Wno-ctor-dtor-privacy
        -fPIC>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wsuggest-override
        -Wstrict-null-sentinel
        -Wlogical-op
        -Wduplicated-branches
        -Wimplicit-fallthrough=5>
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,IntelLLVM>:-Wmost>>
        -Wno-double-promotion
)

target_compile_options(
        ${SYCL_LSH_LIBRARY_NAME}
        PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,IntelLLVM>:-Werror=switch
        -fstrict-enums>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/we4062
        /wd4005
        /wd4702
        /wd4849
        /wd4127
        /wd4250 # disable multiple-inheritance dominance warning
        /wd4589> # disable constructor inheritance warning
        # /wd4849: ignore "OpenMP 'reduction' clause ignored in 'simd' directive" -> no SIMD clause currently effective in MSVC /wd4127: ignore "conditional
        # expression is constant" from {fmt} ranges.h header
)

# compile with more optimizations
target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC
        $<$<AND:$<NOT:$<CONFIG:DEBUG>>,$<CXX_COMPILER_ID:GNU,Clang>>:-ffast-math -march=native>
)

# add math defines to library
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PRIVATE _USE_MATH_DEFINES)

########################################################################################################################
#                                                   add SYCL support                                                   #
########################################################################################################################
list(APPEND CMAKE_MESSAGE_INDENT "SYCL:  ")

# we have to specify the correct target (e.g., nvidia_gpu_sm_80, amd_gpu_gfx90a, intel_gpu_bmg_g21, spir64_x86_64, etc.)
# emit an error if it was not specified
if (NOT SYCL_LSH_TARGET_ARCH)
    message(FATAL_ERROR "The necessary \"SYCL_LSH_TARGET_ARCH\" variable was not set!\n"
            "Valid examples are: nvidia_gpu_sm_80, amd_gpu_gfx90a, intel_gpu_bmg_g21, or spir64_x86_64")
endif ()

set(SYCL_LSH_IMPLEMENTATION "icpx" CACHE STRING "Choose the SYCL implementation. Either icpx or acpp.")

if (SYCL_LSH_IMPLEMENTATION STREQUAL "icpx")
    # check for the icpx compiler
    message(STATUS "Use Intel's icpx as SYCL implementation.")
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        message(FATAL_ERROR "Couldn't find the requested Intel compiler for SYCL!")
    endif ()
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -sycl-std=2020 -fno-sycl-id-queries-fit-in-int -fsycl -fsycl-targets=${SYCL_LSH_TARGET_ARCH})
    target_link_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -fsycl -fsycl-targets=${SYCL_LSH_TARGET_ARCH})
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION="icpx")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION_ICPX)
elseif (SYCL_LSH_IMPLEMENTATION STREQUAL "acpp")
    # check for AdaptiveCpp
    message(STATUS "Use AdaptiveCpp as SYCL implementation.")
    set(ACPP_TARGETS "generic" CACHE STRING "" FORCE)
    find_package(AdaptiveCpp CONFIG REQUIRED)
    add_sycl_to_target(TARGET ${SYCL_LSH_LIBRARY_NAME} SOURCES ${SYCL_LSH_LIBRARY_SOURCES})
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION="acpp")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION_ACPP)
else ()
    message(FATAL_ERROR "Unrecognized SYCL implementation \"${SYCL_LSH_IMPLEMENTATION}\"! Supported values are: icpx and acpp.")
endif ()

# if the target arch is spir64_x86_64, we use a CPU, otherwise we ALWAYS use a GPU!
if (SYCL_LSH_TARGET_ARCH STREQUAL "spir64_x86_64")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_USE_CPU)
endif ()
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TARGET_ARCH="${SYCL_LSH_TARGET_ARCH}")

list(POP_BACK CMAKE_MESSAGE_INDENT)

########################################################################################################################
#                                                   create executable                                                  #
########################################################################################################################
set(SYCL_LSH_EXECUTABLE_NAME prog)
add_executable(${SYCL_LSH_EXECUTABLE_NAME} src/main.cpp)

# link executables against library
target_link_libraries(${SYCL_LSH_EXECUTABLE_NAME} PRIVATE ${SYCL_LSH_LIBRARY_NAME})

########################################################################################################################
#                                           check for Link Time Optimization                                           #
########################################################################################################################
# enable Link Time Optimization (LTO)
option(SYCL_LSH_ENABLE_LTO "Enable Link Time Optimizations." OFF)
if (SYCL_LSH_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT SYCL_LSH_LTO_SUPPORTED LANGUAGES CXX)
    if (SYCL_LSH_LTO_SUPPORTED)
        message(STATUS "Interprocedural optimization (IPO/LTO) enabled.")
        set_property(TARGET ${SYCL_LSH_EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else ()
        message(STATUS "Interprocedural optimization (IPO/LTO) not supported.")
    endif ()
endif ()

########################################################################################################################
#                          enable the requested vectorization widths for the auto-vectorizers                          #
########################################################################################################################
# GCC and clang both do not automatically auto-vectorize for AVX-512 (only AVX2)
# -> if the SYCL_LSH_TARGET_ARCH is "spir64_x86_64" we are compiling for a CPU
# -> the CPU vectorization standard to use MUST be explicitly provided
if (SYCL_LSH_TARGET_ARCH STREQUAL "spir64_x86_64")
    if (NOT SYCL_LSH_CPU_VECTORIZATION_TARGET)
        message(FATAL_ERROR "For the CPU target, the \"SYCL_LSH_CPU_VECTORIZATION_TARGET\" must be explicitly provided!")
    endif ()

    if (${SYCL_LSH_CPU_VECTORIZATION_TARGET} STREQUAL "avx512")
        message(STATUS "Enabling AVX512 support for the auto-vectorizers (-mprefer-vector-width=512).")
        target_compile_options(
                ${SYCL_LSH_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,IntelLLVM>:-mprefer-vector-width=512>>
        )
    elseif (${SYCL_LSH_CPU_VECTORIZATION_TARGET} STREQUAL "avx2" OR ${SYCL_LSH_CPU_VECTORIZATION_TARGET} STREQUAL "avx")
        message(STATUS "Enabling AVX/AVX2 support for the auto-vectorizers (-mprefer-vector-width=256).")
        target_compile_options(
                ${SYCL_LSH_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,IntelLLVM>:-mprefer-vector-width=256>>
        )
    elseif (${SYCL_LSH_CPU_VECTORIZATION_TARGET} MATCHES "^sse")
        message(STATUS "Enabling SSE for the auto-vectorizers (-mprefer-vector-width=128).")
        target_compile_options(
                ${SYCL_LSH_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,IntelLLVM>:-mprefer-vector-width=128>>
        )
    else ()
        message(FATAL_ERROR "Unrecognized CPU target architecture \"${SYCL_LSH_CPU_VECTORIZATION_TARGET}\". Allowed values are: avx512, avx2, avx, sse.")
    endif ()
endif ()

########################################################################################################################
#                                     check for optional and necessary dependencies                                    #
########################################################################################################################
# use MPI + OpenMP to speedup the calculations
find_package(MPI REQUIRED)
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC MPI::MPI_CXX)
find_package(OpenMP REQUIRED)
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)

# setup dependencies
include(FetchContent)
list(APPEND CMAKE_MESSAGE_INDENT "Dependencies:  ")

# try finding cxxopts
set(SYCL_LSH_cxxopts_VERSION v3.3.1)
find_package(cxxopts 3.3.1 QUIET)
if (cxxopts_FOUND)
    message(STATUS "Found package cxxopts.")
    target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC ${cxxopts_INCLUDE_DIR})
else ()
    message(STATUS "Couldn't find package cxxopts. Building version ${SYCL_LSH_cxxopts_VERSION} from source.")
    # set options for cxxopts
    set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
    set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
    set(CXXOPTS_ENABLE_WARNINGS OFF CACHE INTERNAL "" FORCE)
    # fetch command line parser library cxxopts
    FetchContent_Declare(
            cxxopts
            GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
            GIT_TAG ${SYCL_LSH_cxxopts_VERSION}
            QUIET
    )
    FetchContent_MakeAvailable(cxxopts)
    add_dependencies(${SYCL_LSH_LIBRARY_NAME} cxxopts)
    target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${cxxopts_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
endif ()

# try finding igor
set(SYCL_LSH_igor_VERSION fef608b458ab2fca736ac8e415e4d4160a48745d) # can't use a newer commit since igor switch to mandatory C++20
find_package(igor QUIET)
if (igor_FOUND)
    message(STATUS "Found package igor.")
    target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC ${igor_INCLUDE_DIR})
else ()
    message(STATUS "Couldn't find package igor. Building version ${SYCL_LSH_igor_VERSION} from source.")
    # set options for igor
    set(IGOR_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
    # disable additional compiler flags check (only warning flags) since for some reasons it takes ages with stdpar compilers
    set(_YACMACompilerLinkerSettingsIncluded ON CACHE INTERNAL "" FORCE)
    # fetch named argument library igor
    FetchContent_Declare(
            igor
            GIT_REPOSITORY https://github.com/bluescarni/igor
            GIT_TAG ${SYCL_LSH_igor_VERSION}
            QUIET
    )
    FetchContent_MakeAvailable(igor)
    add_dependencies(${SYCL_LSH_LIBRARY_NAME} igor)
    target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${igor_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
endif ()

# try finding fmt
set(SYCL_LSH_fmt_VERSION 11.0.2)
find_package(fmt 11.0.2 QUIET)
if (fmt_FOUND)
    message(STATUS "Found package fmt.")
else ()
    message(STATUS "Couldn't find package fmt. Building version ${SYCL_LSH_fmt_VERSION} from source.")
    # set options for fmt
    set(FMT_PEDANTIC OFF CACHE INTERNAL "" FORCE)
    set(FMT_WERROR OFF CACHE INTERNAL "" FORCE)
    set(FMT_DOC OFF CACHE INTERNAL "" FORCE)
    set(FMT_INSTALL ON CACHE INTERNAL "" FORCE) # let {fmt} handle the install target
    set(FMT_TEST OFF CACHE INTERNAL "" FORCE)
    set(FMT_FUZZ OFF CACHE INTERNAL "" FORCE)
    set(FMT_CUDA_TEST OFF CACHE INTERNAL "" FORCE)
    set(FMT_MODULE OFF CACHE INTERNAL "" FORCE)
    set(FMT_SYSTEM_HEADERS ON CACHE INTERNAL "" FORCE)
    # fetch string formatting library fmt
    FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG ${SYCL_LSH_fmt_VERSION}
            QUIET
    )
    FetchContent_MakeAvailable(fmt)
    set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(fmt PROPERTIES CXX_CLANG_TIDY "")
    target_compile_definitions(fmt PRIVATE FMT_USE_FULL_CACHE_DRAGONBOX)
    add_dependencies(${SYCL_LSH_LIBRARY_NAME} fmt)
endif ()
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC fmt::fmt)

list(POP_BACK CMAKE_MESSAGE_INDENT)

########################################################################################################################
#                                         set additional configuration options                                         #
########################################################################################################################
# set timer behavior
set(SYCL_LSH_TIMERS NONE NON_BLOCKING BLOCKING) # NONE = 0, NON_BLOCKING = 1, BLOCKING = 2
set(SYCL_LSH_TIMER BLOCKING CACHE STRING "The used timer implementation.")
set_property(CACHE SYCL_LSH_TIMER PROPERTY STRINGS ${SYCL_LSH_TIMERS})
if (NOT SYCL_LSH_TIMER IN_LIST SYCL_LSH_TIMERS)
    string(REPLACE ";" ", " SYCL_LSH_TIMERS_OUT "${SYCL_LSH_TIMERS}")
    message(FATAL_ERROR "Timer \"${SYCL_LSH_TIMER}\" not supported!\nMust be one of: ${SYCL_LSH_TIMERS_OUT}")
else ()
    message(STATUS "Using \"${SYCL_LSH_TIMER}\" as timer implementation.")

    list(FIND SYCL_LSH_TIMERS "${SYCL_LSH_TIMER}" SYCL_LSH_TIMER_IDX)
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TIMER=${SYCL_LSH_TIMER_IDX})
endif ()

# add custom assertions
option(SYCL_LSH_ENABLE_ASSERTS "Enable assertion macros." OFF)
if (SYCL_LSH_ENABLE_ASSERTS)
    message(STATUS "Enabled additional sycl-lsh sanity checks using assertions.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_ASSERTS_ENABLED)
endif ()

# do not seed random number generators
option(SYCL_LSH_RANDOM_NUMBERS_DEBUG "If enabled, random number generators are not seeded resulting in predictability." OFF)
if (SYCL_LSH_RANDOM_NUMBERS_DEBUG)
    message(STATUS "Random numbers are not seeded enabling easier debugability.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_RANDOM_NUMBERS_DEBUG)
endif ()

# enable benchmarking using an output file # TODO: combine with hardware sampling?
set(SYCL_LSH_BENCHMARK "" CACHE STRING "The path to the benchmarking file if benchmarking should be enabled.")
if (NOT "${SYCL_LSH_BENCHMARK}" STREQUAL "")
    # in order to benchmark the code, a timer must be enabled
    if (SYCL_LSH_TIMER MATCHES "NONE")
        message(FATAL_ERROR "In order to enable benchmarking, a valid timer (NON_BLOCKING or BLOCKING) must be enabled!")
    endif ()
    message(STATUS "Enabled benchmarking using the file: \"${SYCL_LSH_BENCHMARK}\"")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_BENCHMARK="${SYCL_LSH_BENCHMARK}")
endif ()

########################################################################################################################
#                                               enable hardware sampling                                               #
########################################################################################################################
list(APPEND CMAKE_MESSAGE_INDENT "hws:  ") #  TODO: implement on the C++ side!

# option for the used sampling interval
set(SYCL_LSH_HARDWARE_SAMPLING_INTERVAL "100" CACHE STRING "The interval in milliseconds in which the hardware information (like clock frequency or power draw) are queried.")
message(STATUS "Setting the hardware sampling interval to ${SYCL_LSH_HARDWARE_SAMPLING_INTERVAL}ms.")
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_HARDWARE_SAMPLING_INTERVAL=${SYCL_LSH_HARDWARE_SAMPLING_INTERVAL}ms)

set(SYCL_LSH_HWS_VERSION v1.1.2)
find_package(hws QUIET)
if (hws_FOUND)
    message(STATUS "Found package hws.")
else ()
    message(STATUS "Couldn't find package hws. Building version ${SYCL_LSH_HWS_VERSION} from source.")
    set(HWS_ENABLE_ERROR_CHECKS OFF CACHE INTERNAL "" FORCE)
    set(HWS_SAMPLING_INTERVAL ${SYCL_LSH_HARDWARE_SAMPLING_INTERVAL} CACHE INTERNAL "" FORCE)
    set(HWS_ENABLE_PYTHON_BINDINGS OFF CACHE INTERNAL "" FORCE)
    FetchContent_Declare(hws
            GIT_REPOSITORY https://github.com/SC-SGS/hardware_sampling
            GIT_TAG ${SYCL_LSH_HWS_VERSION}
            QUIET
    )
    FetchContent_MakeAvailable(hws)
    add_dependencies(${SYCL_LSH_LIBRARY_NAME} hws)
endif ()
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC hws::hws)

list(POP_BACK CMAKE_MESSAGE_INDENT)

########################################################################################################################
#                                     enable documentation generation via doxygen                                      #
########################################################################################################################
option(SYCL_LSH_ENABLE_DOCUMENTATION "Enable the generation of the documentation using Doxygen" OFF)
if (SYCL_LSH_ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
    # if documentation is enabled, add documentation warning flag when compiling with clang
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,Clang,IntelLLVM>:-Wdocumentation>)
endif ()
