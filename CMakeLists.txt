cmake_minimum_required(VERSION 3.10)

# specify used SYCL implementation
set(SUPPORTED_SYCL_LSH_IMPLEMENTATIONS hipSYCL ComputeCpp oneAPI)
set(SYCL_LSH_IMPLEMENTATION hipSYCL CACHE STRING "The used SYCL implementation backend.")
set_property(CACHE SYCL_LSH_IMPLEMENTATION PROPERTY STRINGS ${SUPPORTED_SYCL_LSH_IMPLEMENTATIONS})
if (NOT SYCL_LSH_IMPLEMENTATION IN_LIST SUPPORTED_SYCL_LSH_IMPLEMENTATIONS)
    string(REPLACE ";" ", " SUPPORTED_SYCL_LSH_IMPLEMENTATIONS_OUT "${SUPPORTED_SYCL_LSH_IMPLEMENTATIONS}")
    message(FATAL_ERROR "SYCL implementation \"${SYCL_LSH_IMPLEMENTATION}\" not supported!\nMust be one of: ${SUPPORTED_SYCL_LSH_IMPLEMENTATIONS_OUT}")
else ()
    message(STATUS "Using \"${SYCL_LSH_IMPLEMENTATION}\" as SYCL implementation.")
endif ()


# set compiler wrapper to dpcpp if using Intel's oneAPI, since find_package(oneAPI) isn't supported
if (SYCL_LSH_IMPLEMENTATION MATCHES "oneAPI")
    set(CMAKE_CXX_COMPILER dpcpp)
endif()


# set supported SYCL targets
set(SUPPORTED_SYCL_LSH_TARGETS CPU NVIDIA AMD INTEL) # CPU = 0, NVIDIA GPU = 1, AMD GPU = 2, INTEL GPU = 3
set(SYCL_LSH_TARGET NVIDIA CACHE STRING "The used SYCL target.")
set_property(CACHE SYCL_LSH_TARGET PROPERTY STRINGS ${SUPPORTED_SYCL_LSH_TARGETS})
if (NOT SYCL_LSH_TARGET IN_LIST SUPPORTED_SYCL_LSH_TARGETS)
    string(REPLACE ";" ", " SUPPORTED_SYCL_LSH_TARGETS_OUT "${SUPPORTED_SYCL_LSH_TARGETS}")
    message(FATAL_ERROR "SYCL target \"${SYCL_LSH_TARGET}\" not supported!\nMust be one of: ${SUPPORTED_SYCL_LSH_TARGETS_OUT}")
else ()
    message(STATUS "Using \"${SYCL_LSH_TARGET}\" as SYCL target.")
endif ()


project("Distributed k-nearest Neighbors using Locality Sensitive Hashing and SYCL"
        VERSION 2.0.0
        LANGUAGES CXX
        DESCRIPTION "Master Thesis Marcel Breyer")

# TODO: check whether OpenMP must be added manually 

if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL")

    # MUST set the environment variable hipSYCL_DIR to the root directory of hipSYCL
    set(hipSYCL_DIR $ENV{hipSYCL_DIR}/lib/cmake CACHE INTERNAL "Path to the root directory of the hipSYCL installation.")

    # set HIPSYCL_PLATFORM based on the provided SYCL_LSH_TARGET
    # MUST set the environment variable hipSYCL_GPU_ARCH to the correct architecture value
    if (SYCL_LSH_TARGET MATCHES "CPU")
        # use the CPU as target
        set(HIPSYCL_PLATFORM cpu CACHE INTERNAL "")
    elseif (SYCL_LSH_TARGET MATCHES "NVIDIA")
        # use NVIDIA GPUs as target
        set(HIPSYCL_PLATFORM cuda CACHE INTERNAL "")
        set(HIPSYCL_GPU_ARCH $ENV{hipSYCL_GPU_ARCH} CACHE INTERNAL "The GPU architecture used.")
    elseif (SYCL_LSH_TARGET MATCHES "AMD")
        # use AMD GPUs as target
        set(HIPSYCL_PLATFORM rocm CACHE INTERNAL "")
        set(HIPSYCL_GPU_ARCH $ENV{HIPSYCL_GPU_ARCH} CACHE INTERNAL "The GPU architecture used.")
    elseif (SYCL_LSH_TARGET MATCHES "INTEL")
        # Intel GPUs aren't supported by hipSYCL
        unset(HIPSYCL_PLATFORM CACHE)
        message(FATAL_ERROR "Intel GPUs aren't supported with hipSYCL.")
    endif ()

    # find hipSYCL
    find_package(hipSYCL REQUIRED)

elseif (SYCL_LSH_IMPLEMENTATION MATCHES "ComputeCpp")

    # needed to be able to use ComputeCpp
    set(CMAKE_CXX_STANDARD 17)
    
    # set bitcode to ptx64 if targeting NVIDIA GPUs
    if (SYCL_LSH_TARGET MATCHES "NVIDIA")
        set(COMPUTECPP_BITCODE ptx64 CACHE INTERNAL "")
    else ()
        unset(COMPUTECPP_BITCODE CACHE)
    endif ()

    # MUST set the environment variable ComputeCpp_DIR to the root directory of ComputeCpp
    set(ComputeCpp_DIR $ENV{ComputeCpp_DIR} CACHE INTERNAL "Path to the root directory of the ComputeCpp installation.")
    
    # MUST set the environment variable ComputeCpp_SDK_DIR to the root directory of the ComputeCpp SDK
    list(APPEND CMAKE_MODULE_PATH "$ENV{ComputeCpp_SDK_DIR}/cmake/Modules")

    # find ComputeCpp
    include(FindComputeCpp)
    include_directories(${COMPUTE_CPP_INCLUDE_DIRECTORIES})

endif ()




# set cmake build type if none was specified
set(SUPPORTED_SYCL_LSH_BUILD_TYPES Debug Release MinSizeRel RelWithDebInfo)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(SYCL_LSH_DEFAULT_BUILD_TYPE "Release")
    message(STATUS "Setting build type to \"${SYCL_LSH_DEFAULT_BUILD_TYPE}\" as none was specified!")
    set(CMAKE_BUILD_TYPE "${SYCL_LSH_DEFAULT_BUILD_TYPE}" CACHE STRING "The used CMake build type." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${SUPPORTED_SYCL_LSH_BUILD_TYPES})
endif ()
if (NOT CMAKE_BUILD_TYPE IN_LIST SUPPORTED_SYCL_LSH_BUILD_TYPES)
    string(REPLACE ";" ", " SUPPORTED_SYCL_LSH_BUILD_TYPES_OUT "${SUPPORTED_SYCL_LSH_BUILD_TYPES}")
    message(FATAL_ERROR "Build type \"${CMAKE_BUILD_TYPE}\" not supported!\nMust be one of: ${SUPPORTED_SYCL_LSH_BUILD_TYPES_OUT}")
else ()
    message(STATUS "Using \"${CMAKE_BUILD_TYPE}\" as build type.")
endif ()




# create library
set(SYCL_LSH_LIBRARY_NAME "sycl_lsh")
add_library(${SYCL_LSH_LIBRARY_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/argv_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/communicator_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/file_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/window_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/communicator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/errhandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/timer.cpp
)

# set include directory
target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC include)

# set c++ standard
target_compile_features(${SYCL_LSH_LIBRARY_NAME} PUBLIC cxx_std_17)


# find MPI library and add it to the target
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC MPI::MPI_CXX)


# find {fmt} formatting library and add it to the target
find_package(fmt REQUIRED)
option(SYCL_LSH_FMT_HEADER_ONLY "Use the header only mode of the {fmt} formatting library." OFF)
if (SYCL_LSH_FMT_HEADER_ONLY)
    message(STATUS "Using the {fmt} library in its header only mode.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC FMT_HEADER_ONLY=1)
else ()
    message(STATUS "Using the {fmt} library as shared library and link against it.")
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC fmt::fmt)
endif ()


# add debug mode
option(SYCL_LSH_ENABLE_DEBUG "Enable Debug mode using assertion macros." OFF)
if (SYCL_LSH_ENABLE_DEBUG)
    message(STATUS "Enabled Debug mode.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC -DSYCL_LSH_DEBUG)

    # sanitizers
#    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
#    target_link_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
endif ()


# set timer behavior
set(SUPPORTED_SYCL_LSH_TIMERS NONE NON_BLOCKING BLOCKING) # NONE = 0, NON_BLOCKING = 1, BLOCKING = 2
set(SYCL_LSH_TIMER BLOCKING CACHE STRING "The used timer implementation.")
set_property(CACHE SYCL_LSH_TIMER PROPERTY STRINGS ${SUPPORTED_SYCL_LSH_TIMERS})
if (NOT SYCL_LSH_TIMER IN_LIST SUPPORTED_SYCL_LSH_TIMERS)
    string(REPLACE ";" ", " SUPPORTED_SYCL_LSH_TIMERS_OUT "${SUPPORTED_SYCL_LSH_TIMERS}")
    message(FATAL_ERROR "Timer \"${SYCL_LSH_TIMER}\" not supported!\nMust be one of: ${SUPPORTED_SYCL_LSH_TIMERS_OUT}")
else ()
    message(STATUS "Using \"${SYCL_LSH_TIMER}\" as timer implementation.")

    list(FIND SUPPORTED_SYCL_LSH_TIMERS "${SYCL_LSH_TIMER}" SYCL_LSH_TIMER_IDX)
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TIMER=${SYCL_LSH_TIMER_IDX})
endif ()


# in order to benchmark the code, a timer must be enabled
set(SYCL_LSH_BENCHMARK "" CACHE STRING "The path to the benchmarking file if benchmarking should be enabled.")
if (NOT "${SYCL_LSH_BENCHMARK}" STREQUAL "")
    if (SYCL_LSH_TIMER MATCHES "NONE")
        message(FATAL_ERROR "In order to enable benchmarking, a valid timer (NON_BLOCKING or BLOCKING) must be enabled!")
    endif ()
    message(STATUS "Enabled benchmarking using the file: \"${SYCL_LSH_BENCHMARK}\"")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_BENCHMARK="${SYCL_LSH_BENCHMARK}")
endif ()


# enable <experimental/filesystem> header if requested
option(SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM "Use the <experimental/filesystem> header instead of <filesystem>." OFF)
if (SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM)
    message(STATUS "Using <experimental/filesystem> instead of <filesystem>.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM)
    # silence LLVM deprecated warning
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM)
endif ()


# propagate SYCL_LSH_IMPLEMENTATION and SYCL_LSH_TARGET to library
list(FIND SUPPORTED_SYCL_LSH_IMPLEMENTATIONS "${SYCL_LSH_IMPLEMENTATION}" SYCL_LSH_IMPLEMENTATION_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION=${SYCL_LSH_IMPLEMENTATION_IDX})
list(FIND SUPPORTED_SYCL_LSH_TARGETS "${SYCL_LSH_TARGET}" SYCL_LSH_TARGET_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TARGET=${SYCL_LSH_TARGET_IDX})


# set SYCL implementation specific options
if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL")
    # disable -Wpedantic if compiling for hipSYCL targeting the CPU
    if (SYCL_LSH_TARGET MATCHES "CPU")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -Wno-pedantic)
    endif ()
    # disable -Wunused-parameter if compiling for hipSYCL
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -Wno-unused-parameter -Wno-gcc-compat)
elseif (SYCL_LSH_IMPLEMENTATION MATCHES "ComputeCpp")
    # add additional ComputeCpp specific flags
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC -lstdc++fs)
    list(APPEND COMPUTECPP_USER_FLAGS -no-serial-memop)
elseif (SYCL_LSH_IMPLEMENTATION MATCHES "oneAPI")
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --gcc-toolchain=$ENV{DPCPP_GCC_TOOLCHAIN})
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC -fsycl -lstdc++fs)
endif()


# create executable
add_executable(prog src/main.cpp)
target_compile_options(prog PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(prog PRIVATE ${SYCL_LSH_LIBRARY_NAME})


# add necessary SYCL flags to the target
if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL|ComputeCpp")
    add_sycl_to_target(TARGET prog SOURCES src/main.cpp)
endif ()


# generate documentation if requested
option(SYCL_LSH_ENABLE_DOCUMENTATION "Enable the generation of the documentation using Doxygen." OFF)
if (SYCL_LSH_ENABLE_DOCUMENTATION)
    message(STATUS "Using Doxygen to generate documentation.")
    add_subdirectory(doc)
endif ()
