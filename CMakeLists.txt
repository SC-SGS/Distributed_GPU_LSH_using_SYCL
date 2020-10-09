cmake_minimum_required(VERSION 3.10)

# specify used SYCL implementation
set(SUPPORTED_SYCL_LSH_IMPLEMENTATIONS hipSYCL ComputeCpp oneAPI)
if (NOT SYCL_LSH_IMPLEMENTATION)
    set(SYCL_LSH_IMPLEMENTATION hipSYCL)
endif ()
if (NOT ${SYCL_LSH_IMPLEMENTATION} IN_LIST SUPPORTED_SYCL_LSH_IMPLEMENTATIONS)
    message(FATAL_ERROR "SYCL implementation \"${SYCL_LSH_IMPLEMENTATION}\" currently not supported!")
else ()
    message(STATUS "Using \"${SYCL_LSH_IMPLEMENTATION}\" as SYCL implementation.")
endif ()


# set compiler wrapper based on used SYCL implementation
if (SYCL_LSH_IMPLEMENTATION MATCHES "oneAPI")
    set(CMAKE_CXX_COMPILER dpcpp)
endif()


# set supported SYCL targets
set(SUPPORTED_SYCL_LSH_TARGETS CPU NVIDIA AMD) # CPU = 0, NVIDIA = 1, AMD = 2
if (NOT SYCL_LSH_TARGET)
    set(SYCL_LSH_TARGET NVIDIA)
endif ()
if (NOT ${SYCL_LSH_TARGET} IN_LIST SUPPORTED_SYCL_LSH_TARGETS)
    message(FATAL_ERROR "SYCL target \"${SYCL_LSH_TARGET}\" not supported!")
else ()
    message(STATUS "Using \"${SYCL_LSH_TARGET}\" as SYCL target.")
endif ()


project("Distributed GPU LSH implementation using SYCL"
        VERSION 1.0.0
        LANGUAGES CXX
        DESCRIPTION "master theses Marcel Breyer")


if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL")
    set(hipSYCL_DIR $ENV{hipSYCL_DIR}/lib/cmake CACHE STRING "Path to the root directory of the hipSYCL installation." FORCE)
    if (SYCL_LSH_TARGET MATCHES "CPU")
        # TODO: using this line results in many linker errors
        #        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=cpu)
        set(HIPSYCL_PLATFORM cpu)
    elseif (SYCL_LSH_TARGET MATCHES "NVIDIA")
        #        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=cuda)
        set(HIPSYCL_PLATFORM cuda)
        set(HIPSYCL_GPU_ARCH sm_75)
    elseif (SYCL_LSH_TARGET MATCHES "AMD")
        #        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=rocm)
        set(HIPSYCL_PLATFORM rocm)
    endif ()
    find_package(hipSYCL REQUIRED)
elseif (SYCL_LSH_IMPLEMENTATION MATCHES "ComputeCpp")
    # needed to be able to use ComputeCpp
    set(CMAKE_CXX_STANDARD 17)
    
    # set bitcode to ptx64 if targetting NVIDIA GPUs
    if (SYCL_LSH_TARGET MATCHES "NVIDIA")
        set(COMPUTECPP_BITCODE ptx64 CACHE STRING "Must use ptx64 bitcode if targeting NVIDIA GPUs." FORCE)
    endif ()

    # MUST set ComputeCpp_DIR to the root directory of ComputeCpp
    set(ComputeCpp_DIR $ENV{ComputeCpp_DIR} CACHE STRING "Path to the root directory of the ComputeCpp installation." FORCE)
    
    # MUST set ComputeCpp_SDK_DIR to the root directory of the ComputeCpp SDK
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{ComputeCpp_SDK_DIR}/cmake/Modules")

    # find ComputeCpp
    include(FindComputeCpp)
    include_directories(${COMPUTE_CPP_INCLUDE_DIRECTORIES})
endif ()


# cmake configuration options
option(SYCL_LSH_ENABLE_DEBUG "Enable debugging macros" OFF)
option(SYCL_LSH_ENABLE_DOCUMENTATION "Generate documentation" OFF)
option(SYCL_LSH_FMT_HEADER_ONLY "Use the header only mode of the fmt library" OFF)
option(SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM "Use the '<experimental/filesystem>' header instead of '<filesystem>'" OFF)


# set timer behavior
set(SUPPORTED_SYCL_LSH_TIMERS NONE NON_BLOCKING BLOCKING) # NONE = 0, NON_BLOCKING = 1, BLOCKING = 2
if (NOT DEFINED SYCL_LSH_TIMER)
    set(SYCL_LSH_TIMER BLOCKING)
endif ()
if (NOT ${SYCL_LSH_TIMER} IN_LIST SUPPORTED_SYCL_LSH_TIMERS)
    message(FATAL_ERROR "Timer \"${SYCL_LSH_TIMER}\" not supported!")
else ()
    message(STATUS "Using \"${SYCL_LSH_TIMER}\" as timer.")
endif ()
# in order to benchmark the code, a timer must be enabled
if (DEFINED SYCL_LSH_BENCHMARK AND ${SYCL_LSH_TIMER} MATCHES "NONE")
    message(FATAL_ERROR "In order to benchmark, a timer (NON_BLOCKING or BLOCKING) must be enabled!")
endif ()


# set cmake build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified!")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type" FORCE)
endif ()
message(STATUS "Using \"${CMAKE_BUILD_TYPE}\" as build type.")


# add custom cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")


# create library
set(SYCL_LSH_LIBRARY_NAME "sycl_lsh")
add_library(${SYCL_LSH_LIBRARY_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/argv_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/communicator_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/file_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/window_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/communicator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/errhandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/timer.cpp
)

# set include directory
target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC include)


# find MPI and add it to the target
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC MPI::MPI_CXX)


# find fmt lib and add it to the target
find_package(fmt REQUIRED)
if (SYCL_LSH_FMT_HEADER_ONLY)
    message(STATUS "Using fmt lib in its header only mode.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC FMT_HEADER_ONLY=1)
else ()
    message(STATUS "Linking against fmt lib.")
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC fmt::fmt)
endif ()


# set c++ standard
target_compile_features(${SYCL_LSH_LIBRARY_NAME} PUBLIC cxx_std_17)
target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -Wno-format-security)


# set library compile definitions
if (SYCL_LSH_ENABLE_DEBUG)
    message(STATUS "Enabled debugging.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC -DSYCL_LSH_DEBUG)

#    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
#    target_link_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
endif ()

# enable benchmark if requested
if (NOT DEFINED SYCL_LSH_BENCHMARK)
    message(STATUS "Disabled benchmarking.")
else ()
    message(STATUS "Enabled benchmarking. Results will be written to: \"${SYCL_LSH_BENCHMARK}\"")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_BENCHMARK="${SYCL_LSH_BENCHMARK}")
endif ()

# enable <experimental/filesystem> header if requested
if (SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM)
    message(STATUS "Using <experimental/filesystem> instead of <filesystem>.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM)
    # silence LLVM deprecated warning
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM)
endif ()

list(FIND SUPPORTED_SYCL_LSH_IMPLEMENTATIONS "${SYCL_LSH_IMPLEMENTATION}" SYCL_LSH_IMPLEMENTATION_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION=${SYCL_LSH_IMPLEMENTATION_IDX})
list(FIND SUPPORTED_SYCL_LSH_TARGETS "${SYCL_LSH_TARGET}" SYCL_LSH_TARGET_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TARGET=${SYCL_LSH_TARGET_IDX})
list(FIND SUPPORTED_SYCL_LSH_TIMERS "${SYCL_LSH_TIMER}" SYCL_LSH_TIMER_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TIMER=${SYCL_LSH_TIMER_IDX})

# set SYCL implementation specific options # TODO: doesn't work as expected!
if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL")
#    if (SYCL_LSH_TARGET MATCHES "CPU")
#        # TODO: using this line results in many linker errors
##        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=cpu)
#        set(HIPSYCL_PLATFORM cpu)
#    elseif (SYCL_LSH_TARGET MATCHES "NVIDIA")
##        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=cuda)
#        set(HIPSYCL_PLATFORM cuda)
#    elseif (SYCL_LSH_TARGET MATCHES "AMD")
##        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=rocm)
#        set(HIPSYCL_PLATFORM rocm)
#    endif ()
#    # disable -Wunused-parameter if compiling for hipSYCL
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -Wno-unused-parameter)
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
target_compile_options(prog PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(prog PRIVATE ${SYCL_LSH_LIBRARY_NAME})


# add necessary SYCL flags to the target
if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL|ComputeCpp")
    add_sycl_to_target(TARGET prog SOURCES src/main.cpp)
endif ()


# generate documentation if requested
if (SYCL_LSH_ENABLE_DOCUMENTATION)
    message(STATUS "Using doxygen to generate documentation.")
    add_subdirectory(doc)
endif ()


# unset variables
unset(SUPPORTED_SYCL_LSH_IMPLEMENTATIONS)
unset(SYCL_LSH_IMPLEMENTATION)
unset(SUPPORTED_SYCL_LSH_TIMERS)
unset(SYCL_LSH_TIMER)
unset(SUPPORTED_SYCL_LSH_TARGETS)
unset(SYCL_LSH_TARGET)
unset(SYCL_LSH_ENABLE_DOCUMENTATION)
