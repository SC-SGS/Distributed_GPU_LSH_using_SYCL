cmake_minimum_required(VERSION 3.13)

# specify used SYCL implementation
set(SUPPORTED_SYCL_LSH_IMPLEMENTATIONS hipSYCL ComputeCpp)
if (NOT SYCL_LSH_IMPLEMENTATION)
    set(SYCL_LSH_IMPLEMENTATION hipSYCL)
endif ()
if (NOT ${SYCL_LSH_IMPLEMENTATION} IN_LIST SUPPORTED_SYCL_LSH_IMPLEMENTATIONS)
    message(FATAL_ERROR "SYCL implementation \"${SYCL_LSH_IMPLEMENTATION}\" currently not supported!")
else ()
    message(STATUS "Using \"${SYCL_LSH_IMPLEMENTATION}\" as SYCL implementation.")
endif ()


# set compiler wrapper based on used SYCL implementation
if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL")
    set(CMAKE_CXX_COMPILER syclcc-clang)
elseif (SYCL_LSH_IMPLEMENTATION MATCHES "ComputeCpp")
    set(CMAKE_CXX_COMPILER compute++)
endif()


project("Distributed GPU LSH implementation using SYCL"
        VERSION 1.0.0
        LANGUAGES CXX
        DESCRIPTION "master theses Marcel Breyer"
        HOMEPAGE_URL "https://gitlab-sim.informatik.uni-stuttgart.de/breyerml/distributed_gpu_lsh_using_sycl")


# cmake configuration options
option(SYCL_LSH_ENABLE_DEBUG "Enable debugging macros" OFF)
option(SYCL_LSH_ENABLE_DOCUMENTATION "Generate documentation" OFF)


# set timer behaviour
#set(SUPPORTED_TIMERS NONE REDUCED_FILE DETAILED) # NONE = 0, REDUCED_FILE = 1, DETAILED = 2
#if (NOT TIMER)
#    set(TIMER DETAILED)
#endif ()
#if (NOT ${TIMER} IN_LIST SUPPORTED_TIMERS)
#    message(FATAL_ERROR "Timer \"${TIMER}\" not supported!")
#else ()
#    message(STATUS "Using \"${TIMER}\" as timer.")
#endif ()


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
add_library(${SYCL_LSH_LIBRARY_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/communicator_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/file_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/window_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/communicator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/errhandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/argv_parser.cpp
)

# set include directory
target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC include)


# find MPI and add it to the target
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC MPI::MPI_CXX)


# find fmt lib and add it to the target
find_package(fmt REQUIRED)
if (NOT DEFINED FMT_HEADER_ONLY OR ${FMT_HEADER_ONLY} MATCHES "0")
    message(STATUS "Linking against fmt lib.")
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC fmt::fmt)
else ()
    message(STATUS "Using fmt lib in its header only mode.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC FMT_HEADER_ONLY=1)
endif ()


# set c++ standard
target_compile_features(${SYCL_LSH_LIBRARY_NAME} PUBLIC cxx_std_17)
target_compile_options(${SYCL_LSH_LIBRARY_NAME} PRIVATE -Wno-format-security)


# set library compile definitions
if (SYCL_LSH_ENABLE_DEBUG)
    message(STATUS "Enabled debugging.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC -DSYCL_LSH_DEBUG)

#    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
#    target_link_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
endif ()
list(FIND SUPPORTED_SYCL_LSH_TARGETS "${SYCL_LSH_TARGET}" SYCL_LSH_TARGET_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TARGET=${SYCL_LSH_TARGET_IDX})
#list(FIND SUPPORTED_TIMERS "${TIMER}" TIMER_IDX)
#target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC TIMER=${TIMER_IDX})


# set SYCL implementation specific options
if (SYCL_LSH_IMPLEMENTATION MATCHES "hipSYCL")
    if (SYCL_LSH_TARGET MATCHES "CPU")
        # TODO: using this line results in many linker errors
#        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=cpu)
    elseif (SYCL_LSH_TARGET MATCHES "NVIDIA")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=cuda)
    elseif (SYCL_LSH_TARGET MATCHES "AMD")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC --hipsycl-platform=rocm)
    endif ()
    # disable -Wunused-parameter if compiling for hipSYCL
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -Wno-unused-parameter)
elseif (SYCL_LSH_IMPLEMENTATION MATCHES "ComputeCpp")
    if (SYCL_LSH_TARGET MATCHES "NVIDIA")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -sycl-target ptx64)
    endif ()
    if (SYCL_LSH_TARGET MATCHES "CPU")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -sycl-host-only)
    else ()
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -sycl-driver)
    endif ()
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -no-serial-memop)
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC -lComputeCpp -lstdc++fs)
endif()


# create executable
add_executable(prog src/main.cpp)
target_compile_options(prog PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(prog PRIVATE ${SYCL_LSH_LIBRARY_NAME})


# generate documentation if requested
if (SYCL_LSH_ENABLE_DOCUMENTATION)
    message(STATUS "Using doxygen to generate documentation.")
    add_subdirectory(doc)
endif ()


# unset variables
unset(SUPPORTED_SYCL_LSH_IMPLEMENTATIONS)
unset(SYCL_LSH_IMPLEMENTATION)
#unset(SUPPORTED_TIMERS)
#unset(TIMER)
unset(SUPPORTED_SYCL_LSH_TARGETS)
unset(SYCL_LSH_TARGET)
unset(SYCL_LSH_ENABLE_DOCUMENTATION)
