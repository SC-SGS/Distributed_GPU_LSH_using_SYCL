cmake_minimum_required(VERSION 3.13)

# specify used SYCL implementation
set(SUPPORTED_SYCL_IMPLEMENTATIONS hipSYCL ComputeCpp)
if (NOT SYCL_IMPLEMENTATION)
    set(SYCL_IMPLEMENTATION hipSYCL)
endif ()
if (NOT ${SYCL_IMPLEMENTATION} IN_LIST SUPPORTED_SYCL_IMPLEMENTATIONS)
    message(FATAL_ERROR "SYCL implementation \"${SYCL_IMPLEMENTATION}\" currently not supported!")
else ()
    message(STATUS "Using \"${SYCL_IMPLEMENTATION}\" as SYCL implementation.")
endif ()


# set compiler wrapper based on used SYCL implementation
if (SYCL_IMPLEMENTATION MATCHES "hipSYCL")
    set(CMAKE_CXX_COMPILER syclcc-clang)
elseif (SYCL_IMPLEMENTATION MATCHES "ComputeCpp")
    set(CMAKE_CXX_COMPILER compute++)
endif()


project("Distributed GPU LSH implementation using SYCL"
        VERSION 1.0.0
        LANGUAGES CXX
        DESCRIPTION "master theses Marcel Breyer"
        HOMEPAGE_URL "https://gitlab-sim.informatik.uni-stuttgart.de/breyerml/distributed_gpu_lsh_using_sycl")


# cmake configuration options
option(ENABLE_DOCUMENTATION "Generate documentation" OFF)


# set timer behaviour
set(SUPPORTED_TIMERS NONE REDUCED_FILE DETAILED) # NONE = 0, REDUCED_FILE = 1, DETAILED = 2
if (NOT TIMER)
    set(TIMER DETAILED)
endif ()
if (NOT ${TIMER} IN_LIST SUPPORTED_TIMERS)
    message(FATAL_ERROR "Timer \"${TIMER}\" not supported!")
else ()
    message(STATUS "Using \"${TIMER}\" as timer.")
endif ()


# set supported SYCL targets
set(SUPPORTED_SYCL_TARGETS CPU NVIDIA AMD) # CPU = 0, NVIDIA = 1, AMD = 2
if (NOT SYCL_TARGET)
    set(SYCL_TARGET NVIDIA)
endif ()
if (NOT ${SYCL_TARGET} IN_LIST SUPPORTED_SYCL_TARGETS)
    message(FATAL_ERROR "SYCL target \"${SYCL_TARGET}\" not supported!")
else () 
    message(STATUS "Using \"${SYCL_TARGET}\" as SYCL target.")
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
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PRIVATE MPI::MPI_CXX)


# find OpenMP add it to the target
#find_package(OpenMP REQUIRED)
#target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PRIVATE OpenMP::OpenMP_CXX)


# set c++ standard
target_compile_features(${SYCL_LSH_LIBRARY_NAME} PUBLIC cxx_std_17)


# set library compile definitions
list(FIND SUPPORTED_SYCL_TARGETS "${SYCL_TARGET}" SYCL_TARGET_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PRIVATE SYCL_TARGET=${SYCL_TARGET_IDX})
list(FIND SUPPORTED_TIMERS "${TIMER}" TIMER_IDX)
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PRIVATE TIMER=${TIMER_IDX})


# set SYCL implementation specific options
if (SYCL_IMPLEMENTATION MATCHES "hipSYCL")
    if (SYCL_TARGET MATCHES "CPU")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PRIVATE --hipsycl-platform=cpu)
    elseif (SYCL_TARGET MATCHES "NVIDIA")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PRIVATE --hipsycl-platform=cuda)
    elseif (SYCL_TARGET MATCHES "AMD")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PRIVATE --hipsycl-platform=rocm)
    endif ()
elseif (SYCL_IMPLEMENTATION MATCHES "ComputeCpp")
    if (SYCL_TARGET MATCHES "NVIDIA")
        set(COMPUTECPP_BITCODE ptx64)
    endif ()
    if (SYCL_TARGET MATCHES "CPU")
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PRIVATE -sycl-host-only)
    else ()
        target_compile_options(${SYCL_LSH_LIBRARY_NAME} PRIVATE -sycl-driver)
    endif ()
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PRIVATE -lComputeCpp)
endif()


# create executable
add_executable(prog src/main.cpp)
target_compile_options(prog PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(prog PRIVATE ${SYCL_LSH_LIBRARY_NAME})


# generate documentation if requested
if (ENABLE_DOCUMENTATION)
    message(STATUS "Using doxygen to generate documentation.")
    add_subdirectory(doc)
endif ()


# unset variables
unset(SUPPORTED_SYCL_IMPLEMENTATIONS)
unset(SYCL_IMPLEMENTATION)
unset(SUPPORTED_TIMERS)
unset(TIMER)
unset(SUPPORTED_SYCL_TARGETS)
unset(SYCL_TARGET)
unset(ENABLE_DOCUMENTATION)


# add custom hardcoded path so that clion can find the header files (NOT (!!!) needed to compile code)
if ($ENV{CLION_IDE})
    include_directories({PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest-src/googletest/include)
    include_directories(/home/marcel/Programs/mpi/include)
    include_directories(/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/hipsycl-master-44bjhhamjo3w4k74hsanb73ulhc27now/include)
    include_directories(/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/llvm-10.0.0-uxnrorpisttivegjr4eqshrdva3heg55/include/c++/v1)

    file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/test/*.cpp)
    add_executable(test_prog ${SOURCES})
endif ()
