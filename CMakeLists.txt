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
        VERSION 0.0.0
        LANGUAGES CXX
        DESCRIPTION "master theses Marcel Breyer"
        HOMEPAGE_URL "https://gitlab-sim.informatik.uni-stuttgart.de/breyerml/distributed_gpu_lsh_using_sycl")


# cmake configuration options
option(ENABLE_TESTS "Generate tests" OFF)
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


# add custom cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")


# find boost
find_package(Boost 1.72 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


# find MPI and add it to the target
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
link_libraries(MPI::MPI_CXX)


# find OpenMP add it to the target
#find_package(OpenMP)
#link_libraries(OpenMP::OpenMP_CXX)


# set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


# set CMAKE_CXX_FLAGS based on CMAKE_BUILD_TYPE
string(APPEND CMAKE_CXX_FLAGS_INIT " ")
string(APPEND CMAKE_CXX_FLAGS_DEBUG_INIT " -g")
string(APPEND CMAKE_CXX_FLAGS_MINSIZEREL_INIT " -Os -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS_RELEASE_INIT " -O3 -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT " -O2 -g -DNDEBUG")


# include directory
include_directories(include)


# create executable
add_executable(prog src/main.cpp)
target_compile_options(prog PRIVATE -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-format-security)
target_link_libraries(prog PRIVATE -lstdc++fs)

list(FIND SUPPORTED_SYCL_TARGETS "${SYCL_TARGET}" SYCL_TARGET_IDX)
target_compile_definitions(prog PRIVATE SYCL_TARGET=${SYCL_TARGET_IDX})
list(FIND SUPPORTED_TIMERS "${TIMER}" TIMER_IDX)
target_compile_definitions(prog PRIVATE TIMER=${TIMER_IDX})


# set SYCL implementation specific options
if (SYCL_IMPLEMENTATION MATCHES "hipSYCL")
    if (SYCL_TARGET MATCHES "CPU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hipsycl-platform=cpu")
    elseif (SYCL_TARGET MATCHES "NVIDIA")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hipsycl-platform=cuda")
    elseif (SYCL_TARGET MATCHES "AMD")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hipsycl-platform=rocm")
    endif ()
elseif (SYCL_IMPLEMENTATION MATCHES "ComputeCpp")
    if (SYCL_TARGET MATCHES "NVIDIA")
        set(COMPUTECPP_BITCODE ptx64)
    endif ()
    if (SYCL_TARGET MATCHES "CPU")
        target_compile_options(prog PRIVATE -sycl-host-only)
    else ()
        target_compile_options(prog PRIVATE -sycl-driver)
    endif ()
    target_link_libraries(prog PRIVATE -lComputeCpp)
endif()


# unset variable
unset(SYCL_IMPLEMENTATION)




# generate documentation if requested
if (ENABLE_DOCUMENTATION)
    message(STATUS "Using doxygen to generate documentation.")
    add_subdirectory(doc)
endif ()


# generate tests if requested
if (ENABLE_TESTS)
    # check the set CMAKE_BUILD_TYPE
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to 'Debug' as none was specified and tests are enabled!")
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type" FORCE)
    elseif(NOT(CMAKE_BUILD_TYPE MATCHES "Debug"))
        message(FATAL_ERROR "You have to use the build type 'Debug' as long as tests are enabled!")
    endif ()

    message(STATUS "Enabled tests using google's test framework (GTest)")
    # install test framework in third_party directory
    include(GoogleTest)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
else ()
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to 'Release' as none was specified and tests are disabled!")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type" FORCE)
    endif ()
endif ()
message(STATUS "Using \"${CMAKE_BUILD_TYPE}\" as build type.")


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
