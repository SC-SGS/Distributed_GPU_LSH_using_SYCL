cmake_minimum_required(VERSION 3.13)

project("Distributed GPU LSH implementation using SYCL"
        VERSION 0.0.0
        DESCRIPTION "master theses Marcel Breyer"
        LANGUAGES C CXX
        HOMEPAGE_URL "https://gitlab-sim.informatik.uni-stuttgart.de/breyerml/distributed_gpu_lsh_using_sycl")

# cmake configuration options
option(ENABLE_TESTS "Generate tests" OFF)
option(ENABLE_DOCUMENTATION "Generate documentation" OFF)

# add custom cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

# set build type to Release if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Debug/Release)" FORCE)
endif ()

# create library
set(LIBRARY_NAME "lsh")
add_library(${LIBRARY_NAME})


# set needed compiler dependencies
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)


# find MPI and add it to the library target
find_package(MPI COMPONENTS C QUIET)
target_include_directories(${LIBRARY_NAME} PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC MPI::MPI_C)


# add custom hardcoded path so that clion can find the header files (NOT (!!!) needed to compile code)
# TODO: better
include_directories(/home/marcel/Programs/mpi/include)
include_directories(/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/hipsycl-0.8.0-zh3xxc5qkcrliknrratsbccxnlykpzqe/include)


# set header include directory
target_include_directories(${LIBRARY_NAME} PUBLIC include)

# set source include files
target_sources(${LIBRARY_NAME} PUBLIC
        src/dummy/dummy.cpp)

# create executable
add_executable(prog src/main.cpp)

# link against library
target_link_libraries(prog ${LIBRARY_NAME})


# generate documentation if requested
if (ENABLE_DOCUMENTATION)
    message(STATUS "Enabled generation of documentation using Doxygen")
    add_subdirectory(doc)
endif ()

# generate tests if requested
if (ENABLE_TESTS)
    message(STATUS "Enabled tests using google's test framework (GTest)")
    # install test framework in third_party directory
    include(GoogleTest)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif ()