cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_COMPILER syclcc-clang)

project("Distributed GPU LSH implementation using SYCL"
        VERSION 0.0.0
        LANGUAGES CXX
        DESCRIPTION "master theses Marcel Breyer"
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


# find MPI and add it to the library target
#find_package(MPI COMPONENTS C QUIET)
#target_include_directories(${LIBRARY_NAME} PUBLIC ${MPI_C_INCLUDE_DIRS})
#target_link_libraries(${LIBRARY_NAME} PUBLIC MPI::MPI_C)

#find_package(hipSYCL)


# add custom hardcoded path so that clion can find the header files (NOT (!!!) needed to compile code)
if ($ENV{CLION_IDE})
    include_directories(/home/marcel/Programs/mpi/include)
    include_directories(/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/hipsycl-master-44bjhhamjo3w4k74hsanb73ulhc27now/include)
endif ()


# set source include files
set(SOURCES
        src/hash_table.cpp
        )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#if (NOT DEFINED ENV{CLION_IDE})
#    set(CMAKE_CXX_FLAGS "--hipsycl-platform=cpu")
#endif ()

include_directories(include)

# create executable
add_executable(prog src/main.cpp ${SOURCES})
target_compile_options(prog PRIVATE -Wno-format)

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