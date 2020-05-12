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
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the build type" FORCE)
endif ()
message(STATUS "cmake build type is: ${CMAKE_BUILD_TYPE}")


# set source include files
set(SOURCES
#        src/hash_table.cpp
)


# set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# set CMAKE_CXX_FLAGS based on CMAKE_BUILD_TYPE
string(APPEND CMAKE_CXX_FLAGS_INIT " ")
string(APPEND CMAKE_CXX_FLAGS_DEBUG_INIT " -g")
string(APPEND CMAKE_CXX_FLAGS_MINSIZEREL_INIT " -Os -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS_RELEASE_INIT " -O3 -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT " -O2 -g -DNDEBUG")

if (NOT DEFINED ENV{CLION_IDE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hipsycl-platform=cpu") # this line breaks GTest
endif ()

# include directory
include_directories(include)

# create executable
add_executable(prog src/main.cpp ${SOURCES})
target_compile_options(prog PRIVATE -Wno-format-security)
target_link_libraries(prog PRIVATE -lstdc++fs) # don't know why this is necessary


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



# add custom hardcoded path so that clion can find the header files (NOT (!!!) needed to compile code)
if ($ENV{CLION_IDE})
    include_directories(/home/marcel/Programs/mpi/include)
    include_directories(/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/hipsycl-master-44bjhhamjo3w4k74hsanb73ulhc27now/include)
    include_directories(/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/llvm-10.0.0-uxnrorpisttivegjr4eqshrdva3heg55/include/c++/v1)
endif ()