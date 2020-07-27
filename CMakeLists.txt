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
option(ENABLE_GPU "Enable the usage of GPUs" ON)
option(ENABLE_TIMING "Enable the timing of various functions" OFF)


# add custom cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")


# generate timing information if requested
if (ENABLE_TIMING)
    message(STATUS "Enabled timing")
    add_definitions(-DENABLE_TIMING)
else ()
    message(STATUS "Disabled timing")
endif ()


# find boost
find_package(Boost 1.72 REQUIRED PATHS /home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/boost-1.72.0-zex4jeneghfjf4xvolu2lzqor7urievz)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

# find MPI and add it to the library target
find_package(MPI)
include_directories(${MPI_CXX_INCLUDE_DIRS})
link_libraries(MPI::MPI_CXX)



# set source include files
set(SOURCES )


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

    if (ENABLE_GPU)
        message(STATUS "Enabled the usage of GPUs")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hipsycl-platform=cpu") # this line breaks GTest
        message(STATUS "Disabled the usage of GPUs")
    endif ()
endif ()


# include directory
include_directories(include)

# create executable
add_executable(prog src/main.cpp ${SOURCES})
target_compile_options(prog PRIVATE -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-format-security)
target_compile_definitions(prog PRIVATE ENABLE_GPU)
target_link_libraries(prog PRIVATE -lstdc++fs) # don't know why this is necessary


# generate documentation if requested
if (ENABLE_DOCUMENTATION)
    message(STATUS "Enabled generation of documentation using Doxygen")
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
message(STATUS "CMAKE_BUILD_TYPE is: ${CMAKE_BUILD_TYPE}")


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