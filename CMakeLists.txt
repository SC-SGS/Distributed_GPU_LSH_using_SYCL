cmake_minimum_required(VERSION 3.13)

project("Distributed GPU LSH implementation using SYCL"
        VERSION 0.0.0
        DESCRIPTION "master theses Marcel Breyer"
        LANGUAGES CXX
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

# set general compiler flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # clang and GCC specific flags
    set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Wextra-semi -Wnull-dereference -Wswitch-enum)

    # clang specific flags
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND WARNING_FLAGS -Wno-gnu-zero-variadic-macro-arguments)
    endif ()

    # GCC specific flags
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND WARNING_FLAGS -Wduplicated-cond -Wduplicated-branches -Wsuggest-override)
    endif ()

    message(STATUS "Enabled '${WARNING_FLAGS}' compiler flags to catch more potential errors")
    target_compile_options(${LIBRARY_NAME} PUBLIC ${WARNING_FLAGS})
endif ()


# find MPI and add it to the library target
find_package(MPI REQUIRED)
target_include_directories(${LIBRARY_NAME} INTERFACE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} INTERFACE MPI::MPI_CXX)


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
    include_directories(${LIBRARY_NAME} PUBLIC third_party)
endif ()