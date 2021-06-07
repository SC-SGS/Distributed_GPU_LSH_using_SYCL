cmake_minimum_required(VERSION 3.19)


project("Distributed k-nearest Neighbors using Locality Sensitive Hashing and SYCL"
        VERSION 2.0.0
        LANGUAGES CXX
        DESCRIPTION "Master Thesis Marcel Breyer")


message(STATUS "Using \"${SYCL_LSH_SYCL_IMPLEMENTATION}\" as SYCL implementation.")


# get target hardware
# omp = CPU, cuda = NVIDIA GPUs, hip = AMD GPUs, spirv = Intel GPUs
if (DEFINED SYCL_LSH_TARGET)
  set(SYCL_LSH_TARGET_LIST ${SYCL_LSH_TARGET})
elseif (DEFINED ENV{SYCL_LSH_TARGET})
  set(SYCL_LSH_TARGET_LIST $ENV{SYCL_LSH_TARGET})
else ()
  message(FATAL_ERROR "Either the command line argument '-DSYCL_LSH_TARGET' must be present or the environment variable 'SYCL_LSH_TAREGT' must be defined!")
endif ()
list(LENGTH SYCL_LSH_TARGET_LIST SYCL_LSH_TARGET_LIST_LEN)
if (NOT ${SYCL_LSH_TARGET_LIST_LEN} MATCHES 1)
  message(FATAL_ERROR "Currently only exactly one target architecture is supported!")
endif ()
string(FIND "${SYCL_LSH_TARGET_LIST}" ":" SYCL_LSH_ARCH_POS)
string(SUBSTRING "${SYCL_LSH_TARGET_LIST}" 0 ${SYCL_LSH_ARCH_POS} SYCL_LSH_ARCH)
set(SYCL_LSH_TARGETS omp cuda hip spirv)
list(FIND SYCL_LSH_TARGETS "${SYCL_LSH_ARCH}" SYCL_LSH_TARGET_IDX)
if (${SYCL_LSH_TARGET_IDX} MATCHES -1)
  message(FATAL_ERROR "Unrecognized target \"${SYCL_LSH_ARCH}\"!")
endif ()
message(STATUS "Using \"${SYCL_LSH_ARCH}\" as SYCL target architecture.")


# find different SYCL implementations
if (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "hipsycl")

  set(HIPSYCL_TARGETS ${SYCL_LSH_TARGET_LIST})

  find_package(hipSYCL CONFIG REQUIRED)
  message(STATUS "Found hipSYCL.")

elseif (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "computecpp")

  # must explicitly set bitcode to ptx64 if targetting NVIDIA GPUs using ComputeCpp
  if (SYCL_LSH_ARCH MATCHES "cuda")
    set(COMPUTECPP_BITCODE ptx64 CACHE INTERNAL "")
  else ()
    unset(COMPUTECPP_BITCODE CACHE)
  endif ()

  list(APPEND CMAKE_MODULE_PATH "${COMPUTECPP_MODULE_PATH}")
  include(FindComputeCpp)
  include_directories(${COMPUTE_CPP_INCLUDE_DIRECTORIES})
  message(STATUS "Found ComputeCpp.")

elseif (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "dpcpp")

  message(STATUS "Found dpcpp.")

endif ()



# check for correct CMAKE_BUILD_TYPE
set(SYCL_LSH_BUILD_TYPES Debug Release MinSizeRel RelWithDebInfo)
if (NOT CMAKE_BUILD_TYPE IN_LIST SYCL_LSH_BUILD_TYPES)
  string(REPLACE ";" ", " SYCL_LSH_BUILD_TYPES_OUT "${SYCL_LSH_BUILD_TYPES}")
  message(FATAL_ERROR "CMAKE_BUILD_TYPE \"${CMAKE_BUILD_TYPE}\" not supported!\nMust be one if ${SYCL_LSH_BUILD_TYPES_OUT}")
else ()
  message(STATUS "Using \"${CMAKE_BUILD_TYPE}\" as build type.")
endif ()


# create library
set(SYCL_LSH_LIBRARY_NAME "sycl_lsh")
add_library(${SYCL_LSH_LIBRARY_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/argv_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/detail/sycl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/detail/utility.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/device_selector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/communicator_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/file_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/not_implemented_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/exceptions/window_exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/communicator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/errhandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_lsh/mpi/timer.cpp
)

# set include directory
target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC include)

# set c++ standard
target_compile_features(${SYCL_LSH_LIBRARY_NAME} PUBLIC cxx_std_17)

# link against pthread
target_link_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -lpthread)


# set SYCL implementation specific options
if (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "hipsycl")

  #find_package(OpenMP REQUIRED)
  #target_include_directories(${SYCL_LSH_LIBRARY_NAME} PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
  #target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)

elseif (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "computecpp")

elseif (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "dpcpp")

  if (SYCL_LSH_ARCH MATCHES "cuda")
    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
    target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
  endif ()

endif ()



# find MPI library and add it to the target
find_package(MPI REQUIRED)
message(STATUS "Found MPI message passing library.")
include_directories(${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC MPI::MPI_CXX)


# find {fmt} formatting library and add it to the target
find_package(fmt REQUIRED)
message(STATUS "Found {fmt} formatting library.")
target_link_libraries(${SYCL_LSH_LIBRARY_NAME} PUBLIC fmt::fmt)


# add debug mode
option(SYCL_LSH_ENABLE_DEBUG "Enable debug mode using assertion macros." OFF)
if (SYCL_LSH_ENABLE_DEBUG)
    message(STATUS "Enabled sycl_lsh debug mode.")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC -DSYCL_LSH_DEBUG)

    # sanitizers
#    target_compile_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
#    target_link_options(${SYCL_LSH_LIBRARY_NAME} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address,undefined,integer)
endif ()


# set timer behavior
set(SYCL_LSH_TIMERS NONE NON_BLOCKING BLOCKING) # NONE = 0, NON_BLOCKING = 1, BLOCKING = 2
set(SYCL_LSH_TIMER BLOCKING CACHE STRING "The used timer implementation.")
set_property(CACHE SYCL_LSH_TIMER PROPERTY STRINGS ${SYCL_LSH_TIMERS})
if (NOT SYCL_LSH_TIMER IN_LIST SYCL_LSH_TIMERS)
    string(REPLACE ";" ", " SYCL_LSH_TIMERS_OUT "${SYCL_LSH_TIMERS}")
    message(FATAL_ERROR "Timer \"${SYCL_LSH_TIMER}\" not supported!\nMust be one of: ${SYCL_LSH_TIMERS_OUT}")
else ()
    message(STATUS "Using \"${SYCL_LSH_TIMER}\" as timer implementation.")

    list(FIND SYCL_LSH_TIMERS "${SYCL_LSH_TIMER}" SYCL_LSH_TIMER_IDX)
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TIMER=${SYCL_LSH_TIMER_IDX})
endif ()


# enable benchmarking using an output file
set(SYCL_LSH_BENCHMARK "" CACHE STRING "The path to the benchmarking file if benchmarking should be enabled.")
if (NOT "${SYCL_LSH_BENCHMARK}" STREQUAL "")
    # in order to benchmark the code, a timer must be enabled
    if (SYCL_LSH_TIMER MATCHES "NONE")
        message(FATAL_ERROR "In order to enable benchmarking, a valid timer (NON_BLOCKING or BLOCKING) must be enabled!")
    endif ()
    message(STATUS "Enabled benchmarking using the file: \"${SYCL_LSH_BENCHMARK}\"")
    target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_BENCHMARK="${SYCL_LSH_BENCHMARK}")
endif ()


# propagate SYCL_LSH_IMPLEMENTATION to library
set(SYCL_LSH_SYCL_IMPLEMENTATIONS hipsycl computecpp dpcpp)
list(FIND SYCL_LSH_SYCL_IMPLEMENTATIONS "${SYCL_LSH_SYCL_IMPLEMENTATION}" SYCL_LSH_SYCL_IMPLEMENTATION_IDX)
if (${SYCL_LSH_SYCL_IMPLEMENTATION_IDX} MATCHES -1)
  message(FATAL_ERROR "Unrecognized SYCL implementation \"${SYCL_LSH_SYCL_IMPLEMENTATION}\"!")
endif ()
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_IMPLEMENTATION=${SYCL_LSH_SYCL_IMPLEMENTATION_IDX})

# propagate SYCL_LSH_TARGET to library
target_compile_definitions(${SYCL_LSH_LIBRARY_NAME} PUBLIC SYCL_LSH_TARGET=${SYCL_LSH_TARGET_IDX})



# create executable
set(SYCL_LSH_EXECUTABLE_NAME prog)
add_executable(prog src/main.cpp)
target_compile_options(${SYCL_LSH_EXECUTABLE_NAME} PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(${SYCL_LSH_EXECUTABLE_NAME} PRIVATE ${SYCL_LSH_LIBRARY_NAME})


# add necessary SYCL flags to the target
if (SYCL_LSH_SYCL_IMPLEMENTATION MATCHES "hipsycl|computecpp")
    add_sycl_to_target(TARGET ${SYCL_LSH_EXECUTABLE_NAME} SOURCES src/main.cpp)
endif ()



# generate documentation if requested
option(SYCL_LSH_ENABLE_DOCUMENTATION "Enable the generation of the documentation using Doxygen." OFF)
if (SYCL_LSH_ENABLE_DOCUMENTATION)
    message(STATUS "Using Doxygen to generate documentation.")
    add_subdirectory(doc)
endif ()
