#find_program(SYCLCC_EXECUTABLE syclcc-clang
#        PATH_SUFFIXES bin)

set(SYCLCC_EXECUTABLE "/home/marcel/Programs/spack/opt/spack/linux-linuxmint19-zen/gcc-8.4.0/hipsycl-0.8.0-zh3xxc5qkcrliknrratsbccxnlykpzqe/bin/syclcc-clang")

if(NOT SYCLCC_EXECUTABLE)
    message(SEND_ERROR "Could not find hipSYCL syclcc-clang compiler")
endif()

set(CMAKE_C_COMPILER    ${SYCLCC_EXECUTABLE})
set(CMAKE_CXX_COMPILER  ${SYCLCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 14)

add_library(SYCL::SYCL INTERFACE IMPORTED GLOBAL)