/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 *
 * @brief Core header which includes every other necessary header file, i.e. \c \#include` <sycl_lsh/core.hpp>` is sufficient to use every
 *        important function or class.
 */
#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CORE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CORE_HPP

// include all necessary library headers
#include <sycl_lsh/exceptions/communicator_exception.hpp>
#include <sycl_lsh/exceptions/file_exception.hpp>
#include <sycl_lsh/exceptions/window_exception.hpp>

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/errhandler.hpp>
#include <sycl_lsh/mpi/main.hpp>


/// The main namespace. Nearly all functions are located in this namespace.
namespace sycl_lsh { }
/// This namespace is for implementation details only and **should not** be used directly bey users.
namespace sycl_lsh::detail { }

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CORE_HPP
