/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Core header which includes every other necessary header file, i.e. \c \#include` <sycl_lsh/core.hpp>` is sufficient to use every
 *        important function or class.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CORE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CORE_HPP

// include all necessary library headers
#include <fmt/chrono.h>

#include <sycl_lsh/hash_functions/entropy_based.hpp>
#include <sycl_lsh/hash_functions/hash_functions.hpp>
#include <sycl_lsh/hash_functions/mixed_hash_functions.hpp>
#include <sycl_lsh/hash_functions/random_projections.hpp>

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/errhandler.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/main.hpp>
#include <sycl_lsh/mpi/timer.hpp>

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/data.hpp>
#include <sycl_lsh/hash_tables.hpp>
#include <sycl_lsh/knn.hpp>
#include <sycl_lsh/options.hpp>

/// The main namespace. Nearly all functions are located in this namespace.
namespace sycl_lsh { }
/// This namespace is for wrapper classes and functions related to MPI.
namespace sycl_lsh::mpi { }
/// This namespace is for implementation details only and **should not** be used directly bey users.
namespace sycl_lsh::detail { }

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CORE_HPP
