/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Core header which includes every other necessary header file, i.e. \c \#include` <sycl_lsh/core.hpp>` is sufficient to use every
 *        important function or class.
 */

#ifndef SYCL_LSH_CORE_HPP
#define SYCL_LSH_CORE_HPP
#pragma once

// include all necessary public library headers
#include "sycl_lsh/constants.hpp"
#include "sycl_lsh/data_set.hpp"
#include "sycl_lsh/hash_function_types.hpp"
#include "sycl_lsh/matrix.hpp"
#include "sycl_lsh/mpi/communicator.hpp"
#include "sycl_lsh/mpi/environment.hpp"
#include "sycl_lsh/mpi/file_parser_types.hpp"
#include "sycl_lsh/mpi/main.hpp"
#include "sycl_lsh/nearest_neighbors.hpp"
#include "sycl_lsh/nearest_neighbors_result.hpp"
#include "sycl_lsh/options.hpp"
#include "sycl_lsh/profiler.hpp"

/// The main namespace. Nearly all functions are located in this namespace.
namespace sycl_lsh { }

/// This namespace is for implementation details only and **should not** be used directly bey users.
namespace sycl_lsh::detail { }

/// This namespace is for hashing related implementation details only and **should not** be used directly bey users.
namespace sycl_lsh::detail::hashing { }

/// This namespace is for wrapper classes and functions related to MPI.
namespace sycl_lsh::mpi { }

/// This namespace is for implementation details regarding the MPI functionality only and **should not** be used directly bey users.
namespace sycl_lsh::mpi::detail { }

#endif  // SYCL_LSH_CORE_HPP
