/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the factory functions for the hash functions classes.
 */

#ifndef SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTIONS_HPP
#define SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTIONS_HPP
#pragma once

#include "sycl_lsh/memory_layout.hpp"  // sycl_lsh::memory_layout

namespace sycl_lsh {

// forward declare hash functions classes
class random_projections;
class entropy_based;
class mixed_hash_functions;

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTIONS_HPP
