/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a @ref sycl_lsh::data_attributes class for managing attributes of the used data set represented by @ref sycl_lsh::data.
 */

#ifndef SYCL_LSH_DATA_ATTRIBUTES_HPP
#define SYCL_LSH_DATA_ATTRIBUTES_HPP
#pragma once

#include "sycl_lsh/constants.hpp"      // sycl_lsh::index_type
#include "sycl_lsh/memory_layout.hpp"  // sycl_lsh::memory_layout

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <ostream>  // std::ostream

namespace sycl_lsh {

/**
 * @brief Class containing and managing the attributes of the data set represented by a @ref sycl_lsh::data object.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct data_attributes {
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructors                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::data_attributes object representing the sizes of a @ref sycl_lsh::data object.
     * @param[in] total_size the **total** number of data points
     * @param[in] rank_size the number of data points on the current MPI rank
     * @param[in] dims the number of dimensions per data point
     */
    data_attributes(const index_type total_size, const index_type rank_size, const index_type dims) : total_size{ total_size }, rank_size{ rank_size }, dims{ dims } {}
    /**
     * @brief Construct a new @ref sycl_lsh::data_attributes object as a copy from @p other.
     * @details The @ref sycl_lsh::memory_layout my differ, but the index_type must be the same.
     * @param[in] other the other @ref sycl_lsh::data_attributes object
     */
    template <memory_layout other_layout>
    data_attributes(const data_attributes<other_layout> &other) : total_size{ other.total_size }, rank_size{ other.rank_size }, dims{ other.dims } {}

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                 attributes                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /// The **total** number of data points of the used data set.
    const index_type total_size;
    /// The number of data points on **the current** MPI rank.
    const index_type rank_size;
    /// The number of dimensions of each data point of the used data set.
    const index_type dims;
};

/**
 * @brief Print all attributes set in @p data_attr to the output stream @p out.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @param[in,out] out the output stream
 * @param[in] data_attr the @ref sycl_lsh::data_attributes
 * @return the output stream
 */
template <memory_layout layout>
std::ostream &operator<<(std::ostream &out, const data_attributes<layout> &data_attr) {
    out << fmt::format("memory_layout '{}'\n"
                       "total_size {}\n"
                       "rank_size {}\n"
                       "dims {}",
                       layout,
                       data_attr.total_size,
                       data_attr.rank_size,
                       data_attr.dims);
    return out;
}

}  // namespace sycl_lsh

template <sycl_lsh::memory_layout layout>
struct fmt::formatter<sycl_lsh::data_attributes<layout>> : fmt::ostream_formatter {};

#endif  // SYCL_LSH_DATA_ATTRIBUTES_HPP
