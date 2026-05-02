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
 */
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
    data_attributes(const index_type total_size, const index_type rank_size, const index_type dims) :
        total_size{ total_size },
        rank_size{ rank_size },
        dims{ dims } { }

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
 * @param[in,out] out the output stream
 * @param[in] data_attr the @ref sycl_lsh::data_attributes
 * @return the output stream
 */
inline std::ostream &operator<<(std::ostream &out, const data_attributes &data_attr) {
    out << fmt::format("total_size {}\n"
                       "rank_size {}\n"
                       "dims {}",
                       data_attr.total_size,
                       data_attr.rank_size,
                       data_attr.dims);
    return out;
}

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::data_attributes> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_DATA_ATTRIBUTES_HPP
