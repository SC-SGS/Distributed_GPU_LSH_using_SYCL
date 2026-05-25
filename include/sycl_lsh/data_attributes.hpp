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

#include "sycl_lsh/constants.hpp"  // sycl_lsh::index_type

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::ostream forward declaration

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
     * @param[in] total_size_p the **total** number of data points
     * @param[in] rank_size_p the number of data points on the current MPI rank
     * @param[in] dims_p the number of dimensions per data point
     */
    data_attributes(index_type total_size_p, index_type rank_size_p, index_type dims_p);

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
std::ostream &operator<<(std::ostream &out, const data_attributes &data_attr);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::data_attributes> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_DATA_ATTRIBUTES_HPP
