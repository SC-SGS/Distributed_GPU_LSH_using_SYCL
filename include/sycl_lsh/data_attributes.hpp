/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-29
 *
 * @brief Implements a @ref sycl_lsh::data_attributes class for managing attributes of the used data set represented by @ref sycl_lsh::data.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_ATTRIBUTES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_ATTRIBUTES_HPP

#include <sycl_lsh/memory_layout.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <ostream>
#include <type_traits>

namespace sycl_lsh {

    /**
     * @brief Class containing and managing the attributes of the data set represented by a @ref sycl_lsh::data object.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam index_type an integral type (used for indices)
     */
    template <memory_layout layout, typename index_type>
    struct data_attributes {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity check                                       //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_integral_v<index_type>, "The template parameter must be an integer type!");


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructors                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::data_attributes object representing the sizes of a @ref sycl_lsh::data object.
         * @param[in] total_size the **total** number of data points
         * @param[in] rank_size the number of data points on the current MPI rank
         * @param[in] dims the number of dimensions per data point
         */
        data_attributes(const index_type total_size, const index_type rank_size, const index_type dims)
                : total_size(total_size), rank_size(rank_size), dims(dims) { }
        /**
         * @brief Construct a new @ref sycl_lsh::data_attributes object as a copy from @p other.
         * @details The @ref sycl_lsh::memory_layout my differ, but the index_type must be the same.
         * @param[in] other the other @ref sycl_lsh::data_attributes object
         */
        template <memory_layout other_layout>
        data_attributes(const data_attributes<other_layout, index_type>& other)
                : total_size(other.total_size), rank_size(other.rank_size), dims(other.dims) { }


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
     * @tparam index_type an integral type (used for indices)
     * @param[in,out] out the output stream
     * @param[in] data_attr the @ref sycl_lsh::data_attributes
     * @return the output stream
     */
    template <memory_layout layout, typename index_type>
    std::ostream& operator<<(std::ostream& out, const data_attributes<layout, index_type>& data_attr) {
        out << fmt::format("memory_layout '{}'\n", layout);
        out << fmt::format("total_size {}\n", data_attr.total_size);
        out << fmt::format("rank_size {}\n", data_attr.rank_size);
        out << fmt::format("dims {}\n", data_attr.dims);
        return out;
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_ATTRIBUTES_HPP
