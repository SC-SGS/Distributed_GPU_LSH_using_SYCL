/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-14
 *
 * @brief Implements the @ref knn class representing the result of the k-nearest-neighbour search.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP

#include <vector>

#include <config.hpp>
#include <options.hpp>
#include <data.hpp>


/**
 * @brief Class representing the result of the k-nearest-neighbour search.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class knn {
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;
public:


    /// The SYCL buffer holding all data: `buffer.get_count() == data::size * options::k`.
    sycl::buffer<index_type, 1> buffer;


    /**
     * @brief Returns the ids (index) of the k-nearest-neighbors found for @p point.
     * @param[in] point the data point
     * @return the indices of the k-nearest-neighbours of @p point
     *
     * @attention Copies the ids (indices) to the result vector!
     * @pre @p point **must** be greater or equal than `0` and less than `data::dims`.
     */
    std::vector<index_type> get_knn_ids(const index_type point) const {
        DEBUG_ASSERT(0 <= point && point < data_.size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.size);

        std::vector<index_type> res(opt_.k);
        auto acc = buffer.template get_access<sycl::access::mode::read>();
        for (index_type i = 0; i < opt_.k; ++i) {
            res[i] = acc[this->get_linear_id(point, i)];
        }
        return res;
    }
    /**
     * @brief Returns the data points of the k-nearest-neighbors found for @p point.
     * @tparam knn_points_layout the @ref memory_layout used for the k-nearest-neighbours
     * @param[in] point the data point
     * @return the data points of the k-nearest-neighbours of @p point
     *
     * @attention Copies the underlying points to the result vector!
     * @pre @p point **must** be greater or equal than `0` and less than `data::dims`.
     */
    template <memory_layout knn_points_layout>
    std::vector<real_type> get_knn_points(const index_type point) const {
        DEBUG_ASSERT(0 <= point && point < data_.size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.size);

        std::vector<real_type> res(opt_.k * data_.dims);
        auto acc_knn = buffer.template get_access<sycl::access::mode::read>();
        auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>();
        for (index_type i = 0; i < opt_.k; ++i) {
            // get the knn index
            const index_type knn_id = acc_knn[this->get_linear_id(point, i)];
            for (index_type dim = 0; dim < data_.dims; ++dim) {
                // get the concrete data point value of the current dimension
                const real_type knn_dim = acc_data[data_.get_linear_id(knn_id, dim)];
                if constexpr (knn_points_layout == memory_layout::aos) {
                    // Array of Structs
                    res[i * data_.dims + dim] = knn_dim;
                } else {
                    // Structs of Array
                    res[dim * opt_.k + i] = knn_dim;
                }
            }
        }
        return res;
    }

    /**
     * @brief Converts a two-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] point the provided data point
     * @param[in] i the provided knn index
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p point **must** be greater or equal than `0` and less than `data::dims`.
     * @pre @p i **must** be greater or equal than `0` and less than `options::k`.
     */
    [[nodiscard]] constexpr index_type get_linear_id(const index_type point, const index_type i) const noexcept {
        DEBUG_ASSERT(0 <= point && point < data_.size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.size);
        DEBUG_ASSERT(0 <= i && i < opt_.k, "Out-of-bounce access!: 0 <= {} < {}", i, opt_.k);

        if constexpr (layout == memory_layout::aos) {
            return point * opt_.k + i;
        } else {
            return i * data_.size + point;
        }
    }

    /**
     * @brief Returns the @ref options object which has been used to create this @ref knn object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    /**
     * @brief Returns the @ref data object which has been used to create this @ref knn object.
     * @return the @ref data object (`[[nodiscard]]`)
     */
    [[nodiscard]] Data& get_data() const noexcept { return data_; }
    /**
     * @brief Returns the specified @ref memory_layout (*Array of Structs* or *Struct of Arrays*).
     * @return the specified @ref memory_layout (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept { return layout; }

private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Data_>
    friend knn<layout_, typename Data_::options_type, Data_> make_knn(Data_&);
    /// Befriend knn class (including the one with another @ref memory_layout).
    template <memory_layout, typename, typename>
    friend class knn;


    /**
     * @brief Construct a new @ref knn object given the options in @p opt and sizes in @p data.
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     */
    knn(const Options& opt, Data& data) : opt_(opt), data_(data), buffer(data.size * opt.k) { }

    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;
};


/**
 * @brief Factory function for creating a new @ref knn object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[in] data the used data object
 * @return the newly constructed @ref knn object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline knn<layout, typename Data::options_type, Data> make_knn(Data& data) {
    return knn<layout, typename Data::options_type, Data>(data.get_options(), data);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
