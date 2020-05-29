/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-29
 *
 * @brief Implements the @ref knn class representing the result of the k-nearest-neighbor search.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP

#include <vector>

#include <config.hpp>
#include <options.hpp>
#include <data.hpp>


/**
 * @brief Class representing the result of the k-nearest-neighbor search.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class knn {
public:
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;


    /// The SYCL buffer holding all data: `buffer.get_count() == data::size * knn::k`.
    sycl::buffer<index_type, 1> buffer;
    /// The number of nearest neighbors to search for.
    const index_type k;


    /**
     * @brief Returns the ids (index) of the k-nearest-neighbors found for @p point.
     * @param[in] point the data point
     * @return the indices of the k-nearest-neighbors of @p point
     *
     * @attention Copies the ids (indices) to the result vector!
     * @pre @p point **must** be greater or equal than `0` and less than `data::dims`.
     */
    std::vector<index_type> get_knn_ids(const index_type point) {
        DEBUG_ASSERT(0 <= point && point < data_.size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.size);

        std::vector<index_type> res(k);
        auto acc = buffer.template get_access<sycl::access::mode::read>();
        for (index_type i = 0; i < k; ++i) {
            res[i] = acc[this->get_linear_id(point, i)];
        }
        return res;
    }
    /**
     * @brief Returns the data points of the k-nearest-neighbors found for @p point.
     * @tparam knn_points_layout the @ref memory_layout used for the k-nearest-neighbors
     * @param[in] point the data point
     * @return the data points of the k-nearest-neighbors of @p point
     *
     * @attention Copies the underlying points to the result vector!
     * @pre @p point **must** be greater or equal than `0` and less than `data::dims`.
     */
    template <memory_layout knn_points_layout>
    std::vector<real_type> get_knn_points(const index_type point) {
        DEBUG_ASSERT(0 <= point && point < data_.size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.size);

        std::vector<real_type> res(k * data_.dims);
        auto acc_knn = buffer.template get_access<sycl::access::mode::read>();
        auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>();
        for (index_type i = 0; i < k; ++i) {
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
                    res[dim * k + i] = knn_dim;
                }
            }
        }
        return res;
    }

    template <memory_layout new_layout>
    [[nodiscard]] knn<new_layout, Options, Data> get_as()
//            __attribute__((diagnose_if(new_layout == layout, "new_layout == layout (simple copy)", "warning")))
    {
        knn<new_layout, Options, Data> new_knn(k, data_);
        auto acc_this = buffer.template get_access<sycl::access::mode::read>();
        auto acc_new = new_knn.buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type s = 0; s < data_.size(); ++s) {
            for (index_type nn = 0; nn < k; ++nn) {
                // transform memory layout
                acc_new[new_knn.get_linear_id(s, nn)] = acc_this[this->get_linear_id(s, nn)];
            }
        }
        return new_knn;
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
        DEBUG_ASSERT(0 <= i && i < k, "Out-of-bounce access!: 0 <= {} < {}", i, k);

        if constexpr (layout == memory_layout::aos) {
            return point * k + i;
        } else {
            return i * data_.size + point;
        }
    }

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

    /**
     * @brief Saves the nearest-neighbors to @p file using the current @ref memory_layout.
     * @details The content of @p file is overwritten if it already exists.
     * @param[in] file the name of the @p file
     *
     * @throw std::invalid_argument if @p file can't be opened or created.
     */
    void save(const std::string& file) {
        std::ofstream out(file, std::ofstream::trunc);
        if (out.bad()) {
            // something went wrong while opening/creating the file
            throw std::invalid_argument("Can't write to file '" + file + "'!");
        }
        auto acc = buffer.template get_access<sycl::access::mode::read>();
        for (index_type point = 0; point < data_.size; ++point) {
            out << acc[this->get_linear_id(point, 0)];
            for (index_type i = 1; i < k; ++i) {
                out << ',' << acc[this->get_linear_id(point, i)];
            }
            out << '\n';
        }
    }

private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Data_, typename Options_>
    friend knn<layout_, Options_, Data_> make_knn(typename Options_::index_type, Data_&);
    /// Befriend knn class (including the one with another @ref memory_layout).
    template <memory_layout, typename, typename>
    friend class knn;


    /**
     * @brief Construct a new @ref knn object given @p k and sizes in @p data.
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in] data the @ref data object representing the used data set
     *
     * @pre @p k **must** be greater than `0`.
     */
    knn(const index_type k, Data& data) : k(k), data_(data), buffer(data.size * k) {
        DEBUG_ASSERT(0 < k, "Illegal number of nearest-neighbors to search for!: 0 < {}", k);
    }

    /// Reference to @ref data object.
    Data& data_;
};


/**
 * @brief Factory function for creating a new @ref knn object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[in] k the number of nearest-neighbors to search for
 * @param[in] data the used data object
 * @return the newly constructed @ref knn object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data, typename Options = typename Data::options_type>
[[nodiscard]] inline knn<layout, Options, Data> make_knn(const typename Options::index_type k, Data& data) {
    return knn<layout, Options, Data>(k, data);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
