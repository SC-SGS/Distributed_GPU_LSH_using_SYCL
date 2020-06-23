/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-19
 *
 * @brief Implements the @ref knn class representing the result of the k-nearest-neighbor search.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP

#include <vector>

#include <mpi.h>

#include <config.hpp>
#include <data.hpp>
#include <mpi_buffer.hpp>
#include <options.hpp>


namespace detail {
    /**
     * @brief Empty base class for the @ref knn class. Only for static_asserts.
     */
    class knn_base {};
}


/**
 * @brief Class representing the result of the k-nearest-neighbor search.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class knn : detail::knn_base {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
    static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must by a 'data' type!");
public:
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;


    /// The buffers containing the found knns.
    mpi_buffers<index_type, index_type> buffers;
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
        std::vector<index_type>& buffer = buffers.active();
        for (index_type i = 0; i < k; ++i) {
            res[i] = buffer[this->get_linear_id(point, i)];
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
        std::vector<index_type> buffer = buffers.active();
        auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>();
        for (index_type i = 0; i < k; ++i) {
            // get the knn index
            const index_type knn_id = buffer[this->get_linear_id(point, i)];
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
        std::vector<index_type> buffer_this = buffers.active();
        std::vector<index_type> buffer_new = new_knn.buffers.active();
        for (index_type s = 0; s < data_.size(); ++s) {
            for (index_type nn = 0; nn < k; ++nn) {
                // transform memory layout
                buffer_new[new_knn.get_linear_id(s, nn)] = buffer_this[this->get_linear_id(s, nn)];
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
     * @pre @p i **must** be greater or equal than `0` and less than `k`.
     */
    [[nodiscard]] constexpr index_type get_linear_id(const index_type point, const index_type i) const noexcept {
        return knn::get_linear_id(point, data_.size, i, k);
    }
    /**
     * @brief Converts a two-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] point the provided data point
     * @param[in] size the total number of data points
     * @param[in] i the provided knn index
     * @param[in] k the number of nearest-neighbors to search for
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p size **must** be greater than `0`
     * @pre @p point **must** be greater or equal than `0` and less than @p dims.
     * @pre @p k **must** be greater than `0`
     * @pre @p i **must** be greater or equal than `0` and less than @p k.
     */
    [[nodiscard]] static constexpr index_type get_linear_id(const index_type point, [[maybe_unused]] const index_type size,
                                                            const index_type i, [[maybe_unused]] const index_type k) noexcept
    {
        DEBUG_ASSERT(0 < size, "Illegal total number of data points!: 0 < {}", size);
        DEBUG_ASSERT(0 <= point && point < size, "Out-of-bounce access!: 0 <= {} < {}", point, size);
        DEBUG_ASSERT(0 < k, "Illegal number of k-nearest-neighbors to search for!: 0 < {}", k);
        DEBUG_ASSERT(0 <= i && i < k, "Out-of-bounce access!: 0 <= {} < {}", i, k);

        if constexpr (layout == memory_layout::aos) {
            return point * k + i;
        } else {
            return i * size + point;
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
     * @param[in] file_name the name of the @p file
     * @param[in] communicator the MPI communicator used to write the data to the file
     *
     * @throw std::invalid_argument if @p file can't be opened or created.
     */
    void save(const std::string& file_name, const MPI_Comm& communicator) {
        START_TIMING(save_knns);
        MPI_File file;

        MPI_File_open(communicator, file_name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &file);
        MPI_File_write_ordered(file, buffers.active(), buffers.size * buffers.dims, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);

//        MPI_File_close(&file);
        END_TIMING_MPI(save_knns, comm_rank_);
    }

private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Data_, typename Options_>
    friend auto make_knn(typename Options_::index_type, Data_&, const MPI_Comm&);
    /// Befriend knn class (including the one with another @ref memory_layout).
    template <memory_layout, typename, typename>
    friend class knn;


    /**
     * @brief Construct a new @ref knn object given @p k and sizes in @p data.
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in] data the @ref data object representing the used data set
     * @param[in] communicator the *MPI_Comm* communicator used
     * @param[in] comm_rank the current MPI rank
     *
     * @pre @p k **must** be greater than `0`.
     */
    knn(const index_type k, Data& data, const MPI_Comm& communicator, const int comm_rank)
        : buffers(communicator, data.size, k), k(k), comm_rank_(comm_rank), data_(data)
    {
        DEBUG_ASSERT(0 < k, "Illegal number of nearest-neighbors to search for!: 0 < {}", k);
    }

    /// The current MPI rank.
    const int comm_rank_;
    /// Reference to @ref data object.
    Data& data_;
};


/**
 * @brief Factory function for creating a new @ref knn object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[in] k the number of nearest-neighbors to search for
 * @param[in] data the used data object
 * @param[in] communicator the *MPI_Comm* communicator used
 * @return the newly constructed @ref knn object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data, typename Options = typename Data::options_type>
[[nodiscard]] inline auto make_knn(const typename Options::index_type k, Data& data, const MPI_Comm& communicator) {
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);
    return knn<layout, Options, Data>(k, data, communicator, comm_rank);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
