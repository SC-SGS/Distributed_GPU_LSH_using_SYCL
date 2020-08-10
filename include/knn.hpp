/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-10
 *
 * @brief Implements the @ref knn class representing the result of the k-nearest-neighbor search.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP

#include <filesystem>
#include <vector>

#include <mpi.h>

#include <config.hpp>
#include <data.hpp>
#include <mpi_buffer.hpp>
#include <options.hpp>


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
    /// The type of the provided @ref options class.
    using options_type = Options;
    /// The type of the provided @ref data class.
    using data_type = Data;


    /// The buffers containing the found knns.
    mpi_buffers<index_type, index_type> buffers_knn;
    /// The buffers containing the distances to the found knns.
    mpi_buffers<real_type, index_type> buffers_dist;
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
        DEBUG_ASSERT_MPI(comm_rank_, 0 <= point && point < data_.rank_size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.rank_size);

        std::vector<index_type> res(k);
        std::vector<index_type>& buffer = buffers_knn.active();
        for (index_type i = 0; i < k; ++i) {
            res[i] = buffer[this->get_linear_id(comm_rank_, point, i, data_, k)];
        }
        return res;
    }
    std::vector<real_type> get_knn_dist(const index_type point) {
        DEBUG_ASSERT_MPI(comm_rank_, 0 <= point && point < data_.rank_size, "Out-of-bounce access!: 0 <= {} < {}", point, data_.rank_size);

        std::vector<real_type> res(k);
        std::vector<real_type>& buffer = buffers_dist.active();
        for (index_type i = 0; i < k; ++i) {
            res[i] = buffer[this->get_linear_id(comm_rank_, point, i, data_, k)];
        }
        return res;
    }

    /**
     * @brief Returns the current knns with `new_layout`.
     * @details If `new_layout == layout` a compiler error is issued.
     * @tparam new_layout the layout of the knns
     * @return the knns with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] knn<new_layout, Options, Data> get_as() {
        static_assert(new_layout != layout, "using new_layout == layout result in a simple copy");

        knn<new_layout, Options, Data> new_knn(k, data_, comm_, comm_rank_);
        std::vector<index_type>& buffer_knn_this = buffers_knn.active();
        std::vector<index_type>& buffer_knn_new = new_knn.buffers_knn.active();
        std::vector<real_type>& buffer_dist_this = buffers_dist.active();
        std::vector<real_type>& buffer_dist_new = new_knn.buffers_dist.active();
        for (index_type point = 0; point < data_.rank_size; ++point) {
            for (index_type nn = 0; nn < k; ++nn) {
                // transform memory layout
                buffer_knn_new[new_knn.get_linear_id(comm_rank_, point, nn, data_, k)] =
                        buffer_knn_this[this->get_linear_id(comm_rank_, point, nn, data_, k)];
                buffer_dist_new[new_knn.get_linear_id(comm_rank_, point, nn, data_, k)] =
                        buffer_dist_this[this->get_linear_id(comm_rank_, point, nn, data_, k)];
            }
        }
        return new_knn;
    }

    /**
     * @brief Converts a two-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] comm_rank the current MPI rank
     * @param[in] point the provided data point
     * @param[in] i the provided knn index
     * @param[in] data the used data set
     * @param[in] k the number of nearest-neighbors to search for
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p point **must** be greater or equal than `0` and less than @p dims.
     * @pre @p i **must** be greater or equal than `0` and less than @p k.
     * @pre @p k **must** be greater than `0`
     */
    [[nodiscard]] static constexpr index_type get_linear_id([[maybe_unused]] const int comm_rank,
                                                            const index_type point, const index_type nn,
                                                            [[maybe_unused]] const Data& data, [[maybe_unused]] const index_type k) noexcept
    {
        DEBUG_ASSERT_MPI(comm_rank, 0 <= point && point < data.rank_size, "Out-of-bounce access!: 0 <= {} < {}", point, data.rank_size);
        DEBUG_ASSERT_MPI(comm_rank, 0 <= nn && nn < k, "Out-of-bounce access!: 0 <= {} < {}", nn, k);
        DEBUG_ASSERT_MPI(comm_rank, 0 < k, "Illegal number of k-nearest-neighbors to search for!: 0 < {}", k);

        if constexpr (layout == memory_layout::aos) {
            return point * k + nn;
        } else {
            return nn * data.rank_size + point;
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
    void save(const std::string& knn_file_name, const std::string& dist_file_name, const MPI_Comm& communicator) {
        START_TIMING(save_knns);
        if constexpr (layout == memory_layout::soa) {
            const auto transform = [&](auto& buffer) {
                using soa_layout = knn<memory_layout::soa, Options, Data>;
                using aos_layout = knn<memory_layout::aos, Options, Data>;

                auto& active = buffer.active();
                auto& inactive = buffer.inactive();
                for (index_type point = 0; point < data_.rank_size; ++point) {
                    for (index_type nn = 0; nn < k; ++nn) {
                        inactive[aos_layout::get_linear_id(comm_rank_, point, nn, data_, k)]
                                = active[soa_layout::get_linear_id(comm_rank_, point, nn, data_, k)];
                    }
                }
                buffer.swap_buffers();
            };

            transform(buffers_knn);
            transform(buffers_dist);
        }

        MPI_File file;

        const auto save_to_file = [&](const std::string& file_name, auto& buffer) {
            using value_type = typename std::remove_reference_t<decltype(buffer)>::value_type;
            
            // open file in create mode to write header information
            if (comm_rank_ == 0) {
                std::ofstream out(file_name, std::ios::out | std::ios::binary);
                out.write(reinterpret_cast<const char*>(&data_.total_size), sizeof(data_.total_size));
                out.write(reinterpret_cast<const char*>(&k), sizeof(k));
            }
            MPI_Barrier(communicator);

            // open file in append mode to write knns
            MPI_File_open(communicator, file_name.c_str(), MPI_MODE_APPEND | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
            MPI_File_write_ordered(file, buffer.active().data(), buffer.active().size(), detail::mpi_type_cast<value_type>(), MPI_STATUS_IGNORE);
            MPI_File_close(&file);
        };

        save_to_file(knn_file_name, buffers_knn);
        save_to_file(dist_file_name, buffers_dist);

        if constexpr (layout == memory_layout::soa) {
            buffers_knn.swap_buffers();
            buffers_dist.swap_buffers();
        }
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
        : buffers_knn(data.rank_size, k, communicator), buffers_dist(data.rank_size, k, communicator),
          k(k), comm_rank_(comm_rank), comm_(communicator), data_(data)
    {
        DEBUG_ASSERT_MPI(comm_rank, 0 < k, "Illegal number of nearest-neighbors to search for!: 0 < {}", k);

        for (index_type point = 0; point < data.rank_size; ++point) {
            for (index_type nn = 0; nn < k; ++nn) {
                buffers_knn.active()[this->get_linear_id(comm_rank, point, nn, data, k)] = point + comm_rank * data.rank_size;
                buffers_dist.active()[this->get_linear_id(comm_rank, point, nn, data, k)] = std::numeric_limits<real_type>::max();
            }
        }
    }

    /// The current MPI rank.
    const int comm_rank_;
    /// The used MPI communicator.
    const MPI_Comm& comm_;
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
