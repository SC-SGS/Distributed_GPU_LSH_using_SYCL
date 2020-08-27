/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-25
 *
 * @brief Implements the @ref entropy_based_hash_functions class representing the used entropy-based LSH hash functions
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ENTROPY_BASED_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ENTROPY_BASED_HASH_FUNCTION_HPP

#include <config.hpp>
#include <data.hpp>
#include <detail/mpi_type.hpp>
#include <detail/timing.hpp>
#include <options.hpp>

#include <mpi.h>

#include <cmath>
#include <random>
#include <type_traits>
#include <vector>


template <typename real_type>
void pairwise_exchange(std::vector<real_type>& a, const int sendrank, const int recvrank, const MPI_Comm& communicator) {
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);

    std::vector<real_type> remote(a.size());
    std::vector<real_type> all(2 * a.size());
    constexpr int merge_tag = 1;
    constexpr int sorted_tag = 2;

    if (comm_rank == sendrank) {
        MPI_Send(a.data(), a.size(), detail::mpi_type_cast<real_type>(), recvrank, merge_tag, communicator);
        MPI_Recv(a.data(), a.size(), detail::mpi_type_cast<real_type>(), recvrank, sorted_tag, communicator, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(remote.data(), remote.size(), detail::mpi_type_cast<real_type>(), sendrank, merge_tag, communicator, MPI_STATUS_IGNORE);
        std::copy(a.begin(), a.end(), all.begin());
        std::copy(remote.begin(), remote.end(), all.begin() + a.size());

        std::sort(all.begin(), all.end());

        std::size_t theirstart = sendrank > comm_rank ? a.size() : 0;
        std::size_t mystart = sendrank > comm_rank ? 0 : a.size();
        MPI_Send(all.data() + theirstart, a.size(), detail::mpi_type_cast<real_type>(), sendrank, sorted_tag, communicator);
        std::copy(all.begin() + mystart, all.begin() + mystart + a.size(), a.begin());
    }
}

// https://stackoverflow.com/questions/23633916/how-does-mpi-odd-even-sort-work
template <typename real_type>
void odd_even_sort(std::vector<real_type>& a, const MPI_Comm& communicator) {
    int comm_size, comm_rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &comm_rank);

    // sort local vector
    std::sort(a.begin(), a.end());

    // odd-even
    for (std::size_t i = 1; i <= static_cast<std::size_t>(comm_size); ++i) {
        if ((i + comm_rank) % 2 == 0) {
            if (comm_rank < comm_size - 1) {
                pairwise_exchange(a, comm_rank, comm_rank + 1, communicator);
            }
        } else if (comm_rank > 0) {
            pairwise_exchange(a, comm_rank - 1, comm_rank, communicator);
        }
    }
}


/**
 * @brief Class representing the entropy-based LSH hash functions.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class entropy_based_hash_functions : detail::hash_functions_base {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
    static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must by a 'data' type!");
public:
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;
    /// The used type of a hash value.
    using hash_value_type = typename Options::hash_value_type;
    /// The type of the provided @ref data class.
    using data_type = Data;

    
    /// The SYCL buffer holding all used hash functions: `buffer.get_count() == options::num_hash_tables * options::num_hash_functions * (data::dims + 1)`.
    sycl::buffer<real_type, 1> buffer;


    /**
     * @brief Calculates the hash value of the data point @p point in hash table @p hash_table.
     * @tparam AccData the type of data set accessor
     * @tparam AccHashFunctions the type of the hash functions accessor
     * @param[in] comm_rank the current MPI rank
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] acc_data the data set accessor
     * @param[in] acc_hash_functions the hash functions accessor
     * @param[in] opt the used options
     * @param[in] data the used data set
     * @return the hash value (`[[nodiscard]]`)
     *
     * @pre @p hash_table **must** be greater or equal than `0` and less than @p num_hash_tables.
     * @pre @p point **must** be greater or equal than `0` and less than @p size.
     */
    template <typename AccData, typename AccHashFunctions>
    [[nodiscard]] static constexpr hash_value_type hash([[maybe_unused]] const int comm_rank,
                                                        const index_type hash_table, const index_type point,
                                                        AccData& acc_data, AccHashFunctions& acc_hash_functions,
                                                        const Options& opt, const Data& data)
    {
        DEBUG_ASSERT_MPI(comm_rank, 0 <= hash_table && hash_table < opt.num_hash_tables,
                         "Out-of-bounce access!: 0 <= {} < {}", hash_table, opt.num_hash_tables);
        DEBUG_ASSERT_MPI(comm_rank, 0 <= point && point < data.rank_size,
                         "Out-of-bounce access!: 0 <= {} < {}", point, data.rank_size);

        hash_value_type combined_hash = opt.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            real_type hash = 0.0;
            const index_type idx = hash_table * opt.num_hash_functions * (data.dims + opt.num_cut_off_points) + hash_function * (data.dims + opt.num_cut_off_points);
            for (index_type dim = 0; dim < data.dims; ++dim) {
                hash += acc_data[data.get_linear_id(comm_rank, point, data.rank_size, dim, data.dims)] *
                        acc_hash_functions[idx + dim];
            }
            hash_value_type i = 0;
            while (i < opt.num_cut_off_points && acc_hash_functions[idx + data.dims + i] < hash) { ++i; }
            combined_hash ^= static_cast<hash_value_type>(i)
                             + static_cast<hash_value_type>(0x9e3779b9)
                             + (combined_hash << static_cast<hash_value_type>(6))
                             + (combined_hash >> static_cast<hash_value_type>(2));
        }
        if constexpr (std::is_signed_v<hash_value_type>) {
            combined_hash = combined_hash < 0 ? -combined_hash : combined_hash;
        }
        return combined_hash %= opt.hash_table_size;
    }

    /**
     * @brief Returns the current hash functions with `new_layout`.
     * @details If `new_layout == layout` a compiler error is issued.
     * @tparam new_layout the layout of the hash functions
     * @return the hash functions with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] entropy_based_hash_functions<new_layout, Options, Data> get_as() {
//        static_assert(new_layout != layout, "using new_layout == layout result in a simple copy");
//
//        entropy_based_hash_functions<new_layout, Options, Data> new_hash_functions(opt_, data_, buffer.get_count(), comm_rank_);
//        auto acc_this = buffer.template get_access<sycl::access::mode::read>();
//        auto acc_new = new_hash_functions.buffer.template get_access<sycl::access::mode::discard_write>();
//        std::vector<real_type>& pool_new = new_hash_functions.hash_functions_pool;
//        for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
//            for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
//                for (index_type dim = 0; dim <= data_.dims; ++dim) {
//                    // transform memory layout
//                    const index_type idx_new = new_hash_functions.get_linear_id(hash_table, hash_function, dim);
//                    const index_type idx_this = this->get_linear_id(hash_table, hash_function, dim);
//                    acc_new[idx_new] = acc_this[idx_this];
//                    pool_new[idx_new] = acc_this[idx_this];
//                }
//            }
//        }
//        return new_hash_functions;
        return 0;
    }

    /**
     * @brief Converts a three-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] comm_rank the current MPI rank
     * @param[in] hash_table the provided hash table
     * @param[in] hash_function the provided hash function
     * @param[in] dim the provided dimension
     * @param[in] opt the used options
     * @param[in] data the used data set
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p hash_table **must** be greater or equal than `0` and less than @p num_hash_tables.
     * @pre @p hash_function **must** be greater or equal than `0` and less than @p num_hash_functions.
     * @pre @p dim **must** be greater or equal than `0` and less than @p dims + 1.
     */
    [[nodiscard]] static constexpr index_type get_linear_id([[maybe_unused]] const int comm_rank,
                                                            const index_type hash_table, const index_type hash_function, const index_type dim,
                                                            const Options& opt, const Data& data) noexcept
    {
//        DEBUG_ASSERT_MPI(comm_rank, 0 <= hash_table && hash_table < opt.num_hash_tables,
//                         "Out-of-bounce access!: 0 <= {} < {}", hash_table, opt.num_hash_tables);
//        DEBUG_ASSERT_MPI(comm_rank, 0 <= hash_function && hash_function < opt.num_hash_functions,
//                         "Out-of-bounce access!: 0 <= {} < {}", hash_function, opt.num_hash_functions);
//        DEBUG_ASSERT_MPI(comm_rank, 0 <= dim && dim < data.dims + 1,
//                         "Out-of-bounce access!: 0 <= {} < {}", dim, data.dims + 1);
//
//        if constexpr (layout == memory_layout::aos) {
//            // Array of Structs
//            return hash_table * opt.num_hash_functions * (data.dims + 1) + hash_function * (data.dims + 1) + dim;
//        } else {
//            // Struct of Arrays
//            return hash_table * opt.num_hash_functions * (data.dims + 1) + dim * opt.num_hash_functions + hash_function;
//        }
        return 0;
    }

    /**
     * @brief Returns the @ref options object which has been used to create this @ref entropy_based_hash_functions object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    /**
     * @brief Returns the @ref data object which has been used to create this @ref entropy_based_hash_functions object.
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
    friend auto make_entropy_based_hash_functions(Data_&, const MPI_Comm&);
    /// Befriend hash_functions class (including the one with another @ref memory_layout).
    template <memory_layout, typename, typename>
    friend class entropy_based_hash_functions;


    /**
     * @brief Construct new hash functions.
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] hash_functions_pool a pool of hash functions that can be used
     * @param[in] cut_off_points_pool the cut off points corresponding to the hash functions in the pool
     * @param[in] comm_rank the current MPI rank
     */
    entropy_based_hash_functions(const Options& opt, Data& data, std::vector<real_type>& hash_functions_pool, std::vector<real_type>& cut_off_points_pool, const int comm_rank)
        : buffer(opt.num_hash_tables * opt.num_hash_functions * (data.dims + opt.num_cut_off_points)), comm_rank_(comm_rank), opt_(opt), data_(data)
    {
        // TODO 2020-08-12 12:27 marcel: hash pool multiple ranks?
        std::mt19937 rnd_uniform_gen;
        std::uniform_int_distribution<index_type> rnd_uniform_dist(0, opt.hash_pool_size);

        index_type idx = 0;
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                const index_type pool_idx = rnd_uniform_dist(rnd_uniform_gen);
                
                for (index_type dim = 0; dim < data.dims; ++dim) {
                    acc[idx++] = hash_functions_pool[pool_idx * data.dims + dim];
                }
                for (index_type dim = 0; dim < opt.num_cut_off_points; ++dim) {
                    acc[idx++] = cut_off_points_pool[pool_idx * opt.num_cut_off_points + dim];
                }
            }
        }
    }


    /// The current MPI rank.
    const int comm_rank_;
    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;

};


/**
 * @brief Factory function for creating a new @ref entropy_based_hash_functions object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[in] data the used data object
 * @param[in] communicator the *MPI_Comm* communicator used to distribute the hash functions created on MPI rank 0
 * @return the newly constructed @ref entropy_based_hash_functions object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline auto make_entropy_based_hash_functions(Data& data, const MPI_Comm& communicator) {
    using options_type = typename Data::options_type;
    using real_type = typename options_type::real_type;
    using index_type = typename options_type::index_type;
    using hash_functions_type = entropy_based_hash_functions<layout, options_type, Data>;

    START_TIMING(creating_hash_functions);
    int comm_rank, comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    
    options_type opt = data.get_options();

    // create hash functions pool
    std::vector<real_type> hash_functions_pool(opt.hash_pool_size * data.dims);
    std::vector<real_type> cut_off_points_pool(opt.hash_pool_size * opt.num_cut_off_points);
    std::mt19937 rnd_normal_gen;
    std::normal_distribution<real_type> rnd_normal_dist;
    for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
        for (index_type dim = 0; dim < data.dims; ++dim) {
            hash_functions_pool[hash_function * data.dims + dim] = std::abs(rnd_normal_dist(rnd_normal_gen));
        }
    }
    
    // calculate cut-off points
    {
        sycl::queue queue(sycl::default_selector{});
        for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {

            std::vector<real_type> hash_values(data.rank_size, 0.0);

            {
                sycl::buffer<real_type, 1> hf_buffer(hash_functions_pool.data(), hash_functions_pool.size());
                sycl::buffer<real_type, 1> hv_buffer(hash_values.data(), hash_values.size());
                queue.submit([&](sycl::handler& cgh) {
                    auto acc_data = data.buffer.template get_access<sycl::access::mode::read>(cgh);
                    auto acc_hf = hf_buffer.template get_access<sycl::access::mode::read>(cgh);
                    auto acc_hv = hv_buffer.template get_access<sycl::access::mode::discard_write>(cgh);

                    cgh.parallel_for<class fill_unsorted>(sycl::range<>(data.rank_size), [=](sycl::item<> item) {
                        const index_type idx = item.get_linear_id();

                        real_type value = 0.0;
                        for (index_type dim = 0; dim < data.dims; ++dim) {
                            value += acc_data[data.get_linear_id(comm_rank, idx, data.rank_size, dim, data.dims)] * acc_hf[hash_function * data.dims + dim];
                        }
                        acc_hv[idx] = value;
                    });
                });
            }
            // distributed sort vector
            odd_even_sort(hash_values, communicator);

            // calculate cut-off points
            std::vector<real_type> cut_off_points(opt.num_cut_off_points, 0.0);

            // calculate cut-off points indices
            std::vector<index_type> cut_off_points_idx(opt.num_cut_off_points);
            const index_type jump = data.total_size / opt.num_cut_off_points;
            for (index_type i = 0; i < opt.num_cut_off_points - 1; ++i) {
                cut_off_points_idx[i] = (i + 1) * jump;
            }
            cut_off_points_idx.back() = data.total_size - 1;

            // fill cut-off points which are located on the current rank
            for (index_type i = 0; i < opt.num_cut_off_points; ++i) {
                // check if index belongs to current rank
                if (cut_off_points_idx[i] >= data.rank_size * comm_rank && cut_off_points_idx[i] < data.rank_size * (comm_rank + 1)) {
                    cut_off_points[i] = hash_values[cut_off_points_idx[i] % data.rank_size];
                }
            }

            // combine to final cut-off points on all ranks
            MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), cut_off_points.size(), detail::mpi_type_cast<real_type>(), MPI_SUM, communicator);
            
            std::copy(cut_off_points.begin(), cut_off_points.end(), cut_off_points_pool.begin() + hash_function * cut_off_points.size());

            // TODO 2020-08-25 14:04 marcel: cut of points
        }
    }


    END_TIMING_MPI(creating_hash_functions, comm_rank);
    return hash_functions_type(data.get_options(), data, hash_functions_pool, cut_off_points_pool, comm_rank);
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ENTROPY_BASED_HASH_FUNCTION_HPP
