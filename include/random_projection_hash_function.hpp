/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-29
 *
 * @brief Implements the @ref hash_functions class representing the used LSH hash functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTION_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTION_HASH_FUNCTION_HPP

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


/**
 * @brief Class representing the LSH hash functions.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class hash_functions : detail::hash_functions_base {
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


    /// The SYCL buffer holding all hash functions: `buffer.get_count() == options::num_hash_tables * options::num_hash_functions * (data::dims + 1)`.
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
            real_type hash = acc_hash_functions[get_linear_id(comm_rank, hash_table, hash_function, data.dims, opt, data)];
            for (index_type dim = 0; dim < data.dims; ++dim) {
                hash += acc_data[data_type::get_linear_id(comm_rank, point, data.rank_size, dim, data.dims)] *
                        acc_hash_functions[get_linear_id(comm_rank, hash_table, hash_function, dim, opt, data)];
            }
            combined_hash ^= static_cast<hash_value_type>(hash / opt.w)
                             + static_cast<hash_value_type>(0x9e3779b9)
                             + (combined_hash << static_cast<hash_value_type>(6))
                             + (combined_hash >> static_cast<hash_value_type>(2));
        }
        // TODO 2020-07-08 14:34 marcel: sycl::abs?
        if constexpr (std::is_signed_v<hash_value_type>) {
            combined_hash = combined_hash < 0 ? -combined_hash : combined_hash;
        }
        return combined_hash % opt.hash_table_size;
    }

    /**
     * @brief Returns the current hash functions with `new_layout`.
     * @details If `new_layout == layout` a compiler error is issued.
     * @tparam new_layout the layout of the hash functions
     * @return the hash functions with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] hash_functions<new_layout, Options, Data> get_as() {
        static_assert(new_layout != layout, "using new_layout == layout result in a simple copy");

        hash_functions<new_layout, Options, Data> new_hash_functions(opt_, data_, buffer.get_count(), comm_rank_);
        auto acc_this = buffer.template get_access<sycl::access::mode::read>();
        auto acc_new = new_hash_functions.buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
                for (index_type dim = 0; dim <= data_.dims; ++dim) {
                    // transform memory layout
                    acc_new[new_hash_functions.get_linear_id(hash_table, hash_function, dim)]
                        = acc_this[this->get_linear_id(hash_table, hash_function, dim)];
                }
            }
        }
        return new_hash_functions;
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
        DEBUG_ASSERT_MPI(comm_rank, 0 <= hash_table && hash_table < opt.num_hash_tables,
                         "Out-of-bounce access!: 0 <= {} < {}", hash_table, opt.num_hash_tables);
        DEBUG_ASSERT_MPI(comm_rank, 0 <= hash_function && hash_function < opt.num_hash_functions,
                         "Out-of-bounce access!: 0 <= {} < {}", hash_function, opt.num_hash_functions);
        DEBUG_ASSERT_MPI(comm_rank, 0 <= dim && dim < data.dims + 1,
                         "Out-of-bounce access!: 0 <= {} < {}", dim, data.dims + 1);

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return hash_table * opt.num_hash_functions * (data.dims + 1) + hash_function * (data.dims + 1) + dim;
        } else {
            // Struct of Arrays
            return hash_table * opt.num_hash_functions * (data.dims + 1) + dim * opt.num_hash_functions + hash_function;
        }
    }

    /**
     * @brief Returns the @ref options object which has been used to create this @ref hash_functions object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    /**
     * @brief Returns the @ref data object which has been used to create this @ref hash_functions object.
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
    friend auto make_hash_functions(Data_&, const MPI_Comm&);
    /// Befriend hash_functions class (including the one with another @ref memory_layout).
    template <memory_layout, typename, typename>
    friend class hash_functions;


    /**
     * @brief Construct new hash functions.
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] tmp_buffer the hash functions to initialize the sycl::buffer with
     * @param[in] comm_rank the current MPI rank
     */
    hash_functions(const Options& opt, Data& data, std::vector<real_type>& tmp_buffer, const int comm_rank)
        : buffer(tmp_buffer.size()), comm_rank_(comm_rank), opt_(opt), data_(data)
    {
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (std::size_t i = 0; i < tmp_buffer.size(); ++i) {
            acc[i] = tmp_buffer[i];
        }
    }

    /**
     * @brief Construct an empty hash functions buffer.
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] size the size of the empty buffer
     * @param[in] comm_rank the current MPI rank
     */
    hash_functions(const Options& opt, Data& data, const index_type size, const int comm_rank)
        : buffer(size), comm_rank_(comm_rank), opt_(opt), data_(data) { }

    /// The current MPI rank.
    const int comm_rank_;
    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;

};


/**
 * @brief Factory function for creating a new @ref hash_functions object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[in] data the used data object
 * @param[in] communicator the *MPI_Comm* communicator used to distribute the hash functions created on MPI rank 0
 * @return the newly constructed @ref hash_functions object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline auto make_hash_functions(Data& data, const MPI_Comm& communicator) {
    using options_type = typename Data::options_type;
    using real_type = typename options_type::real_type;
    using index_type = typename options_type::index_type;
    using hash_functions_type = hash_functions<layout, options_type, Data>;

    START_TIMING(creating_hash_functions);
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);

    options_type opt = data.get_options();
    std::vector<real_type> buffer(opt.num_hash_tables * opt.num_hash_functions * (data.dims + 1));

    if (comm_rank == 0) {
        // create hash functions on MPI rank 0
        // TODO 2020-05-07 19:03 marcel: uncomment for truly random numbers
//        std::random_device rnd_device;
//        std::mt19937 rnd_normal_gen(rnd_device());
//        std::mt19937 rnd_uniform_gen(rnd_device());
        std::mt19937 rnd_normal_gen;
        std::mt19937 rnd_uniform_gen;
        std::normal_distribution<real_type> rnd_normal_dist;
        std::uniform_real_distribution<real_type> rnd_uniform_dist(0, opt.w);

        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                for (index_type dim = 0; dim < data.dims; ++dim) {
                    buffer[hash_functions_type::get_linear_id(comm_rank, hash_table, hash_function, dim, opt, data)]
                        = std::abs(rnd_normal_dist(rnd_normal_gen));
                }
                buffer[hash_functions_type::get_linear_id(comm_rank, hash_table, hash_function, data.dims, opt, data)]
                    = rnd_uniform_dist(rnd_uniform_gen);
            }
        }
    }

    // broadcast hash functions to other MPI ranks
    MPI_Bcast(buffer.data(), buffer.size(), detail::mpi_type_cast<real_type>(), 0, communicator);
    END_TIMING_MPI(creating_hash_functions, comm_rank);

    return hash_functions_type(data.get_options(), data, buffer, comm_rank);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTION_HASH_FUNCTION_HPP
