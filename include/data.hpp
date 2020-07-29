/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-08
 *
 * @brief Implements the @ref data class representing the used data set.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <config.hpp>
#include <detail/assert.hpp>
#include <detail/timing.hpp>
#include <file_parser/file_parser.hpp>
#include <mpi_buffer.hpp>
#include <options.hpp>

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>


/**
 * @brief Class representing a data set.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options>
class data : detail::data_base {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;
    /// The type of the provided @ref options class.
    using options_type = Options;


    /// The number of data points in total.
    const index_type total_size;
    /// The number of data points per MPI rank.
    const index_type rank_size;
    /// The dimension of each data point.
    const index_type dims;
    /// The SYCL buffer holding all data: `buffer.get_count() == rank_size * dims`.
    sycl::buffer<real_type, 1> buffer;


    /**
     * @brief Returns the current data set with `new_layout`.
     * @details If `new_layout == layout` a compiler error is issued.
     * @tparam new_layout the layout of the data set
     * @return the data set with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] data<new_layout, Options> get_as() {
        static_assert(new_layout != layout, "using new_layout == layout result in a simple copy");

        data<new_layout, Options> new_data(opt_, rank_size, dims, total_size, comm_rank_);
        auto acc_this = buffer.template get_access<sycl::access::mode::read>();
        auto acc_new = new_data.buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type s = 0; s < rank_size; ++s) {
            for (index_type d = 0; d < dims; ++d) {
                // transform memory layout
                acc_new[new_data.get_linear_id(s, d)] = acc_this[this->get_linear_id(s, d)];
            }
        }
        return new_data;
    }

    /**
     * @brief Converts a two-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] comm_rank the current MPI rank
     * @param[in] point the provided data point
     * @param[in] size the total number of data points
     * @param[in] dim the provided dimension
     * @param[in] dims the total number of dimensions
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p size **must** be greater than `0`.
     * @pre @p point **must** be greater or equal than `0` and less than @p size.
     * @pre @p dims **must** be greater than `0`.
     * @pre @p dim **must** be greater or equal than `0` and less than @p dims.
     */
    [[nodiscard]] static constexpr index_type get_linear_id([[maybe_unused]] int comm_rank,
                                                            const index_type point, [[maybe_unused]] const index_type size,
                                                            const index_type dim, [[maybe_unused]] const index_type dims) noexcept
    {
        DEBUG_ASSERT_MPI(comm_rank, 0 < size, "Illegal total number of data points!: 0 < {}", size);
        DEBUG_ASSERT_MPI(comm_rank, 0 <= point && point < size, "Out-of-bounce access!: 0 <= {} < {}", point, size);
        DEBUG_ASSERT_MPI(comm_rank, 0 < dims, "Illegal total number of dimensions!: 0 < {}", dims);
        DEBUG_ASSERT_MPI(comm_rank, 0 <= dim && dim < dims, "Out-of-bounce access!: 0 <= {} < {}", dim, dims);

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return dim + point * dims;
        } else {
            // Struct of Arrays
            return point + dim * size;
        }
    }

    /**
     * @brief Returns the @ref options object which has been used to create this @ref data object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    /**
     * @brief Returns the specified @ref memory_layout (*Array of Structs* or *Struct of Arrays*).
     * @return the specified @ref memory_layout (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept { return layout; }


private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_>
    friend auto make_data(const Options_&, const std::string&, const MPI_Comm&);
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_>
    friend auto make_data(const Options_&, typename Options_::index_type, typename Options_::index_type, const MPI_Comm&);
    /// Befriend data class (including the one with another @ref memory_layout).
    template <memory_layout, typename>
    friend class data;


    /**
     * @brief Construct a new data object from the given @p buffers.
     * @param[in] opt the provided @ref options object
     * @param[in] buffers the @ref mpi_buffers containing the data points
     * @param[in] total_size the total number of data points
     * @param[in] comm_rank the current MPI rank
     *
     * @pre The number of total data points **must** be greater than `0`.
     * @pre The number of data points on this rank **must** be greater than `0`.
     * @pre The dimension of the data points **must** be greater than `0`.
     */
    data(const Options& opt, mpi_buffers<real_type, index_type>& buffers, const index_type total_size, const int comm_rank)
        : total_size(total_size), rank_size(buffers.rank_size), dims(buffers.dims),
          buffer(buffers.active().size()), comm_rank_(comm_rank), opt_(opt)
    {
        DEBUG_ASSERT_MPI(comm_rank_, 0 < total_size, "Illegal total_size!: {}", total_size);
        DEBUG_ASSERT_MPI(comm_rank_, 0 < rank_size, "Illegal rank_size!: {}", rank_size);
        DEBUG_ASSERT_MPI(comm_rank_, 0 < dims, "Illegal number of dimensions!: {}", dims);

        std::vector<real_type>& active_buffer = buffers.active();
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (std::size_t i = 0; i < active_buffer.size(); ++i) {
            acc[i] = active_buffer[i];
        }
    }

    /**
     * @brief Constructs a new data object with an empty buffer.
     * @param[in] opt the provided @ref options object
     * @param[in] rank_size the total number of data points on this rank
     * @param[in] dims the total number of dimensions
     * @param[in] total_size the total number of data points
     * @param[in] comm_rank the current MPI rank
     *
     * @pre The number of total data points **must** be greater than `0`.
     * @pre The number of data points on this rank **must** be greater than `0`.
     * @pre The dimension of the data points **must** be greater than `0`.
     */
    data(const Options& opt, const index_type rank_size, const index_type dims, const index_type total_size, const int comm_rank)
        : total_size(total_size), rank_size(rank_size), dims(dims),
          buffer(rank_size * dims), comm_rank_(comm_rank), opt_(opt)
    {
        DEBUG_ASSERT_MPI(comm_rank_, 0 < total_size, "Illegal total_size!: {}", total_size);
        DEBUG_ASSERT_MPI(comm_rank_, 0 < rank_size, "Illegal rank_size!: {}", rank_size);
        DEBUG_ASSERT_MPI(comm_rank_, 0 < dims, "Illegal number of dimensions!: {}", dims);
    }

    /**
     * @brief Print the size and dims of the data set @p data.
     * @param[inout] out the output stream to print @p opt
     * @param[in] data the data set
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& out, const data& data) {
        out << "total_size " << data.total_size << '\n';
        out << "rank_size " << data.rank_size << '\n';
        out << "dims " << data.dims;
        return out;
    }

    /// The current MPI rank.
    const int comm_rank_;
    /// Const reference to @ref options object.
    const Options& opt_;
};


/**
 * @brief Factory function for creating a new @ref data and @ref mpi_buffers object from a @p size and @p dims.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @param[in] opt the used @ref options object
 * @param[in] size the number of data points
 * @param[in] dims the number of dimensions per data point
 * @param[in] communicator the *MPI_Comm* communicator used to read the data in a distributed manner
 * @return the newly constructed @ref data and @ref mpi_buffers object (`[[nodiscard]]`)
 *
 * @note **Each** MPI rank contains `size * dims` points!
 */
template <memory_layout layout, typename Options>
[[nodiscard]] inline auto make_data(const Options& opt, const typename Options::index_type size, const typename Options::index_type dims, const MPI_Comm& communicator) {
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;
    using data_type = data<layout, Options>;
    using mpi_buffers_type = mpi_buffers<real_type, index_type>;

    START_TIMING(creating_data);
    int comm_size, comm_rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &comm_rank);

    mpi_buffers_type buffers(size, dims, communicator);
    // set dummy data based on the memory_layout
    const index_type total_size = comm_size * buffers.rank_size * buffers.dims;
    real_type val = comm_rank * buffers.rank_size * buffers.dims;
    for (index_type point = 0; point < size; ++point) {
        for (index_type dim = 0; dim < dims; ++dim) {
            buffers.active()[data_type::get_linear_id(comm_rank, point, size, dim, dims)] = val++;
        }
    }
    END_TIMING_MPI(creating_data, comm_rank);

    data_type dat(opt, buffers, total_size, comm_rank);
    return std::make_pair<data_type, mpi_buffers_type>(std::move(dat), std::move(buffers));
}

/**
 * @brief Factory function for creating a new @ref data and @ref mpi_buffers object from a @p file.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @param[in] opt the used @ref options object
 * @param[in] file the @p file from which the data should get loaded
 * @param[in] communicator the *MPI_Comm* communicator used to read the data in a distributed manner
 * @return the newly constructed @ref data and @ref mpi_buffers object (`[[nodiscard]]`)
 *
 * @throw std::invalid_argument if @p file doesn't exist.
 */
template <memory_layout layout, typename Options>
[[nodiscard]] inline auto make_data(const Options& opt, const std::string& file, const MPI_Comm& communicator) {
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;
    using data_type = data<layout, Options>;
    using mpi_buffers_type = mpi_buffers<real_type, index_type>;

    START_TIMING(parsing_data_file);
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);

    auto fp = make_file_parser<Options>(file, communicator);
    const index_type total_size = fp->parse_total_size();
    mpi_buffers_type buffers(fp->parse_rank_size(), fp->parse_dims(), communicator);
    fp->parse_content(buffers.active().data());

    // assumed that data is always read in Array of Structs format
    // -> convert to Struct of Arrays if requested
    if constexpr (layout == memory_layout::soa) {
        using active_data_type = data<memory_layout::soa, Options>;
        using inactive_data_type = data<memory_layout::aos, Options>;

        const std::vector<real_type>& active = buffers.active();
        std::vector<real_type>& inactive = buffers.inactive();
        for (index_type point = 0; point < buffers.rank_size; ++point) {
            for (index_type dim = 0; dim < buffers.dims; ++dim) {
                inactive[inactive_data_type::get_linear_id(comm_rank, point, buffers.rank_size, dim, buffers.dims)] =
                        active[active_data_type::get_linear_id(comm_rank, point, buffers.rank_size, dim, buffers.dims)];
            }
        }
        // IMPORTANT: swap active buffers (now buffer_1_ is active and buffer_0_ is inactive)
        buffers.swap_buffers();
    }
    END_TIMING_MPI(parsing_data_file, comm_rank);

    data_type dat(opt, buffers, total_size, comm_rank);
    return std::make_pair<data_type, mpi_buffers_type>(std::move(dat), std::move(buffers));
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP