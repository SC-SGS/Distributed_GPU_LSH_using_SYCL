/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-17
 *
 * @brief File parser for parsing plain data files.
 */


#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP


#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>

#include <mpi.h>

#include <config.hpp>
#include <detail/convert.hpp>
#include <detail/print.hpp>
#include <file_parser/base_file_parser.hpp>
#include <detail/mpi_type.hpp>
#include <mpi_buffer.hpp>


/**
 * @brief Default file parser for parsing plain data files in a distributed fashion using MPI IO.
 * @details Expected file format:
 * @code
 * 5
 * 2
 * 0.2,0.4
 * 1.3,1.6
 * 2.4,2.8
 * 3.6,4.2
 * 3.8,4.6
 * @endcode
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 *
 * @note The file **must** be saved in **binary** form.
 */
template <memory_layout layout, typename Options>
class default_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    /// The type of the base @ref file_parser.
    using base = file_parser<layout, Options>;
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref default_parser object for parsing plain data files.
     * @param[in] file the file to parse
     * @param[in] communicator the *MPI_Comm* communicator used to open the @p file with
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
    default_parser(const std::string& file, const MPI_Comm& communicator, const int comm_rank_)
        : file_parser<layout, Options>(file, communicator, comm_rank_)
    {
        detail::mpi_print<print_rank>(communicator, "Parsing a file using the default_parser together with MPI IO!\n");
    }

    [[nodiscard]] index_type parse_size() const override {
        // read first line containing the size
        int size;
        MPI_File_read_at(base::file_, 0, &size, 1, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
        return static_cast<index_type>(size);
    }
    [[nodiscard]] index_type parse_rank_size() const override {
        // parse the size per rank
        // note: total_size = 14, comm_size = 4 -> rank_size = 4 for ALL comm_ranks
        const index_type total_size = this->parse_size();
        int comm_size;
        MPI_Comm_size(base::comm_, &comm_size);
        if (total_size % comm_size == 0) {
            return total_size / comm_size;
        } else {
            return total_size / comm_size + 1;
        }
    }
    [[nodiscard]] index_type parse_dims() const override {
        // read second line containing the dims
        int dims;
        MPI_File_read_at(base::file_, sizeof(index_type), &dims, 1, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
        return static_cast<index_type>(dims);
    }
    [[nodiscard]] mpi_buffers<real_type, index_type> parse_content() const override {
        // calculate communicator size and rank
        int comm_size, comm_rank;
        MPI_Comm_size(base::comm_, &comm_size);
        MPI_Comm_rank(base::comm_, &comm_rank);

        // calculate total_size, the ceiled rank_size and dims AND perform sanity checks
        const index_type total_size = this->parse_size();
        const index_type ceil_rank_size = this->parse_rank_size();
        const index_type dims = this->parse_dims();
        DEBUG_ASSERT_MPI(base::comm_rank_, 0 < total_size, "Illegal total size!: {}", total_size);
        DEBUG_ASSERT_MPI(base::comm_rank_, 0 < ceil_rank_size, "Illegal (ceiled) rank size!: {}", ceil_rank_size);
        DEBUG_ASSERT_MPI(base::comm_rank_, 0 < dims, "Illegal number of dimensions!: {}", dims);

        // create buffers
        mpi_buffers<real_type, index_type> buffer(base::comm_, ceil_rank_size, dims);

        // check for correct real_type
        MPI_Offset file_size;
        MPI_File_get_size(base::file_, &file_size);
        file_size -= 2 * sizeof(index_type);
        assert((file_size == total_size * dims * sizeof(real_type)));

        // calculate byte offsets
        MPI_Offset initial_offset = 2 * sizeof(index_type);
        MPI_Offset rank_size = total_size / comm_size;
        if (static_cast<index_type>(comm_rank) < total_size % comm_size) ++rank_size;
        MPI_Offset rank_offset = (total_size / comm_size * comm_rank + std::min<MPI_Offset>(comm_rank, total_size % comm_size)) * dims * sizeof(real_type);

        std::vector<real_type>& internal_buffer = layout == memory_layout::aos ? buffer.active() : buffer.inactive();
        // read data elements, ALWAYS in Array of Structs format
        MPI_File_read_at(base::file_, initial_offset + rank_offset, internal_buffer.data(), rank_size * dims,
                detail::mpi_type_cast<real_type>(), MPI_STATUS_IGNORE);

        // fill missing data point with copy of the first
        if (static_cast<index_type>(comm_rank) >= total_size % comm_size) {
            for (index_type dim = 0; dim < dims; ++dim) {
                internal_buffer[rank_size * dims + dim] = internal_buffer[dim];
            }
        }

        // convert to Struct of Arrays format if necessary
        if constexpr (layout == memory_layout::soa) {
            std::vector<real_type>& active_internal_buffer = buffer.active();
            for (index_type point = 0; point < ceil_rank_size; ++point) {
                for (index_type dim = 0; dim < dims; ++dim) {
                    active_internal_buffer[base::get_linear_id(point, dim, ceil_rank_size, dims)] = internal_buffer[point * dims + dim];
                }
            }
        }

        return buffer;
    }

};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP
