/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-21
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
#include <vector>

#include <mpi.h>

#include <config.hpp>
#include <detail/mpi_type.hpp>
#include <detail/print.hpp>
#include <file_parser/base_parser.hpp>
#include <options.hpp>


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
template <typename Options, typename type = typename Options::real_type>
class default_parser final : public file_parser<Options, type> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    /// The type of the base @ref file_parser.
    using base = file_parser<Options, type>;
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref default_parser object for parsing plain data files.
     * @param[in] file_name the file to parse
     * @param[in] communicator the *MPI_Comm* communicator used to open the @p file with
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
    default_parser(const std::string& file_name, const MPI_Comm& communicator) : file_parser<Options, type>(file_name, communicator) {
        detail::mpi_print(base::comm_rank_, "Parsing a file ('{}') using the default_parser together with MPI IO!\n", file_name.c_str());
    }

    [[nodiscard]] index_type parse_total_size() const override {
        // read first line containing the total_size
        int total_size;
        MPI_File_read_at(base::file_, 0, &total_size, 1, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
        return static_cast<index_type>(total_size);
    }
    [[nodiscard]] index_type parse_rank_size() const override {
        // parse the size per rank
        // note: total_size = 14, comm_size = 4 -> rank_size = 4 for ALL comm_ranks
        const index_type total_size = this->parse_total_size();
        if (total_size % base::comm_size_ == 0) {
            return total_size / base::comm_size_;
        } else {
            return total_size / base::comm_size_ + 1;
        }
    }
    [[nodiscard]] index_type parse_dims() const override {
        // read second line containing the dims
        int dims;
        MPI_File_read_at(base::file_, sizeof(index_type), &dims, 1, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
        return static_cast<index_type>(dims);
    }
    void parse_content(type* buffer) const override {
        // calculate total_size, rank_size and dims AND perform sanity checks
        const index_type total_size = this->parse_total_size();
        const bool has_smaller_rank_size = (total_size % base::comm_size_ != 0) && static_cast<index_type>(base::comm_rank_) >= (total_size % base::comm_size_);
        const index_type rank_size = this->parse_rank_size() - static_cast<index_type>(has_smaller_rank_size);
        const index_type dims = this->parse_dims();
        DEBUG_ASSERT_MPI(base::comm_rank_, 0 < total_size, "Illegal total size!: {}", total_size);
        DEBUG_ASSERT_MPI(base::comm_rank_, 0 < rank_size, "Illegal rank size!: {}", rank_size);
        DEBUG_ASSERT_MPI(base::comm_rank_, 0 < dims, "Illegal number of dimensions!: {}", dims);

        // check for correct type
        MPI_Offset file_size;
        MPI_File_get_size(base::file_, &file_size);
        file_size -= 2 * sizeof(index_type);
        assert((static_cast<std::size_t>(file_size) == total_size * dims * sizeof(type)));

        // calculate byte offsets
        constexpr index_type initial_offset = 2 * sizeof(index_type);     // for size and dims
        MPI_Offset rank_offset = ( total_size / base::comm_size_ * base::comm_rank_
                + std::min<MPI_Offset>(base::comm_rank_, total_size % base::comm_size_) ) * dims * sizeof(type);

        // read data
        MPI_Status status;
        MPI_File_read_at(base::file_, initial_offset + rank_offset, buffer, rank_size * dims,
                detail::mpi_type_cast<type>(), &status);

        // check whether the correct number of values were read
        int read_count;
        MPI_Get_count(&status, detail::mpi_type_cast<type>(), &read_count);
        assert((static_cast<index_type>(read_count) == rank_size * dims));

        // fill missing data point with copy of the first
        if (has_smaller_rank_size) {
            for (index_type dim = 0; dim < dims; ++dim) {
                buffer[rank_size * dims + dim] = buffer[(rank_size - 1) * dims + dim];
            }
        }
    }

};

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP
