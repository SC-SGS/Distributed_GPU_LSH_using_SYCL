/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-26
 *
 * @brief File parser for parsing plain binary data files.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BINARY_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BINARY_PARSER_HPP

#include <sycl_lsh/detail/assert.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/file_parser/base_parser.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/timer.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <fmt/format.h>
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <string_view>

namespace sycl_lsh::mpi {

    /**
     * @brief File parser class for a custom **binary** data format.
     * @details Expected file format: header information with the total number of data points and the number of dimensions followed by
     *          the data in *Array of Structs* format.
     *
     * Example (in text format, correct files **must** be in binary form):
     * 4
     * 2
     * 0.0 0.1
     * 0.2 0.3
     * 0.4 0.5
     * 0.6 0.7
     * @tparam Options  type of the used @ref sycl_lsh::options class
     * @tparam T the type of the data to parse
     */
    template <typename Options, typename T>
    class binary_parser final : public file_parser<Options, T> {
        using base_type = file_parser<Options, T>;
    public:
        /// The index type as specified in the provided @ref sycl_lsh::options template type.
        using index_type = typename base_type::index_type;
        /// The type of the data which should get parsed.
        using parsing_type = typename base_type::parsing_type;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::mpi::binary_parser object responsible for parsing the custom binary file format.
         * @param[in] file_name the file to parse
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         */
        binary_parser(std::string_view file_name, const communicator& comm, const logger& logger);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                  parsing                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Parse the **total** number of data points in the file.
         * @details Reads the total size from the first line of the file. The type must be of @ref options::index_type.
         * @return the total number of data points (`[[nodiscard]]`)
         */
        [[nodiscard]]
        index_type parse_total_size() const override;
        /**
         * @brief Parse the number of dimensions of each data point in the file.
         * @details Reads the number of dimensions from the second line of the file. The type must be of @ref options::index_type. \n
         *          Assumes that each data points has the same number of dimensions.
         * @return the number of dimensions (`[[nodiscard]]`)
         */
        [[nodiscard]]
        index_type parse_dims() const override;
        /**
         * @brief Parse the content of the file.
         * @details Fills the last elements of the buffer on the last MPI rank such that all MPI ranks have te same number of data points. \n
         *          Calls *MPI_Abort()* if the file size doesn't match the header information or the values actual read diverge from the
         *          number of values which should theoretically be used
         * @param[out] buffer to write the data to
         */
        void parse_content(parsing_type* buffer) const override;
    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename Options, typename T>
    binary_parser<Options, T>::binary_parser(const std::string_view file_name, const communicator& comm, const logger& logger)
        : file_parser<Options, T>(file_name, comm, logger)
    {
        logger.log("Parsing the data file '{}' using the binary_parser together with MPI IO.\n", file_name);
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  parsing                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename Options, typename T>
    [[nodiscard]]
    typename binary_parser<Options, T>::index_type binary_parser<Options, T>::parse_total_size() const {
        // read first line containing the total_size
        index_type total_size;
        MPI_File_read_at(base_type::file_.get(), 0, &total_size, 1, type_cast<index_type>(), MPI_STATUS_IGNORE);
        return total_size;
    }

    template <typename Options, typename T>
    [[nodiscard]]
    typename binary_parser<Options, T>::index_type binary_parser<Options, T>::parse_dims() const {
        index_type dims;
        MPI_File_read_at(base_type::file_.get(), sizeof(index_type), &dims, 1, type_cast<index_type>(), MPI_STATUS_IGNORE);
        return dims;
    }

    template <typename Options, typename T>
    void binary_parser<Options, T>::parse_content(parsing_type* buffer) const {
        timer t(base_type::comm_);

        const index_type total_size = this->parse_total_size();
        const index_type rank_size = this->parse_rank_size();
        const index_type dims = this->parse_dims();
        const int comm_size = base_type::comm_.size();
        const int comm_rank = base_type::comm_.rank();

        // perform minimal sanity checks
        SYCL_LSH_DEBUG0_ASSERT(0 < total_size, "Illegal total size!");
        SYCL_LSH_DEBUG0_ASSERT(0 < rank_size, "Illegal rank size!");
        SYCL_LSH_DEBUG0_ASSERT(0 < dims, "Illegal number of dimensions!");

        // check for correct type
        MPI_Offset file_size;
        MPI_File_get_size(base_type::file_.get(), &file_size);  // get file size
        file_size -= 2 * sizeof(index_type);                    // subtract header information (size and dims)
        if (static_cast<index_type>(file_size) != total_size * dims * sizeof(parsing_type)) {
            if (comm_rank == 0) {
                fmt::print(stderr, "\nBroken file! File size ({}) doesn't match header information ({} * {} * sizeof(parsing_type) = {})\n\n",
                        file_size, total_size, dims, total_size * dims * sizeof(parsing_type));
            }
            MPI_Abort(base_type::comm_.get(), EXIT_FAILURE);
        }

        // calculate byte offsets per MPI rank
        constexpr index_type header_offset = 2 * sizeof(index_type);    // header information (size and dims)
        const index_type rank_offset = header_offset + comm_rank * rank_size * dims * sizeof(parsing_type);
        const index_type correct_rank_size = comm_rank  == comm_size - 1 ? (total_size - ((comm_size - 1) * rank_size)) : rank_size;

        // read data
        MPI_Status status;
        MPI_File_read_at(base_type::file_.get(), rank_offset, buffer, correct_rank_size * dims, type_cast<parsing_type>(), &status);

        // check whether the correct number of values were read
        int read_count;
        MPI_Get_count(&status, type_cast<parsing_type>(), &read_count);
        if (static_cast<index_type>(read_count) != correct_rank_size * dims) {
            fmt::print(stderr, "\nRead the wrong number of values on rank {}!. Expected {} values but read {} values.\n\n",
                    base_type::comm_.rank(), correct_rank_size * dims, read_count);
            MPI_Abort(base_type::comm_.get(), EXIT_FAILURE);
        }

        // fill missing data points ON THE LAST MPI RANK with dummy points
        if (comm_rank == comm_size - 1) {
            for (index_type point = correct_rank_size; point < rank_size; ++point) {
                for (index_type dim = 0; dim < dims; ++dim) {
                    buffer[point * dims + dim] = buffer[(correct_rank_size - 1) * dims + dim];
                }
            }
        }
        
        base_type::logger_.log("Parsed the data file in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BINARY_PARSER_HPP
