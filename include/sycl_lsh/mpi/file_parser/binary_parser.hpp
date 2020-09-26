/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-26
 *
 * @brief File parser for parsing plain binary data files.
 */


#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BINARY_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BINARY_PARSER_HPP

#include <sycl_lsh/mpi/file_parser/base_parser.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>
#include <sycl_lsh/detail/assert.hpp>

#include <fmt/format.h>
#include <mpi.h>

#include <cstdlib>
#include <cmath>
#include <string_view>

namespace sycl_lsh::mpi {

    template <typename Options, typename T>
    class binary_parser final : public file_parser<Options, T> {
        using base_type = file_parser<Options, T>;
    public:
        using index_type = typename base_type::index_type;
        using parsing_type = typename base_type::parsing_type;

        // TODO 2020-09-25 13:10 marcel: message?
        binary_parser(const std::string_view file_name, const communicator& comm) : file_parser<Options, T>(file_name, comm) { }

        [[nodiscard]]
        index_type parse_total_size() const override {
            // read first line containing the total_size
            index_type total_size;
            MPI_File_read_at(base_type::file_.get(), 0, &total_size, 1, type_cast<index_type>(), MPI_STATUS_IGNORE);
            return total_size;
        }

        [[nodiscard]]
        index_type parse_rank_size() const override {
            // read the total size
            const index_type total_size = this->parse_total_size();
            return static_cast<index_type>(std::ceil(total_size / static_cast<float>(base_type::comm_.size())));
        }

        [[nodiscard]]
        index_type parse_dims() const override {
            index_type dims;
            MPI_File_read_at(base_type::file_.get(), sizeof(index_type), &dims, 1, type_cast<index_type>(), MPI_STATUS_IGNORE);
            return dims;
        }

        void parse_content(parsing_type* buffer) const override {
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
                    fmt::print(stderr, "Broken file! File size ({}) doesn't match header information ({} * {} * sizeof(parsing_type) = {})\n",
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
            assert((static_cast<index_type>(read_count) == correct_rank_size * dims));
            if (static_cast<index_type>(read_count) != correct_rank_size * dims) {
                fmt::print(stderr, "Read the wrong number of values on rank {}!. Expected {} values but read {} values.\n",
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
        }
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BINARY_PARSER_HPP
