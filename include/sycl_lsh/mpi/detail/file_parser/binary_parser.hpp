/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief File parser for parsing plain binary data files.
 */

#ifndef SYCL_LSH_MPI_DETAIL_FILE_PARSER_BINARY_PARSER_HPP
#define SYCL_LSH_MPI_DETAIL_FILE_PARSER_BINARY_PARSER_HPP
#pragma once

#include "sycl_lsh/constants.hpp"                           // sycl_lsh::index_type
#include "sycl_lsh/detail/assert.hpp"                       // SYCL_LSH_ASSERT
#include "sycl_lsh/exceptions/exceptions.hpp"               // sycl_lsh::file_parsing_exception
#include "sycl_lsh/matrix.hpp"                              // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"                    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/file_parser/base_parser.hpp"  // sycl_lsh::mpi::detail::file_parser
#include "sycl_lsh/mpi/detail/file_parser/file.hpp"         // sycl_lsh::mpi::detail::{file, file::mode}
#include "sycl_lsh/mpi/detail/type_cast.hpp"                // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"                  // SYCL_LSH_MPI_ERROR_CHECK

#include "fmt/format.h"  // fmt::format
#include "mpi.h"         // MPI_File related functions

#include <cstdint>  // std::uint64_t
#include <string>   // std::string

namespace sycl_lsh::mpi::detail {

/**
 * @brief File parser class for a custom **binary** data format.
 * @details Expected file format: header information with the size of the used real_type in bytes, the total number of data points, and the number of dimensions followed by
 *          the data in *Array of Structs* format.
 *
 * Example (in text format, correct files **must** be in binary form):
 * 4
 * 4
 * 2
 * 0.0 0.1
 * 0.2 0.3
 * 0.4 0.5
 * 0.6 0.7
 * @tparam T the type of the data to parse
 * @note The first three rows must always be stored using a 64bit unsigned integer.
 */
template <typename T>
class binary_parser final : public file_parser<T> {
    using base_type = file_parser<T>;

    // make base class members visible
    using base_type::comm_;
    using base_type::file_;
    using base_type::file_name_;
    using base_type::mode_;

  public:
    /// The type of the data which should get parsed.
    using parsing_type = typename base_type::parsing_type;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::binary_parser object responsible for parsing the custom binary file format.
     * @param[in] file_name the file to parse
     * @param[in] mode the file open mode (@ref sycl_lsh::mpi::file::mode::read or @ref sycl_lsh::mpi::file::mode::write)
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    binary_parser(const std::string &file_name, file::mode mode, const communicator &comm);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  parsing                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Parse the **total** number of data points in the file.
     * @details Reads the total size from the first line of the file. The type must be of @ref options::index_type.
     * @return the total number of data points (`[[nodiscard]]`)
     */
    [[nodiscard]] index_type parse_total_size() const override;
    /**
     * @brief Parse the number of dimensions of each data point in the file.
     * @details Reads the number of dimensions from the second line of the file. The type must be of @ref options::index_type.\n
     *          Assumes that each data points has the same number of dimensions.
     * @return the number of dimensions (`[[nodiscard]]`)
     */
    [[nodiscard]] index_type parse_dims() const override;
    /**
     * @brief Parse the content of the file.
     * @details Fills the last elements of the buffer on the last MPI rank such that all MPI ranks have te same number of data points.\n
     *          Throws a `sycl_lsh::file_parsing_exception` if the file size doesn't match the header information or the actually read values
     *          diverge from the number of values which should theoretically be available.
     * @return the parsed data (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::file_parsing_exception if the file has been opened in write mode
     * @throws sycl_lsh::file_parsing_exception if the header information and type doesn't match with the file size
     * @throws sycl_lsh::file_parsing_exception if the buffer isn't big enough
     * @throws sycl_lsh::file_parsing_exception if the size of the parsing_type does not match the size used to write the file data
     */
    [[nodiscard]] aos_matrix<parsing_type> parse_content() const override;
    /**
     * @brief Write the content in @p buffer to the file.
     * @param[in] total_size the total number of values to write (sum of all values from **all** MPI ranks)
     * @param[in] dims the number of dimensions of each value
     * @param[in] buffer the data to write to the file
     *
     * @note Throws a `sycl_lsh::file_parsing_exception` if the file has been opened in read mode.
     */
    void write_content(index_type total_size, index_type dims, const aos_matrix<parsing_type> &buffer) const override;

  private:
    /**
     * @brief Check if the file content is stored using the correct parsing_type.
     * @throws sycl_lsh::file_parsing_exception if the size of the parsing type does not match the size used to write the file data
     */
    void validate_parsing_type() const;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <typename T>
binary_parser<T>::binary_parser(const std::string &file_name, const file::mode mode, const communicator &comm) :
    file_parser<T>{ file_name, mode, comm } {
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                  parsing                                                   //
// ---------------------------------------------------------------------------------------------------------- //
template <typename T>
void binary_parser<T>::validate_parsing_type() const {
    // read first line containing the size of the stored parsing_type in bytes
    std::uint64_t real_type_size{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_File_read_at(file_.get(), 0, &real_type_size, 1, detail::mpi_datatype<std::uint64_t>(), MPI_STATUS_IGNORE));
    if (real_type_size != sizeof(real_type)) {
        throw file_parsing_exception{ fmt::format("The data was stored using a {} Byte type but is now read using a {} Byte type which is not supported!", real_type_size, sizeof(parsing_type)) };
    }
}

template <typename T>
index_type binary_parser<T>::parse_total_size() const {
    // read the second line containing the total_size
    std::uint64_t total_size{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_File_read_at(file_.get(), sizeof(std::uint64_t), &total_size, 1, detail::mpi_datatype<std::uint64_t>(), MPI_STATUS_IGNORE));
    return static_cast<index_type>(total_size);
}

template <typename T>
index_type binary_parser<T>::parse_dims() const {
    // read the third line containing the number of dimensions
    std::uint64_t dims{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_File_read_at(file_.get(), 2 * sizeof(std::uint64_t), &dims, 1, detail::mpi_datatype<std::uint64_t>(), MPI_STATUS_IGNORE));
    return static_cast<index_type>(dims);
}

template <typename T>
auto binary_parser<T>::parse_content() const -> aos_matrix<parsing_type> {
    // throw if file has been opened in the wrong mode
    if (mode_ == file::mode::write) {
        throw file_parsing_exception{ "Can't read from a file opened in write mode!" };
    }

    // check that the parsing_type used to store the data is the same as the one used to read it
    this->validate_parsing_type();
    const index_type total_size = this->parse_total_size();
    const index_type rank_size = this->parse_rank_size();
    const index_type dims = this->parse_dims();
    constexpr index_type header_offset = 3 * sizeof(std::uint64_t);  // header information (parsing_type size in bytes, total size, and number of dims)

    // perform minimal sanity checks
    SYCL_LSH_ASSERT(0 < total_size, "Illegal total size!");
    SYCL_LSH_ASSERT(0 < rank_size, "Illegal rank size!");
    SYCL_LSH_ASSERT(0 < dims, "Illegal number of dimensions!");

    sycl_lsh::aos_matrix<parsing_type> buffer{ sycl_lsh::detail::shape{ rank_size, dims } };

    // check for correct type
    MPI_Offset file_size{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_File_get_size(file_.get(), &file_size));  // get file size
    file_size -= header_offset;                                            // subtract header information (size and dims)
    if (static_cast<index_type>(file_size) != total_size * dims * sizeof(parsing_type)) {
        throw file_parsing_exception{ fmt::format("Broken file '{}'! File size ({}) doesn't match header information ({} * {} * sizeof(parsing_type) = {})", file_name_, file_size, total_size, dims, total_size * dims * sizeof(parsing_type)) };
    }

    const int comm_size = comm_.size();
    const int comm_rank = comm_.rank();

    // calculate byte offsets per MPI rank
    const index_type rank_offset = header_offset + comm_rank * rank_size * dims * sizeof(parsing_type);
    const index_type correct_rank_size = (comm_rank == comm_size - 1) ? (total_size - ((comm_size - 1) * rank_size)) : rank_size;

    // check if the provided buffer is big enough
    if (correct_rank_size * dims > buffer.size()) {
        throw file_parsing_exception{ fmt::format("Trying to write {} values to '{}', but the size of the buffer is only {}!", correct_rank_size * dims, file_name_, buffer.size()) };
    }

    // read data
    MPI_Status status;
    MPI_File_read_at(file_.get(), rank_offset, buffer.data(), correct_rank_size * dims, detail::mpi_datatype<parsing_type>(), &status);

    // check whether the correct number of values were read
    int read_count{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Get_count(&status, detail::mpi_datatype<parsing_type>(), &read_count));
    if (static_cast<index_type>(read_count) != correct_rank_size * dims) {
        throw file_parsing_exception{ fmt::format("Read the wrong number of values on rank {} from '{}'!. Expected {} values but read {} values.", comm_rank, file_name_, correct_rank_size * dims, read_count) };
    }

    // fill missing data points ON THE LAST MPI RANK with dummy points
    if (comm_rank == comm_size - 1) {
        for (index_type point = correct_rank_size; point < rank_size; ++point) {
            for (index_type dim = 0; dim < dims; ++dim) {
                buffer(point, dim) = buffer(correct_rank_size - 1, dim);
            }
        }
    }

    return buffer;
}

template <typename T>
void binary_parser<T>::write_content(const index_type total_size, const index_type dims, const aos_matrix<parsing_type> &buffer) const {
    // throw if file has been opened in the wrong mode
    if (mode_ == file::mode::read) {
        throw file_parsing_exception{ "Can't write to a file opened in read mode!" };
    }

    // write header information
    if (comm_.is_main_rank()) {
        SYCL_LSH_MPI_ERROR_CHECK(MPI_File_write(file_.get(), &total_size, 1, detail::mpi_datatype<index_type>(), MPI_STATUS_IGNORE));
        SYCL_LSH_MPI_ERROR_CHECK(MPI_File_write(file_.get(), &dims, 1, detail::mpi_datatype<index_type>(), MPI_STATUS_IGNORE));
    }
    comm_.barrier();
    SYCL_LSH_MPI_ERROR_CHECK(MPI_File_seek_shared(file_.get(), 0, MPI_SEEK_END));

    // write actual content
    index_type correct_rank_size = buffer.size() / dims;
    if (comm_.rank() == comm_.size() - 1) {
        correct_rank_size = total_size - ((comm_.size() - 1) * correct_rank_size);
    }
    SYCL_LSH_MPI_ERROR_CHECK(MPI_File_write_ordered(file_.get(), buffer.data(), correct_rank_size * dims, detail::mpi_datatype<parsing_type>(), MPI_STATUS_IGNORE));
}

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_FILE_PARSER_BINARY_PARSER_HPP
