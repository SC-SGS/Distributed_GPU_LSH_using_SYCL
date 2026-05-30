/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Base class for all different file parsers.
 * @details Pure virtual.
 */

#ifndef SYCL_LSH_FILE_PARSER_BASE_PARSER_HPP
#define SYCL_LSH_FILE_PARSER_BASE_PARSER_HPP
#pragma once

#include "sycl_lsh/constants.hpp"             // sycl_lsh::index_type
#include "sycl_lsh/detail/matrix.hpp"         // sycl_lsh::detail::matrix
#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/file_parser/file.hpp"  // sycl_lsh::mpi::file
#include "sycl_lsh/mpi/logger.hpp"            // sycl_lsh::mpi::logger

#include <cmath>        // std::ceil
#include <string>       // std::string
#include <type_traits>  // std::is_arithmetic_v
#include <vector>       // std::vector

namespace sycl_lsh::mpi {

/**
 * @brief Pure virtual base class for all different file parsers.
 * @tparam T the type of the data to parse
 */
template <typename T>
class file_parser {
    static_assert(std::is_arithmetic_v<T>, "The can only parse arithmetic types!");

  public:
    /// The type of the data which should get parsed.
    using parsing_type = T;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                         constructor and destructor                                         //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::file_parser object and opens the file @p file_name.
     * @param[in] file_name the file to parse
     * @param[in] mode the file open mode (@ref sycl_lsh::mpi::file::mode::read or @ref sycl_lsh::mpi::file::mode::write)
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    file_parser(const std::string &file_name, file::mode mode, const communicator &comm, const logger &logger);
    /**
     * @brief Virtual destructor to enable proper inheritance.
     */
    virtual ~file_parser() = default;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  parsing                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Parse the **total** number of data points in the file.
     * @return the total number of data points (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_total_size() const = 0;
    /**
     * @brief Parse the number of data points **per MPI rank** of the file.
     * @details If the total number of data points isn't dividable by the MPI communicator size,
     *          the **last** MPI rank will be filled with dummy points.\n
     *          Example:\n
     *          data = xxxxxxxxxx (size = 10), communicator size = 3\n
     *          rank 1: xxxx, rank 2: xxxx, rank 3: xxdd\n
     *          -> the rank size is 4 for each MPI rank!
     * @return the number of data points per MPI rank (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_rank_size() const;
    /**
     * @brief Parse the number of dimensions of each data point in the file.
     * @details Assumes that each data points has the same number of dimensions.
     * @return the number of dimensions (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_dims() const = 0;
    /**
     * @brief Parse the content of the file.
     * @return the parsed data (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual sycl_lsh::detail::aos_matrix<parsing_type> parse_content() const = 0;
    /**
     * @brief Write the content in @p buffer to the file.
     * @param[in] total_size the total number of values to write (sum of all values from **all** MPI ranks)
     * @param[in] dims the number of dimensions of each value
     * @param[in] buffer the data to write to the file
     */
    virtual void write_content(index_type total_size, index_type dims, const sycl_lsh::detail::aos_matrix<parsing_type> &buffer) const = 0;

  protected:
    /// The used MPI communicator.
    communicator comm_;
    /// The used MPI-aware logger.
    const logger &logger_;
    /// The file name. Mainly used for better error message.
    std::string file_name_;
    /// The used MPI file wrapper.
    file file_;
    /// The file access mode (read, write, read-write).
    file::mode mode_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <typename T>
file_parser<T>::file_parser(const std::string &file_name, const file::mode mode, const communicator &comm, const logger &logger) :
    comm_{ comm },
    logger_{ logger },
    file_name_{ file_name },
    file_{ file_name, comm, mode },
    mode_{ mode } { }

// ---------------------------------------------------------------------------------------------------------- //
//                                                  parsing                                                   //
// ---------------------------------------------------------------------------------------------------------- //
template <typename T>
index_type file_parser<T>::parse_rank_size() const {
    // read the total size
    const index_type total_size = this->parse_total_size();
    return static_cast<index_type>(std::ceil(total_size / static_cast<float>(comm_.size())));
}

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_FILE_PARSER_BASE_PARSER_HPP
