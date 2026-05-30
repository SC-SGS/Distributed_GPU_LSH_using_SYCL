/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Minimalistic wrapper class around an MPI file.
 */

#ifndef SYCL_LSH_MPI_DETAIL_FILE_PARSER_FILE_HPP
#define SYCL_LSH_MPI_DETAIL_FILE_PARSER_FILE_HPP
#pragma once

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "mpi.h"          // MPI file related functionality

#include <iosfwd>  // std::ostream and std::istream forward declarations
#include <string>  // std::string

namespace sycl_lsh::mpi::detail {

/**
 * @brief Minimalistic wrapper around a MPI file.
 */
class file {
  public:
    /**
     * @brief Enum class for the different file open types (read or write).
     */
    enum class mode {
        /** open file in read only mode */
        read = MPI_MODE_RDONLY,
        /** open file in write only mode */
        write = MPI_MODE_WRONLY | MPI_MODE_APPEND | MPI_MODE_CREATE | MPI_MODE_EXCL
    };

    // ---------------------------------------------------------------------------------------------------------- //
    //                                        constructors and destructor                                         //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::file, i.e., open the file @p file_name in the mode @p open_mode.
     * @details If the file should be opened in write mode and already exists, it gets deleted before the first write.
     * @param[in] file_name the file to open
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] open_mode the open mode (read or write)
     *
     * @throws std::invalid_argument if the file @p file_name doesn't exist and the open mode is `read`
     */
    file(const std::string &file_name, const communicator &comm, mode open_mode);
    /**
     * @brief Delete the copy constructor.
     */
    file(const file &) = delete;
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::file from the resources hold by @p other.
     * @param[in,out] other the @ref sycl_lsh::mpi::file to move-from
     */
    file(file &&other) noexcept;
    /**
     * @brief Destruct the @ref sycl_lsh::mpi::file object, i.e., closes the previously opened file.
     */
    ~file();

    // ---------------------------------------------------------------------------------------------------------- //
    //                                            assignment operators                                            //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Delete the copy assignment operator.
     */
    file &operator=(const file &) = delete;
    /**
     * @brief Move-assigns @p rhs to `*this`.
     * @param[in] rhs the @ref sycl_lsh::mpi::file to move-from
     * @return `*this`
     */
    file &operator=(file &&rhs) noexcept;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Get the underlying MPI file.
     * @return the MPI file wrapped in this @ref sycl_lsh::mpi::file object (`[[nodiscard]]`)
     */
    [[nodiscard]] const MPI_File &get() const noexcept { return file_; }

    /**
     * @brief Get the underlying MPI file.
     * @return the MPI file wrapped in this @ref sycl_lsh::mpi::file object (`[[nodiscard]]`)
     */
    [[nodiscard]] MPI_File &get() noexcept { return file_; }

  private:
    /// The wrapped MPI file.
    MPI_File file_;
};

/**
 * @brief Output the @p mode to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the file open mode to
 * @param[in] mode the file open mode
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, file::mode mode);

/**
 * @brief Use the input-stream @p in to initialize the file open @p mode.
 * @param[in,out] in input-stream to extract the file open mode from
 * @param[in] mode the file open mode
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, file::mode &mode);

}  // namespace sycl_lsh::mpi::detail

template <>
struct fmt::formatter<sycl_lsh::mpi::detail::file::mode> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_MPI_DETAIL_FILE_PARSER_FILE_HPP
