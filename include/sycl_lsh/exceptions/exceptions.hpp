/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements custom exception classes derived from [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)
 *        including source location information.
 */

#ifndef SYCL_LSH_EXCEPTIONS_EXCEPTIONS_HPP
#define SYCL_LSH_EXCEPTIONS_EXCEPTIONS_HPP
#pragma once

#include "sycl_lsh/exceptions/source_location.hpp"  // sycl_lsh::source_location

#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace sycl_lsh {

/**
 * @brief Base class for all custom exception types. Forwards its message to [std::runtime_error](https://en.cppreference.com/w/cpp/error/runtime_error)
 *        and saves the exception name and the call side source location information.
 */
class exception : public std::runtime_error {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message to [std::runtime_error](https://en.cppreference.com/w/cpp/error/runtime_error).
     * @param[in] msg the exception's `what()` message
     * @param[in] class_name the name of the thrown exception class
     * @param[in] loc the exception's call side information
     */
    explicit exception(const std::string &msg, std::string_view class_name = "exception", source_location loc = source_location::current());

    /**
     * @brief Returns the information of the call side where the exception was thrown.
     * @return the exception's call side information (`[[nodiscard]]`)
     */
    [[nodiscard]] const source_location &loc() const noexcept;

    /**
     * @brief Returns a string containing the exception's `what()` message, the name of the thrown exception class, and information about the call
     *        side where the exception has been thrown.
     * @return the exception's `what()` message including source location information (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string what_with_loc() const;

  private:
    /// The name of the thrown exception class.
    std::string_view class_name_;
    /// The call side source location information.
    source_location loc_;
};

/**
 * @brief Exception type thrown for early exit in the cmd parser constructor.
 * @details Used for a graceful tear down.
 */
class cmd_parser_exit : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exit code and source location to @ref sycl_lsh::exception .
     * @param[in] exit_code the exit code
     * @param[in] loc the exception's call side information
     */
    explicit cmd_parser_exit(int exit_code, source_location loc = source_location::current());

    /**
     * @brief Return the previously defined exit code.
     * @return the exit code (`[[nodiscard]]`)
     */
    [[nodiscard]] int exit_code() const noexcept { return exit_code_; }

  private:
    /// The exit code.
    int exit_code_{};
};

/**
 * @brief Exception type thrown if a function is called that is not (yet) implemented.
 */
class not_implemented_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit not_implemented_exception(const std::string &msg = "Function currently not implemented!", source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if something regarding our MPI wrapper went wrong.
 */
class mpi_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit mpi_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if something went wrong in the MPI_File functionality.
 */
class file_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit file_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if something went wrong during file parsing.
 */
class file_parsing_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit file_parsing_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if something went wrong when using the device_ptr wrapper.
 */
class device_ptr_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit device_ptr_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if something went wrong when using the matrix class.
 */
class matrix_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit matrix_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if any of the Locality Sensitive Hashing related options is invalid.
 */
class invalid_lsh_option_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to @ref sycl_lsh::exception .
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit invalid_lsh_option_exception(const std::string &msg, source_location loc = source_location::current());
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_EXCEPTIONS_EXCEPTIONS_HPP
