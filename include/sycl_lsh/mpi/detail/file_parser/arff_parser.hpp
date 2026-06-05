/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief File parser for parsing `.arff` data files.
 */

#ifndef SYCL_LSH_MPI_DETAIL_FILE_PARSER_ARFF_PARSER_HPP
#define SYCL_LSH_MPI_DETAIL_FILE_PARSER_ARFF_PARSER_HPP
#pragma once

#include "sycl_lsh/constants.hpp"                           // sycl_lsh::index_type
#include "sycl_lsh/exceptions/exceptions.hpp"               // sycl_lsh::not_implemented_exception
#include "sycl_lsh/matrix.hpp"                              // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"                    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/file_parser/base_parser.hpp"  // sycl_lsh::mpi::detail::file_parser
#include "sycl_lsh/mpi/detail/file_parser/file.hpp"         // sycl_lsh::mpi::detail::file::mode

#include <string>  // std::string

namespace sycl_lsh::mpi::detail {

/**
 * @brief File parser class for the **arff** data format.
 * @details Not yet implemented.
 * @tparam T the type of the data to parse
 */
template <typename T>
class arff_parser final : public file_parser<T> {
    /// The arff_parser's base type.
    using base_type = file_parser<T>;

  public:
    /// The type of the data which should get parsed.
    using parsing_type = typename base_type::parsing_type;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::detail::arff_parser object responsible for parsing
     *        [.arff](https://www.cs.waikato.ac.nz/~ml/weka/arff.html) files.
     * @param[in] file_name the file to parse
     * @param[in] mode the file open mode (@ref sycl_lsh::mpi::detail::file::mode::read or @ref sycl_lsh::mpi::detail::file::mode::write)
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     *
     * @throws sycl_lsh::exception since .arff files aren't currently supported.
     */
    arff_parser(const std::string &file_name, file::mode mode, const communicator &comm);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  parsing                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Parse the **total** number of data points in the file.
     * @return the total number of data points (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::not_implemented since `.arff` files aren't currently supported.
     */
    [[nodiscard]] index_type parse_total_size() const override;
    /**
     * @brief Parse the number of dimensions of each data point in the file.
     * @return the number of dimensions (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::not_implemented since `.arff` files aren't currently supported.
     */
    [[nodiscard]] index_type parse_dims() const override;
    /**
     * @brief Parse the content of the file.
     * @return the parsed data (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::not_implemented since .arff files aren't currently supported.
     */
    [[nodiscard]] aos_matrix<parsing_type> parse_content() const override;
    /**
     * @brief Write the content in @p buffer to the file.
     * @param[in] total_size the total number of values to write (sum of all values from **all** MPI ranks)
     * @param[in] dims the number of dimensions of each value
     * @param[in] buffer the data to write to the file
     *
     * @throws sycl_lsh::not_implemented since .arff files aren't currently supported.
     */
    void write_content(index_type total_size, index_type dims, const aos_matrix<parsing_type> &buffer) const override;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <typename T>
arff_parser<T>::arff_parser(const std::string &file_name, const file::mode mode, const communicator &comm) :
    file_parser<T>{ file_name, mode, comm } {
    throw not_implemented_exception{ "Parsing an '.arff' file is currently no supported! Maybe run data_sets/convert_arff_to_binary.py first?" };
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                  parsing                                                   //
// ---------------------------------------------------------------------------------------------------------- //
template <typename T>
index_type arff_parser<T>::parse_total_size() const {
    throw not_implemented_exception{};
}

template <typename T>
index_type arff_parser<T>::parse_dims() const {
    throw not_implemented_exception{};
}

template <typename T>
auto arff_parser<T>::parse_content() const -> aos_matrix<parsing_type> {
    throw not_implemented_exception{};
}

template <typename T>
void arff_parser<T>::write_content(const index_type, const index_type, const aos_matrix<parsing_type> &) const {
    throw not_implemented_exception{};
}

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_FILE_PARSER_ARFF_PARSER_HPP
