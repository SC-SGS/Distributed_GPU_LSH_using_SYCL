/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Factory function to create a specific file parser.
 */

#ifndef SYCL_LSH_MPI_DETAIL_FILE_PARSER_FILE_PARSER_HPP
#define SYCL_LSH_MPI_DETAIL_FILE_PARSER_FILE_PARSER_HPP
#pragma once

#include "sycl_lsh/mpi/communicator.hpp"                      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/file_parser/arff_parser.hpp"    // sycl_lsh::mpi::detail::arff_parser
#include "sycl_lsh/mpi/detail/file_parser/base_parser.hpp"    // sycl_lsh::mpi::detail::file_parser
#include "sycl_lsh/mpi/detail/file_parser/binary_parser.hpp"  // sycl_lsh::mpi::detail::binary_parser
#include "sycl_lsh/mpi/detail/file_parser/file.hpp"           // sycl_lsh::mpi::detail::file::mode
#include "sycl_lsh/mpi/file_parser_types.hpp"                 // sycl_lsh::mpi::file_parser_type

#include <memory>  // std::unique_ptr, std::make_unique

namespace sycl_lsh::mpi::detail {

/**
 * @brief Creates a new file parser.
 * @tparam parsing_type the type to parse
 * @param[in] filename the name of the file
 * @param[in] file_parser the type of the file parser to use
 * @param[in] mode the file open mode
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @return a file parser with the requested type (`[[nodiscard]]`)
 */
template <typename parsing_type>
[[nodiscard]] std::unique_ptr<file_parser<parsing_type>> make_file_parser(const std::string &filename, const file_parser_type file_parser, const file::mode mode, const communicator &comm) {
    switch (file_parser) {
        case file_parser_type::arff:
            return std::make_unique<arff_parser<parsing_type>>(filename, mode, comm);
        case file_parser_type::binary:
            return std::make_unique<binary_parser<parsing_type>>(filename, mode, comm);
    }
    // unreachable
    return nullptr;
}

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_FILE_PARSER_FILE_PARSER_HPP
