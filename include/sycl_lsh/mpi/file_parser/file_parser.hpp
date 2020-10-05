/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-05
 *
 * @brief Factory function to create a specific file parser based on the provided command line argument.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_PARSER_HPP

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/file.hpp>
#include <sycl_lsh/mpi/file_parser/arff_parser.hpp>
#include <sycl_lsh/mpi/file_parser/base_parser.hpp>
#include <sycl_lsh/mpi/file_parser/binary_parser.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/options.hpp>

#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sycl_lsh::mpi {

    /**
     * @brief Creates a new file parser based on the provided command line argument *file_parser* in @p parser.
     * @details If the *file_parser* argument isn't provided, returns a @ref sycl_lsh::mpi::binary_parser as fall back.
     * @tparam parsing_type the type to parse
     * @tparam Options the used @ref sycl_lsh::options type
     * @param[in] file_name the name of the file to open
     * @param[in] parser the @ref sycl_lsh::argv_parser
     * @param[in] mode the file open mode (@ref sycl_lsh::mpi::file::mode::read or @ref sycl_lsh::mpi::file::mode::write)
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     * @return a file parser with the requested type (`[[nodiscard]]`)
     *
     * @throw std::invalid_argument if the provided file parser type can't be recognized.
     */
    template <typename parsing_type, typename Options>
    [[nodiscard]]
    inline std::unique_ptr<file_parser<Options, parsing_type>> make_file_parser(const std::string_view file_name, const argv_parser& parser,
                                                                                const file::mode mode, const communicator& comm, const logger& logger)
    {
        std::string file_parser_name;
        // try getting the file parser name, if not provided fall back to the 'binary_parser'
        try {
            file_parser_name = parser.argv_as<std::string>("file_parser");
        } catch (const std::invalid_argument&) {
            logger.log("\nNo file parser type specified! Using the 'binary_parser' as fall back.\n");
            return std::make_unique<binary_parser<Options, parsing_type>>(file_name, mode, comm, logger);
        }

        if (file_parser_name == "arff_parser") {
            // using the arff file parser
            return std::make_unique<arff_parser<Options, parsing_type>>(file_name, mode, comm, logger);
        } else if (file_parser_name == "binary_parser") {
            // using the binary file parser
            return std::make_unique<binary_parser<Options, parsing_type>>(file_name, mode, comm, logger);
        } else {
            throw std::invalid_argument(fmt::format("Unrecognized file parser type '{}'!", file_parser_name));
        }

    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_PARSER_HPP
