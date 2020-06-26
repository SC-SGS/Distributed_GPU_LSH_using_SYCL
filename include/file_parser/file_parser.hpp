/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-26
 *
 * @brief Factory function to create a specific file parser based on the file extension of the given file.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_PARSER_HPP

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <file_parser/arff_parser.hpp>
#include <file_parser/base_parser.hpp>
#include <file_parser/default_parser.hpp>

#include <config.hpp>

/**
 * @brief Creates a new distributed file parser (using MPI IO) based on the file extension of @p file.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @param[in] file_name the path to the data file
 * @param[in] communicator the *MPI_Comm* communicator used for the distributed parsing
 * @return the specific file parser for parsing @p file (`[[nodiscard]]`)
 */
template <typename Options>
[[nodiscard]]
inline std::unique_ptr<file_parser<Options>> make_file_parser(const std::string& file_name, const MPI_Comm& communicator) {
    std::filesystem::path path(file_name);

    // check if file exists
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("File '" + file_name + "' doesn't exist!");
    }

    // create file parser based on file extension
    if (path.extension() == ".arff") {
        return std::make_unique<arff_parser<Options>>(file_name, communicator);
    } else {
        return std::make_unique<default_parser<Options>>(file_name, communicator);
    }
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_PARSER_HPP
