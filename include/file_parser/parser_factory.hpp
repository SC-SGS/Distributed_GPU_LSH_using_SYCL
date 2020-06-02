/**
 * @brief
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PARSER_FACTORY_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PARSER_FACTORY_HPP


#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <file_parser/file_parser.hpp>
#include <file_parser/default_parser.hpp>
#include <file_parser/arff_parser.hpp>


#include <config.hpp>


template <memory_layout layout, typename Options>
std::unique_ptr<file_parser<layout, Options>> make_file_parser(std::string file) {
    std::filesystem::path path(file);

    // check if file exists
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("File '" + file + "' doesn't exist!");
    }

    // create file parser based on file extension
    if (path.extension() == ".arff") {
        return std::make_unique<arff_parser<layout, Options>>(std::move(file));
    } else {
        return std::make_unique<default_parser<layout, Options>>(std::move(file));
    }
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PARSER_FACTORY_HPP
