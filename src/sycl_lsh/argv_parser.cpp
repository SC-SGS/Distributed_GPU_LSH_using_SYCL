/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 */

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/detail/assert.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>


const std::map<std::string, std::pair<std::string, bool>> sycl_lsh::argv_parser::list_of_argvs_ = {
        { "help",                   { "help screen", false } },
        { "data_file",              { "path to the data file", true } },
        { "file_parser",            { "type of the file parser", false } },
        { "k",                      { "the number of nearest-neighbors to search for", true } },
        { "options_file",           { "path to options file", false } },
        { "options_save_file",      { "save the currently used options to the given path", false } },
        { "knn_save_file",          { "save the calculated nearest-neighbors to path", false } },
        { "knn_dist_save_file",     { "save the calculated nearest-neighbor distances to path", false } },
        { "evaluate_knn_file",      { "read the correct nearest-neighbors for calculating the resulting recall", false } },
        { "evaluate_knn_dist_file", { "read the correct nearest-neighbor distances for calculating the error ratio", false } },
        { "hash_pool_size",         { "number of hash functions in the hash pool", false } },
        { "num_hash_functions",     { "number of hash functions per hash table", false } },
        { "num_hash_tables",        { "number of hash tables to create", false } },
        { "hash_table_size",        { "size of each hash table", false } },
        { "w",                      { "segment size for the random projections hash functions", false } },
        { "num_cut_off_points",     { "number of cut-off points for the entropy-based hash functions", false } }
};


// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::argv_parser::argv_parser(const int argc, char** argv) {
    SYCL_LSH_DEBUG_ASSERT(argc >= 1, "Illegal number of command line arguments! Must be greater or equal than 1.");
    SYCL_LSH_DEBUG_ASSERT(argv != nullptr, "Illegal command line argument parameter! Must not be the nullptr.");

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];

        // check whether the key starts with two leading "--"
        if (key.rfind("--", 0) != 0) {
            throw std::invalid_argument(fmt::format("All command line argument keys must start with '--' ({})!", key));
        }
        key.erase(0, 2);    // remove leading '--'

        // check whether the key is legal
        if (list_of_argvs_.count(key) == 0) {
            throw std::invalid_argument(fmt::format("Illegal command line argument key {}!", key));
        }
        // check whether the key hasn't been provided yet
        if (argvs_.count(key) > 0) {
            throw std::invalid_argument(fmt::format("Duplicate command line argument key {}!", key));
        }

        if (key == "help") {
            // if the current key equals 'help' continue parsing the next [key, value]-pair
            // -> DON'T read a value because none will be provided!
            argvs_.emplace(std::move(key), "");
        } else {
            // check whether a value is present
            if (i + 1 >= argc) {
                throw std::invalid_argument("Command line argument key has no value!");
            }
            // check whether the next value isn't a key
            if (key.rfind("--", 0) == 0) {
                throw std::invalid_argument(fmt::format("Expected command line argument value but got another key {}!", argv[i + 1]));
            }

            // add the [key, value]-pair to parsed command line arguments
            argvs_.emplace(std::move(key), argv[++i]);
        }
    }

    // if '--help' isn't provided, check whether the required command line arguments are present
    // if '--help' is provided, the required command line arguments do not have to be present
    if (argvs_.count("help") == 0) {
        for (const auto& [key, desc] : list_of_argvs_) {
            if (desc.second && argvs_.count(key) == 0) {
                throw std::logic_error(fmt::format("The required command line key '{}' is missing!", key));
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------- //
//                                           other member functions                                           //
// ---------------------------------------------------------------------------------------------------------- //
bool sycl_lsh::argv_parser::has_argv(const std::string& key) const { return argvs_.count(key) > 0; }

std::string sycl_lsh::argv_parser::description() {
    const auto max_reduction = [](const std::size_t value, const auto& pair) { return std::max(value, pair.first.size()); };
    const std::size_t alignment_size = std::accumulate(list_of_argvs_.begin(), list_of_argvs_.end(), 0, max_reduction) + 2;

    fmt::memory_buffer buf;

    // write header information
    fmt::format_to(buf, "Usage: ./prog --data_file \"path_to_data_file\" --k \"number_of_knn_to_search\" [options]\n");
    fmt::format_to(buf, "options:\n");

    // write command line arguments and their respective description
    for (const auto& [key, desc] : list_of_argvs_) {
        fmt::format_to(buf, "   --{:<{}} {} {}\n", key, alignment_size, desc.first, desc.second ? "(required)" : "");
    }

    return fmt::to_string(buf);
}