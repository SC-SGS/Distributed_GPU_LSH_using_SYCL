/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-31
 *
 * @brief Implements a very simple command line argument parser specifically for this project.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP

#include <cstdio>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <detail/assert.hpp>
#include <detail/convert.hpp>


/**
 * @brief Class to parse command line arguments.
 * @details The supported command line options are:
 * | command line argument | description                                                                                  |
 * |:----------------------|:---------------------------------------------------------------------------------------------|
 * | help                  | Prints the help screen.                                                                      |
 * | options               | Path to the options file to load.                                                            |
 * | save_options          | Path to the file to save the currently used options to.                                      |
 * | data                  | Path to the data file to load (**required**).                                                |
 * | k                     | The number of nearest-neighbors to search for (**required**).                                |
 * | save_knn              | Path to the file to save the found k-nearest-neighbors to.                                   |
 * | num_hash_tables       | The number of used hash tables.                                                              |
 * | hash_table_size       | The size of each hash table.                                                                 |
 * | num_hash_functions    | The number of hash functions to calculate the hash values with.                              |
 * | w                     | A constant used in the hash functions calculation: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$. |
 */
class argv_parser {
    /// A list of all legal command line options including their description.
    const std::map<std::string, std::string> possible_argvs_ = {
            { "help", "help screen" },
            { "options", "path to options file" },
            { "save_options", "save the currently used options to path" },
            { "data", "path to the data file (required)" },
            { "k", "the number of nearest-neighbors to search for (required)" },
            { "save_knn", "save the calculate nearest-neighbors to path" },
            { "evaluate_knn", "read the correct nearest-neighbors and evaluate computed nearest-neighbors" },
            { "hash_pool_size", "number of hash functions in the hash pool" },
            { "num_cut_off_points", "number of cut-off points for the entropy-based hash functions" },
            { "num_hash_tables", "number of hash tables to create" },
            { "hash_table_size", "size of each hash table (must be a prime)" },
            { "num_hash_functions", "number of hash functions per hash table" },
            { "w", "constant used in the hash functions" }
    };

public:
    /**
     * @brief Parse the given command line arguments.
     * @param[in] argc the number of provided command line arguments
     * @param[in] argv the command line arguments
     * @param[in] comm_rank the current MPI rank
     *
     * @throws std::invalid_argument if **any** key **doesn't** starts with "--".
     * @throws std::invalid_argument if **any** key **isn't** a legal key according to @ref possible_argvs_.
     * @throws std::invalid_argument if **any** key has been provided more than once.
     * @throws std::logic_error if the **required** keys `--data` **and** `--k` were not given.
     *
     * @pre @p argc **must** be greater or equal than 1.
     * @pre @p argv **must not** be `nullptr`.
     */
    argv_parser(const int comm_rank, const int argc, char** argv) : comm_rank_(comm_rank) {
        DEBUG_ASSERT_MPI(comm_rank_, argc >= 1, "Not enough command line arguments given! {} >= 1", argc);
        DEBUG_ASSERT_MPI(comm_rank_, argv != nullptr, "argv must not be the nullptr!{}", "");

        for(int i = 1; i < argc; ++i) {
            std::string key = argv[i];

            // check whether the key starts with two leading "--"
            if (key.rfind("--", 0) != 0) throw std::invalid_argument("All argv keys must start with '--'!: " + key);
            key.erase(0, 2);

            // check whether the key is legal AND hasn't been provided yet
            if (possible_argvs_.count(key) == 0) throw std::invalid_argument("Illegal argv key!: " + key);
            if (argvs_.count(key) > 0) throw std::invalid_argument("Duplicate argv keys!: " + key);

            // if hte current key equals "help" continue with parsing the next [key, value]-pair
            // (DON'T read a value because none will be provided!)
            if (key == "help") {
                argvs_.emplace(std::move(key), "");
                continue;
            }

            // add [key, value]-pair to parsed command line arguments
            argvs_.emplace(std::move(key), argv[++i]);
        }

        // if --help isn't provided, check whether the required command line arguments are present
        // if --help is provided, the program immediately returns, i.e. the required arguments do not have to be present
        if (argvs_.count("help") == 0) {
            if (argvs_.count("data") == 0) {
                throw std::logic_error("The required command line argument --data is missing!");
            }
            if (argvs_.count("k") == 0) {
                throw std::logic_error("The required command line argument --k is missing!");
            }
        }
    }

    /**
     * @brief Check whether the provided command line arguments include @p key.
     * @details @p key **must not** start with '--'.
     * @tparam T the type of the key (**must** be convertible to [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string).
     * @param[in] key the key to check for
     * @return `true` if @p key is present, otherwise `false` (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] bool has_argv(T&& key) const {
        DEBUG_ASSERT_MPI(comm_rank_, possible_argvs_.count(key) > 0, "'{}' isn't a possible command line argument!", key);

        return argvs_.count(std::forward<T>(key)) > 0;
    }
    /**
     * @brief Returns the value associated with @p key.
     * @tparam T the returned type (**must** be either a [arithemtic type](https://en.cppreference.com/w/cpp/types/is_arithmetic) or
     *           [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string)).
     * @tparam U the type of the key (**must** be convertible to [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string).
     * @param[in] key the @p key to get the value for
     * @return the value associated with @p key converted to the type T (`[[nodiscard]]`)
     *
     * @throws std::invalid_argument if **any** key **isn't** a legal key according to @ref possible_argvs_.
     */
    template <typename T, typename U>
    [[nodiscard]] T argv_as(U&& key) const {
        // check whether key is present
        if (!this->has_argv(key)) {
            throw std::invalid_argument("The requested key '" + std::string(std::forward<U>(key)) + "' can't be found!");
        }
        // convert the value to the given type T and return it
        using type = std::decay_t<T>;
        if constexpr(std::is_same_v<type, std::string> || std::is_same_v<type, const char*> || std::is_same_v<type, char*>) {
            return argvs_.at(std::forward<U>(key));
        } else {
            return detail::convert_to<T>(argvs_.at(std::forward<U>(key)));
        }
    }
    /**
     * @brief Returns a description of all possible command line arguments.
     * @return the description (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string description() const {
        std::string desc("Usage: ./prog --data \"path-to-data_set\" --k \"number-of-knn\" [options]\noptions:\n");
        const auto max_reduction = [](const std::size_t value, const auto& key_value_p) { return std::max(value, key_value_p.first.size()); };
        const std::size_t max_size = std::accumulate(possible_argvs_.begin(), possible_argvs_.end(), 0, max_reduction);
        for (const auto& [key, value] : possible_argvs_) {
            desc += "   --" + key + std::string(max_size - key.size() + 2, ' ') + value + '\n';
        }
        return desc;
    }

private:
    /// The current MPI rank.
#if SYCL_TARGET != 0
[[maybe_unused]]
#endif
    const int comm_rank_;
    /// Map containing all provided command line arguments.
    std::map<std::string, std::string> argvs_;
};

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP
