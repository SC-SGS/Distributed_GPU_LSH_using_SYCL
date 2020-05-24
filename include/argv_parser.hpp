/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-24
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

#include <detail/convert.hpp>


/**
 * @brief Class to parse command line arguments.
 */
class argv_parser {
    /// A list of all legal command line options including a description.
    const std::map<std::string, std::string> possible_argvs = {
            { "help", "help screen" },
            { "options", "path to options file" },
            { "save_options", "save the currently used options to path" },
            { "data", "path to the data file (required)" },
            { "k", "the number of nearest-neighbours to search for (required)" },
            { "save_knn", "save the calculate nearest-neighbours to path" },
            { "num_hash_tables", "number of hash tables to create" },
            { "hash_table_size", "size of each hash table (must be a prime)" },
            { "num_hash_functions", "number of hash functions per hash table" },
            { "w", "constant used in the hash functions" }
    };

public:
    /**
     * @brief Parse the given command line arguments.
     * @param argc the number of provided command line arguments
     * @param argv the command line arguments
     *
     * @throws std::invalid_argument if any key **doesn't** starts with "--".
     * @throws std::invalid_argument if any key **isn't** a legal key according to @ref possible_argvs.
     * @throws std::invalid_argument if any key has been provided more than once.
     */
    argv_parser(const int argc, char** argv) {
        for(int i = 1; i < argc - 1; ++i) {
            std::string key = argv[i];

            // check whether the key starts with two leading "--"
            if (key.rfind("--", 0) != 0) throw std::invalid_argument("All argv keys must start with '--'!: " + key);
            key.erase(0, 2);

            // check whether the key is legal AND hasn't been provided yet
            if (possible_argvs.count(key) == 0) throw std::invalid_argument("Illegal argv key!: " + key);
            if (argvs.count(key) > 0) throw std::invalid_argument("Duplicate argv keys!: " + key);

            // if hte current key equals "help" continue with parsing the next [key, value]-pair
            // (DON'T read a value because none will be provided!)
            if (key == "help") {
                argvs.emplace(std::move(key), "");
                continue;
            }

            // add [key, value]-pair to parsed command line arguments
            argvs.emplace(std::move(key), argv[++i]);
        }
    }

    /**
     * @brief Check whether the provided command line arguemnts include @p key.
     * @tparam T the type of the key (**must** be convertible to [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string).
     * @param key the key to check for
     * @return `true` if @p key is present, otherwise `false` (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] bool has_argv(T&& key) const {
        return argvs.count(std::forward<T>(key)) > 0;
    }
    /**
     * @brief Returns the value associated with @p key.
     * @tparam T the returned type (**must** be either a [arithemtic type](https://en.cppreference.com/w/cpp/types/is_arithmetic) or
     *           [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string)).
     * @tparam U the type of the key (**must** be convertible to [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string).
     * @param key the @p key to get the value for
     * @return the value associated with @p key converted to the type T (`[[nodiscard]]`)
     *
     * @throws std::invalid_argument if @p key **isn't** a key provided as command line argument.
     */
    template <typename T, typename U>
    [[nodiscard]] T argv_as(U&& key) {
        // check whether key is present
        if (!this->has_argv(key)) {
            throw std::invalid_argument(std::string("The requested key '") + key + "' can't be found!");
        }
        // convert the value to the given type T and return it
        if constexpr(std::is_same_v<std::decay_t<T>, std::string>) {
            return argvs[std::forward<U>(key)];
        } else {
            return detail::convert_to<T>(argvs[std::forward<U>(key)]);
        }
    }
    /**
     * @brief Returns a description of all possible command line arguments.
     * @return the description (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string description() const {
        std::string desc("Usage: ./prog --data \"path-to-data_set\" [options]\n");
        const auto max_reduction = [](const std::size_t value, const auto& key_value_p) { return std::max(value, key_value_p.first.size()); };
        const std::size_t max_size = std::accumulate(possible_argvs.begin(), possible_argvs.end(), 0, max_reduction);
        for (const auto& [key, value] : possible_argvs) {
            desc += "   --" + key + std::string(max_size - key.size() + 2, ' ') + "(" + value + ")\n";
        }
        return desc;
    }

private:
    /// Map containing all provided command line arguments.
    std::map<std::string, std::string> argvs;
};

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP
