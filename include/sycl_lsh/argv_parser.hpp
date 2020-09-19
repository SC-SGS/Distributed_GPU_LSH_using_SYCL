/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-19
 *
 * @brief Implements a very simple command line argument parser specifically for this project.
 */
 
#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP

#include <sycl_lsh/detail/conversion.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace sycl_lsh {

    /**
     * @brief Minimalistic class to parse command line arguments.
     * @details The supported command line options are:
     * | command line argument | description                                                                                           |
     * |:----------------------|:------------------------------------------------------------------------------------------------------|
     * | help                  | Prints the help screen.                                                                               |
     * | data                  | Path to the data file (**required**).                                                                 |
     * | k                     | The number of nearest-neighbors to search for (**required**).                                         |
     * | options               | Path to the options file to load.                                                                     |
     * | save_options          | Path to the file to save the currently used options to.                                               |
     * | save_knn              | Path to the file to save the found k-nearest-neighbors to.                                            |
     * | evaluate_knn          | Path to the file containing the correct k-nearest-neighbors.                                          |
     * | hash_pool_size        | The number of hash functions in the hash pool.                                                        |
     * | num_hash_functions    | The number of hash functions to calculate the hash values with.                                       |
     * | num_hash_tables       | The number of used hash tables.                                                                       |
     * | hash_table_size       | The size of each hash table.                                                                          |
     * | w                     | The segment size for the random projections hash functions: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$. |
     * | num_cut_off_points    | The number of cut-off points for the entropy-based hash functions.                                    |
     */
    class argv_parser {
    public:
        /**
         * @brief Parse the given command line arguments.
         * @param[in] argc the number of command line arguments
         * @param[in] argv the command line arguments
         *
         * @throws std::invalid_argument if **any** command line argument key doesn't start with '--'.
         * @throws std::invalid_argument if **any** command line argument key is illegal.
         * @throws std::invalid_argument if **any** command line argument key has been provided more than once.
         * @throws std::invalid_argument if **any** command line argument value hasn't been provided.
         * @throws std::invalid_argument if **any** command line argument value starts with '--'.
         * @throws std::logic_error if **any** required command line argument hasn't been provided.
         */
        argv_parser(const int argc, char** argv);

        /**
         * @brief Check whether the command line argument @p key has been specified.
         * @param[in] key the command line argument to check for
         * @return `true` if @p key has been provided, `false` otherwise (`[[nodiscard]]`)
         */
        [[nodiscard]]
        bool has_argv(const std::string& key) const;

        /**
         * @brief Returns the value associated with @p key converted to the type `T`.
         * @tparam T the returned type (**must** be either a [arithemtic type](https://en.cppreference.com/w/cpp/types/is_arithmetic) or
         *           [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string)).
         * @param[in] key the command line argument to get the value for
         * @return the value associated with @p key converted to the type `T` (`[[nodiscard]]`)
         *
         * @throws std::invalid_argument if @p key is an illegal command line argument key.
         * @throws std::invalid_argument if @p key hasn't been provided.
         */
        template <typename T>
        [[nodiscard]]
        T argv_as(const std::string& key) const {
            // check whether the key is legal
            if (list_of_argvs_.count(key) == 0) {
                throw std::invalid_argument("The requested command line argument key '" + key + "' is illegal!");
            }
            // check whether the key has been provided
            if (!this->has_argv(key)) {
                throw std::invalid_argument("The requested command line argument key '" + key + "' hasn't been provided!");
            }

            // convert the value to the given type T
            if (std::is_same_v<T, std::string>) {
                return argvs_.at(key);
            } else {
                return detail::convert_to<T>(argvs_.at(key));
            }
        }

        /**
         * @brief Returns a description of all command line arguments.
         * @return the description (`[[nodiscard]]`)
         */
        [[nodiscard]]
        std::string description() const;

    private:
        /// List of all possible command line options ({ KEY, { DESCRIPTION, IS_REQUIRED } })
        static const std::map<std::string, std::pair<std::string, bool>> list_of_argvs_;
        std::map<std::string, std::string> argvs_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARGV_PARSER_HPP
