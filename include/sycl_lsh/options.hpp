/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-28
 *
 * @brief Implements a @ref sycl_lsh::options class for managing hyperparameters.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_OPTIONS_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_OPTIONS_HPP

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/detail/arithmetic_type_name.hpp>
#include <sycl_lsh/detail/assert.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/filesystem.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/logger.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * @def SYCL_LSH_PARSE_OPTION
 * @brief Defines a macro to parse the @p option using the @ref sycl_lsh::argv_parser @p parser and assign it to the correct struct field.
 * @param[in] parser the @p sycl_lsh::argv_parser
 * @param[in] option the option to parse and assign
 * @param[in] sanity_cond sanity check for the options value @p option
 *
 * @throws std::invalid_argument if the sanity check fails.
 */
#define SYCL_LSH_PARSE_OPTION(parser, option, sanity_cond)                                                                              \
if (parser.has_argv(#option)) {                                                                                                         \
    option = parser.argv_as<decltype(option)>(#option);                                                                                 \
}                                                                                                                                       \
if (!(sanity_cond)) {                                                                                                                   \
    throw std::invalid_argument(fmt::format("Illegal {} value ({})! Legal values must fulfill: '{}'.", #option, option, #sanity_cond)); \
}

namespace sycl_lsh {

    /**
     * @brief Class containing and managing all compile time and runtime hyperparameters to change the behavior of the LSH algorithm.
     * @tparam real_t a floating point type
     * @tparam index_t an integral type (used for indices)
     * @tparam hash_value_t an integral type (use for hash values)
     */
    template <typename real_t, typename index_t, typename hash_value_t>
    struct options : private detail::options_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_floating_point_v<real_t>, "The first template parameter must be a floating point type!");
        static_assert(std::is_integral_v<index_t>, "The second template parameter must be an integral type!");
        static_assert(std::is_integral_v<hash_value_t>, "The third template parameter must be an integral type!");


        // ---------------------------------------------------------------------------------------------------------- //
        //                                            compile time options                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        /// The used floating point type.
        using real_type = real_t;
        /// The used integral type for indices.
        using index_type = index_t;
        /// The used integral type for hash values.
        using hash_value_type = hash_value_t;

        /// The blocking size used in the SYCL kernels.
        static constexpr index_type blocking_size = 10;
        // TODO: type of the hash functions


        // ---------------------------------------------------------------------------------------------------------- //
        //                                              runtime options                                               //
        // ---------------------------------------------------------------------------------------------------------- //
        // TODO 2020-09-22 17:23 marcel: set meaningful defaults
        
        /// The number of hash functions in the hash pool.
        index_type hash_pool_size = 10;
        /// The number of hash functions per hash table.
        index_type num_hash_functions = 4;
        /// The number of used hash tables. 
        index_type num_hash_tables = 2;
        /// The size of each hash table.
        hash_value_type hash_table_size = 105619;
        /// The segment size for the random projections hash functions: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$.
        real_type w = 1.0;
        /// The number of cut-off points for the entropy-based hash functions.
        index_type num_cut_off_points = 4;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructors                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Default construct a @ref sycl_lsh::options class.
         */
        options() noexcept = default;
        /**
         * @brief Construct a @ref sycl_lsh::options class using the command line parser @p parser.
         * @details If an options file was specified via the command line arguments, reads all options from the given file.
         *          Afterwards overrides all read options by options directly given to the command line via (`--your_opt your_val`). \n
         *          Uses the @ref sycl_lsh::mpi::logger @p logger to log additional information.
         * @param[in] parser the @ref sycl_lsh::argv_parser
         * @param[in] logger the @ref sycl_lsh::mpi::logger
         *
         * @throws std::invalid_argument if the file specified by the command line argument `options_file` doesn't exist or isn't a
         *         regular file.
         * @throws std::invalid_argument if any command line argument in the file is illegal.
         * @throws std::invalid_argument if any parsed value is illegal.
         */
        explicit options(const argv_parser& parser, const mpi::logger& logger);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                save options                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Saves the currently set compile time and runtime options only on the MPI master rank to the file parsed from the command
         *        line arguments @ref sycl_lsh::argv_parser @p parser via the command line argument `options_save_file`. \n
         *          Uses the @ref sycl_lsh::mpi::logger @p logger to log additional information.
         * @param[in] comm the @ref sycl_lsh::mpi::communicator
         * @param[in] parser the @ref sycl_lsh::argv_parser
         * @param[in] logger the @ref sycl_lsh::mpi::logger
         *
         * @throws std::invalid_argument if the command line argument `options_save_file` isn't present in @p parser.
         * @throws std::runtime_error if the file couldn't be written
         */
        void save(const mpi::communicator& comm, const argv_parser& parser, const mpi::logger& logger) const;
        /**
         * @brief Saves the currently set compile time and runtime options only on the current MPI master rank to the @p file.\n
         *        Uses the @ref sycl_lsh::mpi::logger @p logger to log additional information.
         * @param[in] comm the @ref sycl_lsh::mpi::communicator
         * @param[in] file the file
         * @param[in] logger the @ref sycl_lsh::mpi::logger
         *
         * @throws std::runtime_error if the @p file couldn't be written
         */
        void save(const mpi::communicator& comm, const std::string& file, const mpi::logger& logger) const;

    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                            output stream overload                                          //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Print all options (compile time and runtime) set in @p opt to the output stream @p out.
     * @tparam real_t a floating point type
     * @tparam index_t an integral type (used for indices)
     * @tparam hash_value_t an integral type (used for hash values)
     * @param[in,out] out the output stream
     * @param[in] opt the @ref sycl_lsh::options
     * @return the output stream
     */
    template <typename real_t, typename index_t, typename hash_value_t>
    std::ostream& operator<<(std::ostream& out, const options<real_t, index_t, hash_value_t>& opt) {
        // get types
        using options_type = options<real_t, index_t, hash_value_t>;
        using real_type = typename options_type::real_type;
        using index_type = typename options_type::index_type;
        using hash_value_type = typename options_type::hash_value_type;

        // compile time options
        out << fmt::format("real_type '{}' ({} byte)\n", detail::arithmetic_type_name<real_type>(), sizeof(real_type));
        out << fmt::format("index_type '{}' ({} byte)\n", detail::arithmetic_type_name<index_type>(), sizeof(index_type));
        out << fmt::format("hash_value_type '{}' ({} byte)\n", detail::arithmetic_type_name<hash_value_type>(), sizeof(hash_value_type));
        out << fmt::format("blocking_size {}\n\n", options_type::blocking_size);

        // runtime options
        // TODO 2020-09-22 18:02 marcel: only print values used in the currently specified hash functions type
        out << fmt::format("hash_pool_size {}\n", opt.hash_pool_size);
        out << fmt::format("num_hash_functions {}\n", opt.num_hash_functions);
        out << fmt::format("num_hash_tables {}\n", opt.num_hash_tables);
        out << fmt::format("hash_table_size {}\n", opt.hash_table_size);
        out << fmt::format("w {}\n", opt.w);
        out << fmt::format("num_cut_off_points {}\n", opt.num_cut_off_points);

        return out;
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename real_t, typename index_t, typename hash_value_t>
    options<real_t, index_t, hash_value_t>::options(const argv_parser& parser, const mpi::logger& logger) {
        // parse command line options given through the (optionally) specified file
        if (parser.has_argv("options_file")) {
            const std::string& file = parser.argv_as<std::string>("options_file");
            // check if file exists and is a regular file
            if (!fs::exists(file) || !fs::is_regular_file(file)) {
                throw std::invalid_argument(fmt::format("Illegal options file '{}'!", file));
            }

            logger.log("Reading options from file: '{}'\n\n", file);

            // parse file
            std::ifstream in(file);
            std::string line;

            int lineno = 0;
            while (std::getline(in, line)) {
                ++lineno;

                // ignore empty lines
                if (line.empty()) {
                    continue;
                }

                const std::size_t pos = line.find_first_of(' ');

                // a line without any whitespace is illegal
                if (pos == std::string::npos) {
                    throw std::invalid_argument(fmt::format("Illegal line ({}) '{}' in file '{}'!", lineno, line, file));
                }

                // parse option key and respective value
                const std::string opt = line.substr(0, pos);
                const std::string value = line.substr(pos + 1, line.size());

                if (opt == "real_type" || opt == "index_type" || opt == "hash_value_type" || opt == "blocking_size") {
                    // can't read compile time options from file
                    continue;
                } else if (opt == "hash_pool_size") {
                    hash_pool_size = detail::convert_to<decltype(hash_pool_size)>(value);
                } else if (opt == "num_hash_functions") {
                    num_hash_functions = detail::convert_to<decltype(num_hash_functions)>(value);
                } else if (opt == "num_hash_tables") {
                    num_hash_tables = detail::convert_to<decltype(num_hash_tables)>(value);
                } else if (opt == "hash_table_size") {
                    hash_table_size = detail::convert_to<decltype(hash_table_size)>(value);
                } else if (opt == "w") {
                    w = detail::convert_to<decltype(w)>(value);
                } else if (opt == "num_cut_off_points") {
                    num_cut_off_points = detail::convert_to<decltype(num_cut_off_points)>(value);
                } else {
                    // option not recognized
                    throw std::invalid_argument(fmt::format("Invalid option in line {} '{} {}' in file '{}'!", lineno, opt, value, file));
                }
            }
        }

        // TODO 2020-09-22 17:49 marcel: check whether loaded save file has been saved with the same hash function type

        // parse command line options given directly through the command line arguments and perform sanity checks
        SYCL_LSH_PARSE_OPTION(parser, hash_pool_size,     hash_pool_size > 0);
        SYCL_LSH_PARSE_OPTION(parser, num_hash_functions, num_hash_functions > 0);
        SYCL_LSH_PARSE_OPTION(parser, num_hash_tables,    num_hash_tables > 0);
        SYCL_LSH_PARSE_OPTION(parser, hash_table_size,    hash_table_size > 0);
        SYCL_LSH_PARSE_OPTION(parser, w,                  w > 0);
        SYCL_LSH_PARSE_OPTION(parser, num_cut_off_points, num_cut_off_points > 0);
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                save options                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename real_t, typename index_t, typename hash_value_t>
    void options<real_t, index_t, hash_value_t>::save(const mpi::communicator& comm, const argv_parser& parser, const mpi::logger& logger) const {
        if (comm.master_rank()) {
            if (!parser.has_argv("options_save_file")) {
                throw std::invalid_argument("Required command line argument 'options_save_file' not provided!");
            }
            
            save(comm, parser.argv_as<std::string>("options_save_file"), logger);
        }
    }

    template <typename real_t, typename index_t, typename hash_value_t>
    void options<real_t, index_t, hash_value_t>::save(const mpi::communicator& comm, const std::string& file, const mpi::logger& logger) const {
        if (comm.master_rank()) {
            std::ofstream out(file, std::ofstream::trunc);
            if (out.bad()) {
                // something went wrong while opening/creating the file
                throw std::runtime_error(fmt::format("Can't write to file '{}'!", file));
            }
            out << *this << std::endl;
        }

        logger.log("Saved options to: '{}'\n\n", file);
    }

#undef SYCL_LSH_PARSE_OPTION

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_OPTIONS_HPP
