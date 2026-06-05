/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/options.hpp"

#include "sycl_lsh/constants.hpp"                    // sycl_lsh::real_type, sycl_lsh::index_type, sycl_lsh::hash_value_type, sycl_lsh::BLOCKING_SIZE
#include "sycl_lsh/detail/arithmetic_type_name.hpp"  // sycl_lsh::detail::arithmetic_type_name
#include "sycl_lsh/detail/assert.hpp"                // SYCL_LSH_ASSERT
#include "sycl_lsh/exceptions/exceptions.hpp"        // sycl_lhs::cmd_parser_exit, sycl_lsh::invalid_lsh_option_exception
#include "sycl_lsh/hash_function_types.hpp"          // sycl_lsh::hash_function_type
#include "sycl_lsh/mpi/communicator.hpp"             // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/logging.hpp"           // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/file_parser_types.hpp"        // sycl_lsh::mpi::file_parser

#include "cxxopts.hpp"   // cxxopts::Options, cxxopts::ParseResult
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <exception>  // std::exception
#include <optional>   // std::optional
#include <ostream>    // std::ostream
#include <string>     // std::string

namespace sycl_lsh {

namespace detail {

void sanity_check_locality_sensitive_hashing_options(const locality_sensitive_hashing_options &options) {
    if (options.hash_pool_size <= 0) {
        throw invalid_lsh_option_exception{ fmt::format("Invalid 'hash_pool_size'! Must be larger than 0 but is {}.", options.hash_pool_size) };
    }
    if (options.num_hash_functions <= 0) {
        throw invalid_lsh_option_exception{ fmt::format("Invalid 'num_hash_functions'! Must be larger than 0 but is {}.", options.num_hash_functions) };
    }
    if (options.num_hash_tables <= 0) {
        throw invalid_lsh_option_exception{ fmt::format("Invalid 'num_hash_tables'! Must be larger than 0 but is {}.", options.num_hash_tables) };
    }
    if (options.hash_table_size <= 0) {
        throw invalid_lsh_option_exception{ fmt::format("Invalid 'hash_table_size'! Must be larger than 0 but is {}.", options.hash_table_size) };
    }
    if (options.w <= 0) {
        throw invalid_lsh_option_exception{ fmt::format("Invalid 'w'! Must be larger than 0 but is {}.", options.w) };
    }
    if (options.num_cut_off_points <= 0) {
        throw invalid_lsh_option_exception{ fmt::format("Invalid 'num_cut_off_points'! Must be larger than 0 but is {}.", options.num_cut_off_points) };
    }
}

}  // namespace detail

options::options(const mpi::communicator &comm, int &argc, char **&argv) {
    // create command line options parser
    cxxopts::Options options(argv[0], "k-nearest-neighbors using Locality Sensitive Hashing and SYCL");
    options
        .positional_help("input_file k")
        .show_positional_help();
    options.set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("h,help", "print this helper message", cxxopts::value<bool>())
            ("file_parser", "the type of the file parser: \n\t0: binary\n\t1: arff", cxxopts::value<mpi::file_parser_type>()->default_value(fmt::format("{}", mpi::file_parser_type::binary)))
            ("profiling_type", "the profiling capabilities: \n\t0: none\n\t1: runtimes\n\t2: hws", cxxopts::value<profiling_types>()->default_value(fmt::format("{}", profiling_types::none)))
            ("profiling_file", "the output file to write the profiling results to (YAML format)", cxxopts::value<std::string>())
            ("indices_save_file", "the file to which the calculated nearest-neighbors should be saved to", cxxopts::value<std::string>())
            ("distances_save_file", "the file to which the calculated nearest-neighbors distances should be saved to", cxxopts::value<std::string>())
            ("indices_ground_truth_file", "the file containing the correct nearest-neighbors for calculating the resulting recall", cxxopts::value<std::string>())
            ("distances_ground_truth_file", "the file containing the correct nearest-neighbors distances for calculating the resulting recall", cxxopts::value<std::string>())
            // locality sensitive hashing specific options
            ("hash_function", "the type of the hash functions: \n\t0: random-projections\n\t1: entropy-based\n\t2: mixed", cxxopts::value<hash_function_type>()->default_value(fmt::format("{}", hash_function_type::random_projections)))
            ("hash_pool_size", "the number of hash functions in the hash pool", cxxopts::value<index_type>()->default_value("32"))
            ("num_hash_functions", "the number of hash functions per hash table", cxxopts::value<index_type>()->default_value("12"))
            ("num_hash_tables", "the number of used hash tables", cxxopts::value<index_type>()->default_value("8"))
            ("hash_table_size", "the size of each hash table", cxxopts::value<index_type>()->default_value("105613"))
            ("w", "the segment size for the random projections hash functions", cxxopts::value<real_type>()->default_value("1.0"))
            ("num_cut_off_points", "the number of cut-off points for the entropy-based hash functions", cxxopts::value<index_type>()->default_value("6"))
            // positional options
            ("file", "the input data file", cxxopts::value<std::string>(), "input_file")
            ("knn", "the number of nearest-neighbors to calculate", cxxopts::value<index_type>(), "knn");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "file", "knn" });
        result = options.parse(argc, argv);
    } catch (const std::exception &e) {
        // output help message only on the MPI main rank
        mpi::detail::log(comm, "{}\n{}\n", e.what(), options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // print help message and exit
    if (result.contains("help")) {
        // output help message only on the MPI main rank
        mpi::detail::log(comm, "{}\n", options.help());
        throw cmd_parser_exit{ EXIT_SUCCESS };
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        // output error message only on the MPI main rank
        mpi::detail::log(comm, "ERROR: only one positional options may be given, but {} (\"{}\") additional option(s) where provided!\n{}\n", result.unmatched().size(), fmt::join(result.unmatched(), " "), options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // parse command line variables

    if (!result.contains("knn")) {
        // output error message only on the MPI main rank
        mpi::detail::log(comm, "ERROR: missing nearest-neighbors number!\n\n{}\n", options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    n_neighbors = result["knn"].as<index_type>();

    if (!result.contains("file")) {
        // output error message only on the MPI main rank
        mpi::detail::log(comm, "ERROR: missing input file!\n\n{}\n", options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    data_file = result["file"].as<std::string>();

    file_parser = result["file_parser"].as<mpi::file_parser_type>();

    profiling_type = result["profiling_type"].as<profiling_types>();
    if (result.contains("profiling_file")) {
        profiling_file = result["profiling_file"].as<std::string>();
    }

    if (result.contains("indices_save_file")) {
        indices_save_file = result["indices_save_file"].as<std::string>();
    }
    if (result.contains("distances_save_file")) {
        distances_save_file = result["distances_save_file"].as<std::string>();
    }
    if (result.contains("indices_ground_truth_file")) {
        indices_ground_truth_file = result["indices_ground_truth_file"].as<std::string>();
    }
    if (result.contains("distances_ground_truth_file")) {
        distances_ground_truth_file = result["distances_ground_truth_file"].as<std::string>();
    }

    lsh_options.hash_function = result["hash_function"].as<hash_function_type>();
    lsh_options.hash_pool_size = result["hash_pool_size"].as<index_type>();
    lsh_options.num_hash_functions = result["num_hash_functions"].as<index_type>();
    lsh_options.num_hash_tables = result["num_hash_tables"].as<index_type>();
    lsh_options.hash_table_size = result["hash_table_size"].as<index_type>();
    lsh_options.w = result["w"].as<real_type>();
    lsh_options.num_cut_off_points = result["num_cut_off_points"].as<index_type>();

    // perform some sanity checks
    detail::sanity_check_locality_sensitive_hashing_options(lsh_options);
}

std::ostream &operator<<(std::ostream &out, const locality_sensitive_hashing_options &opt) {
    std::string str = fmt::format("hash_functions_type: \"{}\"\n"
                                  "hash_pool_size: {}\n"
                                  "num_hash_functions: {}\n"
                                  "num_hash_tables: {}\n"
                                  "hash_table_size: {}\n",
                                  opt.hash_function,
                                  opt.hash_pool_size,
                                  opt.num_hash_functions,
                                  opt.num_hash_tables,
                                  opt.hash_table_size);

    if (opt.hash_function != hash_function_type::entropy_based) {
        str += fmt::format("w: {}\n", opt.w);
    }
    if (opt.hash_function != hash_function_type::random_projections) {
        str += fmt::format("num_cut_off_points: {}\n", opt.num_cut_off_points);
    }
    return out << str;
}

std::ostream &operator<<(std::ostream &out, const options &opt) {
    // compile time constants and options
    std::string str = fmt::format("n_neighbors: {}\n"
                                  "real_type: {} ({} byte)\n"
                                  "index_type: {} ({} byte)\n"
                                  "hash_value_type: {} ({} byte)\n"
                                  "BLOCKING_SIZE: {}\n"
                                  "profiling_type: {}\n"
                                  "input file (data set): '{}'\n",
                                  opt.n_neighbors,
                                  detail::arithmetic_type_name<real_type>(),
                                  sizeof(real_type),
                                  detail::arithmetic_type_name<index_type>(),
                                  sizeof(index_type),
                                  detail::arithmetic_type_name<hash_value_type>(),
                                  sizeof(hash_value_type),
                                  BLOCKING_SIZE,
                                  opt.profiling_type,
                                  opt.data_file);

    if (opt.profiling_file.has_value()) {
        str += fmt::format("profiling file: '{}'\n", opt.profiling_file.value());
    }
    if (opt.indices_save_file.has_value()) {
        str += fmt::format("output file (indices): '{}'\n", opt.indices_save_file.value());
    }
    if (opt.distances_save_file.has_value()) {
        str += fmt::format("output file (distances): '{}'\n", opt.distances_save_file.value());
    }
    if (opt.indices_ground_truth_file.has_value()) {
        str += fmt::format("input file (indices ground truth): '{}'\n", opt.indices_ground_truth_file.value());
    }
    if (opt.distances_ground_truth_file.has_value()) {
        str += fmt::format("input file (distances ground truth): '{}'\n", opt.distances_ground_truth_file.value());
    }
    return out << opt.lsh_options << str;
}

}  // namespace sycl_lsh
