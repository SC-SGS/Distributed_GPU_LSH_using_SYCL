/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/options.hpp"

#include "sycl_lsh/constants.hpp"                           // sycl_lsh::real_type, sycl_lsh::index_type, sycl_lsh::hash_value_type
#include "sycl_lsh/detail/arithmetic_type_name.hpp"         // sycl_lsh::detail::arithmetic_type_name
#include "sycl_lsh/detail/assert.hpp"                       // SYCL_LSH_ASSERT
#include "sycl_lsh/exceptions/exceptions.hpp"               // sycl_lhs::cmd_parser_exit
#include "sycl_lsh/hash_functions/hash_function_types.hpp"  // sycl_lsh::hash_function_type
#include "sycl_lsh/mpi/communicator.hpp"                    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/file_parser/file_parser_types.hpp"   // sycl_lsh::mpi::file_parser
#include "sycl_lsh/mpi/logger.hpp"                          // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"                           // sycl_lsh::timer

#include "cxxopts.hpp"   // cxxopts::Options, cxxopts::ParseResult
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <exception>  // std::exception
#include <optional>   // std::optional
#include <ostream>    // std::ostream
#include <string>     // std::string

namespace sycl_lsh {

options::options(const int argc, char **argv, const mpi::logger &logger) {
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
            ("file_parser", "the type of the file parser: \n\t0: binary\n\t1: arff", cxxopts::value<mpi::file_parser_type>()->default_value("binary"))
            ("knn_save_file", "the file to which the calculated nearest-neighbors should be saved to", cxxopts::value<std::string>())
            ("knn_dist_save_file", "the file to which the calculated nearest-neighbors distances should be saved to", cxxopts::value<std::string>())
            ("evaluate_knn_file", "the file containing the correct nearest-neighbors for calculating the resulting recall", cxxopts::value<std::string>())
            ("evaluate_knn_dist_file", "the file containing the correct nearest-neighbors distances for calculating the resulting recall", cxxopts::value<std::string>())
            // locality sensitive hashing specific options
            ("hash_function", "the type of the hash functions: \n\t0: random-projections\n\t1: entropy-based\n\t2: mixed", cxxopts::value<hash_function_type>()->default_value("random-projections"))
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
        // output help message only on master MPI rank
        logger.log("{}\n{}\n", e.what(), options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // print help message and exit
    if (result.contains("help")) {
        // output help message only on master MPI rank
        logger.log("{}\n", options.help());
        throw cmd_parser_exit{ EXIT_SUCCESS };
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        // output error message only on master MPI rank
        logger.log("ERROR: only one positional options may be given, but {} (\"{}\") additional option(s) where provided!\n{}\n", result.unmatched().size(), fmt::join(result.unmatched(), " "), options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // parse command line variables

    if (!result.contains("knn")) {
        // output error message only on master MPI rank
        logger.log("ERROR: missing nearest-neighbors number!\n\n{}\n", options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    k = result["knn"].as<index_type>();

    if (!result.contains("file")) {
        // output error message only on master MPI rank
        logger.log("ERROR: missing input file!\n\n{}\n", options.help());
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    data_file = result["file"].as<std::string>();

    file_parser = result["file_parser"].as<mpi::file_parser_type>();

    if (result.contains("knn_save_file")) {
        knn_save_file = result["knn_save_file"].as<std::string>();
    }
    if (result.contains("knn_dist_save_file")) {
        knn_dist_save_file = result["knn_dist_save_file"].as<std::string>();
    }
    if (result.contains("evaluate_knn_file")) {
        evaluate_knn_file = result["evaluate_knn_file"].as<std::string>();
    }
    if (result.contains("evaluate_knn_dist_file")) {
        evaluate_knn_dist_file = result["evaluate_knn_dist_file"].as<std::string>();
    }

    lsh_options.hash_function = result["hash_function"].as<hash_function_type>();
    lsh_options.hash_pool_size = result["hash_pool_size"].as<index_type>();
    lsh_options.num_hash_functions = result["num_hash_functions"].as<index_type>();
    lsh_options.num_hash_tables = result["num_hash_tables"].as<index_type>();
    lsh_options.hash_table_size = result["hash_table_size"].as<index_type>();
    lsh_options.w = result["w"].as<real_type>();
    lsh_options.num_cut_off_points = result["num_cut_off_points"].as<index_type>();

    // perform some sanity checks
    SYCL_LSH_ASSERT(lsh_options.hash_pool_size > 0, "Invalid hash_pool_size!");
    SYCL_LSH_ASSERT(lsh_options.num_hash_functions > 0, "Invalid num_hash_functions!");
    SYCL_LSH_ASSERT(lsh_options.num_hash_tables > 0, "Invalid num_hash_tables!");
    SYCL_LSH_ASSERT(lsh_options.hash_table_size > 0, "Invalid hash_table_size!");
    SYCL_LSH_ASSERT(lsh_options.w > 0.0, "Invalid w!");
    SYCL_LSH_ASSERT(lsh_options.num_cut_off_points > 0, "Invalid num_cut_off_points!");
}

void options::save_benchmark_options([[maybe_unused]] const mpi::communicator &comm) const {
#if defined(SYCL_LSH_BENCHMARK)
    if (comm.is_main_rank()) {
        mpi::timer::benchmark_out() << lsh_options.hash_pool_size << ','
                                    << lsh_options.num_hash_functions << ','
                                    << lsh_options.num_hash_tables << ','
                                    << lsh_options.hash_table_size << ','
                                    << lsh_options.w << ','
                                    << lsh_options.num_cut_off_points << '\n';
    }
#endif
}

std::ostream &operator<<(std::ostream &out, const options &opt) {
    // compile time constants and options
    std::string str = fmt::format("real_type: \"{} ({} byte)\"\n"
                                  "index_type: \"{} ({} byte)\"\n"
                                  "hash_value_type: \"{} ({} byte)\"\n"
                                  "blocking_size: {}\n"
                                  "k: {}\n"
                                  "file: \"{}\"\n"
                                  "file_parser: \"{}\"\n",
                                  detail::arithmetic_type_name<real_type>(),
                                  sizeof(real_type),
                                  detail::arithmetic_type_name<index_type>(),
                                  sizeof(index_type),
                                  detail::arithmetic_type_name<hash_value_type>(),
                                  sizeof(hash_value_type),
                                  BLOCKING_SIZE,
                                  opt.k,
                                  opt.data_file,
                                  opt.file_parser);

    if (opt.knn_save_file.has_value()) {
        str += fmt::format("knn_save_file: \"{}\"\n", opt.knn_save_file.value());
    }
    if (opt.knn_dist_save_file.has_value()) {
        str += fmt::format("knn_dist_save_file: \"{}\"\n", opt.knn_dist_save_file.value());
    }
    if (opt.evaluate_knn_file.has_value()) {
        str += fmt::format("evaluate_knn_file: \"{}\"\n", opt.evaluate_knn_file.value());
    }
    if (opt.evaluate_knn_dist_file.has_value()) {
        str += fmt::format("evaluate_knn_dist_file: \"{}\"\n", opt.evaluate_knn_dist_file.value());
    }

    str += fmt::format("hash_functions_type: \"{}\"\n"
                       "hash_pool_size: {}\n"
                       "num_hash_functions: {}\n"
                       "num_hash_tables: {}\n"
                       "hash_table_size: {}\n",
                       opt.lsh_options.hash_function,
                       opt.lsh_options.hash_pool_size,
                       opt.lsh_options.num_hash_functions,
                       opt.lsh_options.num_hash_tables,
                       opt.lsh_options.hash_table_size);

    if (opt.lsh_options.hash_function != hash_function_type::entropy_based) {
        str += fmt::format("w: {}\n", opt.lsh_options.w);
    }
    if (opt.lsh_options.hash_function != hash_function_type::random_projections) {
        str += fmt::format("num_cut_off_points: {}\n", opt.lsh_options.num_cut_off_points);
    }
    return out << str;
}

}  // namespace sycl_lsh
