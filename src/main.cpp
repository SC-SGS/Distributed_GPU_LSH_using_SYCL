/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-08
 *
 * @brief The main file containing the main logic.
 */

#include <sycl_lsh/core.hpp>

struct sycl_test {};

int custom_main(int argc, char** argv) {
    // create MPI communicator
    sycl_lsh::mpi::communicator comm;
    // optionally: set exception handler for the communicator
    sycl_lsh::mpi::errhandler handler(sycl_lsh::mpi::errhandler::type::comm);
    comm.attach_errhandler(handler);

    // create default logger (logs to std::cout)
    sycl_lsh::mpi::logger logger(comm);

    try {

        // parse command line arguments
        sycl_lsh::argv_parser parser(argc, argv);
        // log help message if requested
        if (parser.has_argv("help")) {
            logger.log(sycl_lsh::argv_parser::description());
            return EXIT_SUCCESS;
        }

        // log current number of MPI ranks
        logger.log("MPI_Comm_size: {}\n\n", comm.size());

        // parse options and print
        const sycl_lsh::options<float, std::uint32_t, std::uint32_t, 10, sycl_lsh::hash_functions_type::random_projections> opt(parser, logger);
        logger.log("Used options: \n{}\n", opt);

        // optionally save generated options to file
        if (parser.has_argv("options_save_file")) {
            opt.save(parser, comm, logger);
        }

        // parse data and print data attributes
        auto data = sycl_lsh::make_data<sycl_lsh::memory_layout::aos>(parser, opt, comm, logger);
        logger.log("\nUsed data set:\n{}\n", data);

        // generate LSH hash tables
        auto lsh_tables = sycl_lsh::make_hash_tables<sycl_lsh::memory_layout::aos>(opt, data, comm, logger);
        // calculate k-nearest-neighbors
        auto knns = lsh_tables.get_k_nearest_neighbors(parser);

        // optionally save calculated k-nearest-neighbor IDs
        if (parser.has_argv("knn_save_file")) {
            knns.save_knns(parser);
        }
        // optionally save calculated k-nearest-neighbor distances
        if (parser.has_argv("knn_dist_save_file")) {
            knns.save_distances(parser);
        }

        // optionally calculate the recall of the calculated k-nearest-neighbors
        if (parser.has_argv("evaluate_knn_file")) {
            logger.log("recall: {} %\n", knns.recall(parser));
        }
        // optionally calculate the error ration of the calculated k-nearest-neighbors
        if (parser.has_argv("evaluate_knn_dist_file")) {
            const auto [error_ratio, num_points, num_knn_not_found] = knns.error_ratio(parser);
            if (num_points == 0) {
                logger.log("error ratio: {}\n", error_ratio);
            } else {
                logger.log("error ratio: {} (for {} points a total of {} nearest-neighbors couldn't be found)\n", error_ratio, num_points, num_knn_not_found);
            }
        }

        // if benchmarking is enabled, also output the used options to the benchmark file (as last entry)
        opt.save_benchmark_options(comm);
        
    } catch (const std::exception& e) {
        logger.log("Exception thrown on rank {}: {}\n", comm.rank(), e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
    return sycl_lsh::mpi::main(argc, argv, &custom_main);
}
