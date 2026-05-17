/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief The main file containing the main logic.
 */

#include "sycl_lsh/core.hpp"

#include <variant>  // std::visit

int custom_main(const int argc, char **argv) {
    // create a SYCL queue
    sycl::queue queue{ sycl_lsh::device_selector };

    // create MPI communicator
    const sycl_lsh::mpi::communicator comm{};

    // create default logger (logs to std::cout)
    const sycl_lsh::mpi::logger logger{ comm };

    try {
        // parse options and print
        const sycl_lsh::options opt(argc, argv, logger);
        logger.log("Used options: \n{}\n", opt);

        // log current number of MPI ranks
        logger.log("MPI_Comm_size: {}\n\n", comm.size());

        // parse data and print data attributes
        auto data = sycl_lsh::make_data<sycl_lsh::memory_layout::aos>(opt, queue, comm, logger);
        logger.log("\nUsed data set:\n{}\n", data);

        // generate LSH hash tables and calculate the nearest-neighbors
        std::visit([&](auto &&lsh_tables) {
            // calculate k-nearest-neighbors
            auto knns = lsh_tables.k_nearest_neighbors(opt.k);

            // optionally save calculated k-nearest-neighbor IDs
            if (opt.knn_save_file.has_value()) {
                knns.save_knns(opt);
            }
            // optionally save calculated k-nearest-neighbor distances
            if (opt.knn_dist_save_file.has_value()) {
                knns.save_distances(opt);
            }

            // optionally calculate the recall of the calculated k-nearest-neighbors
            if (opt.evaluate_knn_file.has_value()) {
                logger.log("recall: {}%\n", knns.recall(opt));
            }
            // optionally calculate the error ration of the calculated k-nearest-neighbors
            if (opt.evaluate_knn_dist_file.has_value()) {
                const auto [error_ratio, num_points, num_knn_not_found] = knns.error_ratio(opt);
                if (num_points == 0) {
                    logger.log("error ratio: {}\n", error_ratio);
                } else {
                    logger.log("error ratio: {} (for {} points a total of {} nearest-neighbors couldn't be found)\n", error_ratio, num_points, num_knn_not_found);
                }
            }
        },
                   sycl_lsh::make_hash_tables<sycl_lsh::memory_layout::aos>(opt, data, queue, comm, logger));

        // if benchmarking is enabled, also output the used options to the benchmark file (as last entry)
        opt.save_benchmark_options(comm);
    } catch (const sycl_lsh::cmd_parser_exit &e) {
        return e.exit_code();
    } catch (const sycl_lsh::exception &e) {
        logger.log("Exception thrown on rank {}: {}\n", comm.rank(), e.what_with_loc());
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        logger.log("Exception thrown on rank {}: {}\n", comm.rank(), e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return sycl_lsh::mpi::main(argc, argv, &custom_main);
}
