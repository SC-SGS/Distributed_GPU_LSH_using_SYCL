/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief The main file containing the main logic.
 */

#include "sycl_lsh/core.hpp"
#include "sycl_lsh/mpi/file_parser/file_parser.hpp"

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
        sycl_lsh::data_set data{ opt, comm, logger };
        logger.log("\nUsed data set:\n{}\n", data);

        // create a nearest-neighbors object performing the unsupervised learning task
        sycl_lsh::nearest_neighbors nn{ opt.k, opt.lsh_options, queue, comm, logger };

        // fit the data
        nn.fit(data);

        // calculate the nearest-neighbors
        const auto [indices, distances] = nn.kneighbors(sycl_lsh::n_neighbors = opt.k, sycl_lsh::return_distance = true);

        // TODO: better hide behind some better API?

        // optionally save calculated k-nearest-neighbor IDs
        if (opt.knn_save_file.has_value()) {
            const auto parser = sycl_lsh::mpi::make_file_parser<sycl_lsh::index_type>(opt.knn_save_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::file::mode::write, comm, logger);
            parser->write_content(data.get_attributes().total_size, opt.k, indices);
        }
        // optionally save calculated k-nearest-neighbor distances
        if (opt.knn_dist_save_file.has_value() && distances.has_value()) {
            const auto parser = sycl_lsh::mpi::make_file_parser<sycl_lsh::real_type>(opt.knn_dist_save_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::file::mode::write, comm, logger);
            parser->write_content(data.get_attributes().total_size, opt.k, distances.value());
        }

        // optionally calculate the recall of the calculated k-nearest-neighbors
        if (opt.evaluate_knn_file.has_value()) {
            const auto parser = sycl_lsh::mpi::make_file_parser<sycl_lsh::index_type>(opt.evaluate_knn_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::file::mode::read, comm, logger);
            const sycl_lsh::aos_matrix<sycl_lsh::index_type> correct_indices = parser->parse_content();
            logger.log("recall: {}%\n", sycl_lsh::report::recall(indices, correct_indices, comm, logger));
        }
        // optionally calculate the error ration of the calculated k-nearest-neighbors
        if (opt.evaluate_knn_dist_file.has_value() && distances.has_value()) {
            const auto parser = sycl_lsh::mpi::make_file_parser<sycl_lsh::real_type>(opt.evaluate_knn_dist_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::file::mode::read, comm, logger);
            const sycl_lsh::aos_matrix<sycl_lsh::real_type> correct_distances = parser->parse_content();
            const auto [error_ratio, num_points, num_knn_not_found] = sycl_lsh::report::error_ratio(distances.value(), correct_distances, comm, logger);
            if (num_points == 0) {
                logger.log("error ratio: {}\n", error_ratio);
            } else {
                logger.log("error ratio: {} (for {} points a total of {} nearest-neighbors couldn't be found)\n", error_ratio, num_points, num_knn_not_found);
            }
        }

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
