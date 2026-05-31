/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief The main file containing the main logic.
 */

#include "sycl_lsh/core.hpp"
#include "sycl_lsh/mpi/detail/file_parser/file_parser.hpp"  // sycl_lsh::mpi::detail::make_file_parser
#include "sycl_lsh/mpi/detail/logging.hpp"                  // sycl_lsh::mpi::detail::log

int custom_main(int &argc, char **&argv) {
// create a SYCL queue
#if defined(SYCL_LSH_USE_CPU)
    // Select the default CPU device to run the SYCL kernels on.
    sycl::queue queue{ sycl::cpu_selector_v };
#else
    // Select the default GPU device to run the SYCL kernel on.
    sycl::queue queue{ sycl::gpu_selector_v };
#endif

    // create MPI communicator
    const sycl_lsh::mpi::communicator comm{};

    try {
        // parse options and print
        const sycl_lsh::options opt(comm, argc, argv);
        sycl_lsh::mpi::detail::log(comm, "Used options: \n{}\n", opt);

        // log current number of MPI ranks
        sycl_lsh::mpi::detail::log(comm, "MPI_Comm_size: {}\n\n", comm.size());

        // parse data and print data attributes
        sycl_lsh::data_set data{ comm, opt.data_file };
        sycl_lsh::mpi::detail::log(comm, "\nUsed data set:\n{}\n", data);

        // create a nearest-neighbors object performing the unsupervised learning task
        sycl_lsh::nearest_neighbors nn{ comm, queue, sycl_lsh::n_neighbors = opt.n_neighbors, sycl_lsh::lsh_options = opt.lsh_options };

        // fit the data
        nn.fit(data);

        // calculate the nearest-neighbors
        const auto [indices, distances] = nn.kneighbors(sycl_lsh::return_distance = true);

        // TODO: better hide behind some better API?

        // optionally save calculated k-nearest-neighbor IDs
        if (opt.knn_save_file.has_value()) {
            const auto parser = sycl_lsh::mpi::detail::make_file_parser<sycl_lsh::index_type>(opt.knn_save_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::detail::file::mode::write, comm);
            parser->write_content(data.get_attributes().total_size, opt.n_neighbors, indices);
        }
        // optionally save calculated k-nearest-neighbor distances
        if (opt.knn_dist_save_file.has_value() && distances.has_value()) {
            const auto parser = sycl_lsh::mpi::detail::make_file_parser<sycl_lsh::real_type>(opt.knn_dist_save_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::detail::file::mode::write, comm);
            parser->write_content(data.get_attributes().total_size, opt.n_neighbors, distances.value());
        }

        // optionally calculate the recall of the calculated k-nearest-neighbors
        if (opt.evaluate_knn_file.has_value()) {
            const auto parser = sycl_lsh::mpi::detail::make_file_parser<sycl_lsh::index_type>(opt.evaluate_knn_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::detail::file::mode::read, comm);
            const sycl_lsh::aos_matrix<sycl_lsh::index_type> correct_indices = parser->parse_content();
            sycl_lsh::mpi::detail::log(comm, "recall: {}%\n", sycl_lsh::report::recall(indices, correct_indices, comm));
        }
        // optionally calculate the error ration of the calculated k-nearest-neighbors
        if (opt.evaluate_knn_dist_file.has_value() && distances.has_value()) {
            const auto parser = sycl_lsh::mpi::detail::make_file_parser<sycl_lsh::real_type>(opt.evaluate_knn_dist_file.value(), sycl_lsh::mpi::file_parser_type::binary, sycl_lsh::mpi::detail::file::mode::read, comm);
            const sycl_lsh::aos_matrix<sycl_lsh::real_type> correct_distances = parser->parse_content();
            const auto [error_ratio, num_points, num_knn_not_found] = sycl_lsh::report::error_ratio(distances.value(), correct_distances, comm);
            if (num_points == 0) {
                sycl_lsh::mpi::detail::log(comm, "error ratio: {}\n", error_ratio);
            } else {
                sycl_lsh::mpi::detail::log(comm, "error ratio: {} (for {} points a total of {} nearest-neighbors couldn't be found)\n", error_ratio, num_points, num_knn_not_found);
            }
        }
    } catch (const sycl_lsh::cmd_parser_exit &e) {
        return e.exit_code();
    } catch (const sycl_lsh::exception &e) {
        sycl_lsh::mpi::detail::log(comm, "Exception thrown on rank {}: {}\n", comm.rank(), e.what_with_loc());
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        sycl_lsh::mpi::detail::log(comm, "Exception thrown on rank {}: {}\n", comm.rank(), e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return sycl_lsh::mpi::main(argc, argv, &custom_main);
}
