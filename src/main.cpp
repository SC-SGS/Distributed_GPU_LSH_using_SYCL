/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief The main file containing the main logic.
 */

#include "sycl_lsh/core.hpp"
#include "sycl_lsh/mpi/detail/logging.hpp"  // sycl_lsh::mpi::detail::log

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
    sycl_lsh::mpi::detail::log(comm, "Using {} MPI rank(s) for the nearest-neighbor calculation.\n\n", comm.size());

    try {
        // parse options and print
        const sycl_lsh::options opt(comm, argc, argv);
        sycl_lsh::mpi::detail::log(comm, "{}\n", opt);

        // parse data and print data attributes
        sycl_lsh::data_set data{ comm, opt.data_file };

        // create a nearest-neighbors object performing the unsupervised learning task
        sycl_lsh::nearest_neighbors nn{ comm, queue, sycl_lsh::n_neighbors = opt.n_neighbors, sycl_lsh::lsh_options = opt.lsh_options };

        // fit the data
        nn.fit(data);

        // calculate the nearest-neighbors
        const sycl_lsh::nearest_neighbors_result result = nn.kneighbors(sycl_lsh::return_distance = true);

        // optionally save the calculated nearest-neighbor indices to a file
        if (opt.indices_save_file.has_value()) {
            result.save_indices(opt.indices_save_file.value());
        }
        // optionally save the calculated nearest-neighbor distances to a file if return_distance was enabled
        if (opt.distances_save_file.has_value() && result.has_distances()) {
            result.save_distances(opt.distances_save_file.value());
        }

        // optionally calculate the recall of the calculated nearest-neighbors
        if (opt.indices_ground_truth_file.has_value()) {
            sycl_lsh::mpi::detail::log(comm, "recall: {:.2f}%\n", result.recall(opt.indices_ground_truth_file.value()));
        }
        // optionally calculate the error ratio of the calculated nearest-neighbors if return_distance was enabled
        if (opt.distances_ground_truth_file.has_value() && result.has_distances()) {
            const auto [error_ratio, num_points, num_knn_not_found] = result.error_ratio(opt.distances_ground_truth_file.value());
            if (num_points == 0) {
                sycl_lsh::mpi::detail::log(comm, "error ratio: {:.4f}\n", error_ratio);
            } else {
                sycl_lsh::mpi::detail::log(comm, "error ratio: {:.4f} (for {} points a total of {} nearest-neighbors couldn't be found)\n", error_ratio, num_points, num_knn_not_found);
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
