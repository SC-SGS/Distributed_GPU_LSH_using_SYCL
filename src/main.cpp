/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-29
 *
 * @brief The main file containing the main logic.
 */

#include <filesystem>
#include <iostream>
#include <stdlib.h>
#include <utility>

#include <mpi.h>

#include <argv_parser.hpp>
#include <config.hpp>
#include <detail/assert.hpp>
#include <detail/timing.hpp>
#include <data.hpp>
#include <evaluation.hpp>
#include <exceptions/mpi_exception.hpp>
#include <exceptions/mpi_file_exception.hpp>
#include <hash_function.hpp>
#include <entropy_hash_function.hpp>
#include <hash_table.hpp>
#include <knn.hpp>
#include <mpi_buffer.hpp>
#include <options.hpp>


/**
 * @brief Asynchronous exception handler for exceptions thrown during SYCL kernel invocations.
 * @param[in] exceptions list of thrown SYCL exceptions
 */
void sycl_exception_handler(sycl::exception_list exceptions) {
    for (const std::exception_ptr& ptr : exceptions) {
        try {
            std::rethrow_exception(ptr);
        } catch (const sycl::exception& e) {
            std::cerr << "Asynchronous SYCL exception thrown: " << e.what() << std::endl;
        }
    }
}
/**
 * @brief Exception handler for errors occurred due to a call to a MPI function.
 * @param[in] comm the *MPI_Comm* on which the error occurred
 * @param[in] err the occurred error
 * @param[in] ... additional arguments
 */
void mpi_comm_exception_handler(MPI_Comm* comm, int* err, ...) {
    throw mpi_exception(*comm, *err);
}
/**
 * @brief Exception handler for errors occurred due to a call to a MPI_File function.
 * @param[in] file the *MPI_File* on which the error occurred
 * @param[in] err the occurred error
 * @param[in] ... additional arguments
 */
void mpi_file_exception_handler(MPI_File* file, int* err, ...) {
    throw mpi_file_exception(*file, *err);
}

/**
 * @brief For every GPU on a node one MPI process should be spawned
 * -> set CUDA_VISIBLE_DEVICES to the MPI rank of the MPI process on the current node.
 * @param[in] communicator the MPI_Comm communicator
 * @param[in] num_cuda_devices the number of available CUDA devices on the current node
 */
void setup_cuda_devices(const MPI_Comm& communicator) {
    int comm_size, comm_rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &comm_rank);

    // create communicator for each node
    MPI_Comm node_communicator;
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, comm_size, MPI_INFO_NULL, &node_communicator);
    // get node size and rank
    int comm_node_size, comm_node_rank;
    MPI_Comm_size(node_communicator, &comm_node_size);
    MPI_Comm_rank(node_communicator, &comm_node_rank);

    // set a CUDA_VISIBLE_DEVICES for each MPI process on the current rank
    int err = setenv("CUDA_VISIBLE_DEVICES", std::to_string(comm_node_rank).c_str(), 1);
    if (err != 0) {
        throw std::logic_error("Error while setting CUDA_VISIBLE_DEVICES environment variable!");
    }
    if (const char* env_val = getenv("CUDA_VISIBLE_DEVICES"); env_val != nullptr) {
        detail::mpi_print(comm_rank, "Used CUDA device on world rank {} and node rank: CUDA_VISIBLE_DEVICES={}\n",
                comm_rank, comm_node_rank, env_val);
    }

    // free communicator
    MPI_Comm_free(&node_communicator);


    // test for correctness
    const auto device_list = sycl::platform::get_platforms()[0].get_devices();
    // if the current device is a GPU AND no CUDA_VISIBLE_DEVICE is set, i.e. MORE MPI processes than GPUs per node were spawned, throw
    if (device_list[0].is_gpu() && device_list.size() != 1) {
        throw std::invalid_argument("Can't use more MPI processes per node than available GPUs per node!");
    }
}


int custom_main(MPI_Comm& communicator, const int argc, char** argv) {
    int comm_rank;

    try {
        // duplicate MPI_COMM_WORLD
        MPI_Comm_dup(MPI_COMM_WORLD, &communicator);

        // set special MPI error handlers
        MPI_Errhandler mpi_error_handler;
        MPI_Comm_create_errhandler(mpi_comm_exception_handler, &mpi_error_handler);
        MPI_Comm_set_errhandler(communicator, mpi_error_handler);
        MPI_File_create_errhandler(mpi_file_exception_handler, &mpi_error_handler);
        MPI_File_set_errhandler(MPI_FILE_NULL, mpi_error_handler);

        // get MPI environment information
        int comm_size;
        MPI_Comm_size(communicator, &comm_size);
        MPI_Comm_rank(communicator, &comm_rank);


        // parse command line arguments
        argv_parser parser(comm_rank, argc, argv);

        // display help message
        if (parser.has_argv("help")) {
            detail::mpi_print(comm_rank, parser.description().c_str());
            return EXIT_SUCCESS;
        }


        // print MPI info
        detail::mpi_print(comm_rank, "MPI_Comm_size: {} (on rank: {})\n\n", comm_size, comm_rank);


        // read options file
        using options_type = options<>;
        options_type::factory options_factory(comm_rank);
        if (parser.has_argv("options")) {
            auto options_file = parser.argv_as<std::string>("options");
            options_factory = decltype(options_factory)(comm_rank, options_file);

            detail::mpi_print(comm_rank, "Reading options from file: '{}'\n\n", options_file.c_str());
        }

        // change options values through factory functions using the provided values
        if (parser.has_argv("hash_pool_size")) {
            options_factory.set_hash_pool_size(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options_type>().hash_pool_size)>>("hash_pool_size"));
        }
        if (parser.has_argv("num_cut_off_points")) {
            options_factory.set_num_cut_off_points(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options_type>().num_cut_off_points)>>("num_cut_off_points"));
        }
        if (parser.has_argv("num_hash_tables")) {
            options_factory.set_num_hash_tables(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options_type>().num_hash_tables)>>("num_hash_tables"));
        }
        if (parser.has_argv("hash_table_size")) {
            options_factory.set_hash_table_size(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options_type>().hash_table_size)>>("hash_table_size"));
        }
        if (parser.has_argv("num_hash_functions")) {
            options_factory.set_num_hash_functions(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options_type>().num_hash_functions)>>("num_hash_functions"));
        }
        if (parser.has_argv("w")) {
            options_factory.set_w(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options_type>().w)>>("w"));
        }

        // create options object from factory
        options_type opt = options_factory.create();
        detail::mpi_print(comm_rank, "Used options: \n{}\n\n", detail::to_string(opt).c_str());

        // save the options file
        if (parser.has_argv("save_options")) {
            auto options_save_file = parser.argv_as<std::string>("save_options");
            if (comm_rank == 0) {
                opt.save(options_save_file);
            }

            detail::mpi_print(comm_rank, "Saved options to: '{}'\n\n", options_save_file.c_str());
        }


        // read data file
        std::string data_file;
        if (parser.has_argv("data")) {
            data_file = parser.argv_as<std::string>("data");

            detail::mpi_print(comm_rank, "Reading data from file: '{}'\n", data_file.c_str());
        } else {
            detail::mpi_print(comm_rank, "\nNo data file provided!\n");
            return EXIT_FAILURE;
        }


        // set CUDA_VISIBLE_DEVICES
//        setup_cuda_devices(communicator);

        
        START_TIMING(all);

        // create data object
        START_TIMING(parsing_data);
        auto [data, data_buffer] = make_data<memory_layout::aos>(opt, data_file, communicator);
//        auto [data, data_buffers] = make_data<memory_layout::aos>(opt, 150, 5, communicator);
        detail::mpi_print(comm_rank, "\nUsed data set: \n{}\n\n", detail::to_string(data).c_str());
        END_TIMING_MPI(parsing_data, comm_rank);

        // read the number of nearest-neighbours to search for
        typename options_type::index_type k = 0;
        if (parser.has_argv("k")) {
            k = parser.argv_as<decltype(k)>("k");

            DEBUG_ASSERT_MPI(comm_rank, 0 < k, "Illegal number of nearest-neighbors!: 0 < {}", k);

            detail::mpi_print(comm_rank, "Number of nearest-neighbors to search for: {}\n\n", k);
        } else {
            detail::mpi_print(comm_rank, "\nNo number of nearest-neighbors given!\n");
            return EXIT_FAILURE;
        }


        // create a SYCL queue for each device
        sycl::queue queue(sycl::default_selector{}, sycl::async_handler(&sycl_exception_handler));
        detail::print("[{}, {}]\n", comm_rank, queue.get_device().get_info<sycl::info::device::name>().c_str());

//        [[maybe_unused]] auto entropy_functions = make_entropy_hash_functions<memory_layout::aos>(data, communicator);

        // create hash tables
        START_TIMING(creating_hash_tables);
//        auto functions = make_entropy_hash_functions<memory_layout::aos>(data, communicator);
        auto functions = make_hash_functions<memory_layout::aos>(data, communicator);
        auto tables = make_hash_tables(queue, functions, communicator);
        END_TIMING_MPI_AND_BARRIER(creating_hash_tables, comm_rank, queue);

        // calculate k-nearest-neighbors
        START_TIMING(calculating_knns_total);
        auto knns = make_knn<memory_layout::aos>(k, data, communicator);
        detail::mpi_print(comm_rank, "\n");
        for (int round = 0; round < comm_size; ++round) {
            detail::mpi_print(comm_rank, "Round {} of {}\n", round + 1, comm_size);
            // calculate k-nearest-neighbors
            if (round == 0) {
                tables.calculate_knn(k, knns);
            } else {
                tables.calculate_knn(k, data_buffer, knns);
            }
            // asynchronously send data to next rank
            data_buffer.send_receive();

            // wait until all k-nearest-neighbors were calculated
            queue.wait();
            // send calculated k-nearest-neighbors and distances to next rank
            knns.buffers_knn.send_receive();
            knns.buffers_dist.send_receive();
            // wait until ALL communication has finished
            MPI_Barrier(communicator);
        }
        // wait until all kernels have finished
        queue.wait_and_throw();
        END_TIMING_MPI_AND_BARRIER(calculating_knns_total, comm_rank, queue);

        END_TIMING_MPI_AND_BARRIER(all, comm_rank, queue);

        // save the calculated k-nearest-neighbours
        if (parser.has_argv("save_knn")) {
            auto knns_save_file = parser.argv_as<std::string>("save_knn");

            detail::mpi_print(comm_rank, "\nSaving knns to: '{}'\n", knns_save_file.c_str());
            knns.save(knns_save_file, communicator);
        }

        if (parser.has_argv("evaluate_knn")) {
            detail::mpi_print(comm_rank, "\n");
            START_TIMING(parsing_correct_knns);
            using index_type = typename options_type::index_type;
            auto correct_knns_parser = make_file_parser<options_type, index_type>(parser.argv_as<std::string>("evaluate_knn"), communicator);

            DEBUG_ASSERT_MPI(comm_rank, data.total_size == correct_knns_parser->parse_total_size(),
                    "Total sizes mismatch!: {} != {}", data.total_size, correct_knns_parser->parse_total_size());
            DEBUG_ASSERT_MPI(comm_rank, data.rank_size == correct_knns_parser->parse_rank_size(),
                    "Rank sizes mismatch!: {} != {}", data.rank_size, correct_knns_parser->parse_rank_size());
            DEBUG_ASSERT_MPI(comm_rank, k == correct_knns_parser->parse_dims(),
                    "Number of nearest-neighbors mismatch!: {} != {}", k, correct_knns_parser->parse_dims());

            correct_knns_parser->parse_content(knns.buffers_knn.inactive().data());

            std::filesystem::path p(parser.argv_as<std::string>("evaluate_knn"));
            std::string dist_file_name = p.stem().string() + "_dist" + p.extension().string();
            auto correct_knns_dist_parser = make_file_parser<options_type>(p.replace_filename(dist_file_name).string(), communicator);

            DEBUG_ASSERT_MPI(comm_rank, data.total_size == correct_knns_dist_parser->parse_total_size(),
                             "Total sizes mismatch!: {} != {}", data.total_size, correct_knns_dist_parser->parse_total_size());
            DEBUG_ASSERT_MPI(comm_rank, data.rank_size == correct_knns_dist_parser->parse_rank_size(),
                             "Rank sizes mismatch!: {} != {}", data.rank_size, correct_knns_dist_parser->parse_rank_size());
            DEBUG_ASSERT_MPI(comm_rank, k == correct_knns_dist_parser->parse_dims(),
                             "Number of nearest-neighbors mismatch!: {} != {}", k, correct_knns_dist_parser->parse_dims());

            correct_knns_dist_parser->parse_content(knns.buffers_dist.inactive().data());
            END_TIMING_MPI(parsing_correct_knns, comm_rank);

            START_TIMING(evaluating);
            detail::mpi_print(comm_rank, "\nrecall: {} %\n", average(communicator, recall(knns, comm_rank)));
            const auto [error_ration_percent, num_points_not_found, num_knn_not_found] = error_ratio(knns, data_buffer, comm_rank);
            if (num_points_not_found != 0) {
                detail::mpi_print(comm_rank, "error ratio: {} % (for {} points a total of {} nearest-neighbors couldn't be found)\n",
                        average(communicator, error_ration_percent), sum(communicator, num_points_not_found), sum(communicator, num_knn_not_found));
            } else {
                detail::mpi_print(comm_rank, "error ratio: {} %\n", average(communicator, error_ration_percent));
            }
            END_TIMING_MPI(evaluating, comm_rank);
        }

    } catch (const mpi_exception& e) {
        detail::print("Exception thrown on rank {}: '{}' (error code: {})\n", comm_rank, e.what(), e.error_code());
        return EXIT_FAILURE;
    } catch (const mpi_file_exception& e) {
        detail::print("File exception thrown on rank {}: '{}' (error code: {})\n", comm_rank, e.what(), e.error_code());
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        detail::print("Exception thrown on rank {}: {}\n", comm_rank, e.what());
        return EXIT_FAILURE;
    } catch (...) {
        detail::print("Something went terrible wrong on rank {}!\n", comm_rank);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
    // initialize MPI environment
    MPI_Init(&argc, &argv);

    MPI_Comm communicator;
    int exit_code = custom_main(communicator, argc, argv);

    // release MPI resources
    MPI_Comm_free(&communicator);
    MPI_Finalize();

    return exit_code;
}

