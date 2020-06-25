/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-24
 *
 * @brief The main file containing the main logic.
 */

#include <iostream>
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
void mpi_exception_handler(MPI_Comm* comm, int* err, ...) {
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


int main(int argc, char** argv) {
    MPI_Comm communicator;
    int comm_rank;

    try {
        // initialize MPI environment
        MPI_Init(&argc, &argv);

        // duplicate MPI_COMM_WORLD
        MPI_Comm_dup(MPI_COMM_WORLD, &communicator);

        // set special MPI error handlers
        MPI_Errhandler mpi_error_handler;
        MPI_Comm_create_errhandler(mpi_exception_handler, &mpi_error_handler);
        MPI_Comm_set_errhandler(communicator, mpi_error_handler);
        MPI_File_create_errhandler(mpi_file_exception_handler, &mpi_error_handler);
        MPI_File_set_errhandler(MPI_FILE_NULL, mpi_error_handler);

        // print MPI info
        int comm_size;
        MPI_Comm_size(communicator, &comm_size);
        MPI_Comm_rank(communicator, &comm_rank);
        detail::mpi_print<print_rank>(comm_rank, "MPI_Comm_size: {} (on rank: {})\n\n", comm_size, comm_rank);


        // parse command line arguments
        argv_parser parser(argc, argv, comm_rank);

        // display help message
        if (parser.has_argv("help")) {
            detail::mpi_print<print_rank>(comm_rank, parser.description().c_str());
            return EXIT_SUCCESS;
        }

        // read options file
        using options_type = options<>;
        options_type::factory options_factory(comm_rank);
        if (parser.has_argv("options")) {
            auto options_file = parser.argv_as<std::string>("options");
            options_factory = decltype(options_factory)(options_file, comm_rank);

            detail::mpi_print<print_rank>(comm_rank, "Reading options from file: '{}'\n\n", options_file.c_str());
        }

        // change options values through factory functions using the provided values
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
        options opt = options_factory.create();
        detail::mpi_print<print_rank>(comm_rank, "Used options: \n{}\n\n", detail::to_string(opt).c_str());

        // save the options file
        if (parser.has_argv("save_options")) {
            auto options_save_file = parser.argv_as<std::string>("save_options");
            if (comm_rank == 0) {
                opt.save(options_save_file);
            }

            detail::mpi_print<print_rank>(comm_rank, "Saved options to: '{}'\n\n", options_save_file.c_str());
        }


        // read data file
        std::string data_file;
        if (parser.has_argv("data")) {
            data_file = parser.argv_as<std::string>("data");

            detail::mpi_print<print_rank>(comm_rank, "Reading data from file: '{}'\n", data_file.c_str());
        } else {
            detail::mpi_print<print_rank>(comm_rank, "\nNo data file provided!\n");
            return EXIT_FAILURE;
        }

        // create data object
        auto [data, data_buffers] = make_data<memory_layout::aos>(opt, data_file, communicator);
//        auto [data, buffers] = make_data<memory_layout::aos>(opt, 10, 3, communicator);
        detail::mpi_print<print_rank>(comm_rank, "\nUsed data set: \n{}\n\n", detail::to_string(data).c_str());


        // read the number of nearest-neighbours to search for
        typename options_type::index_type k = 0;
        if (parser.has_argv("k")) {
            k = parser.argv_as<decltype(k)>("k");

            DEBUG_ASSERT_MPI(comm_rank, 0 < k, "Illegal number of nearest neighbors!: 0 < {}", k);

            detail::mpi_print<print_rank>(comm_rank, "Number of nearest-neighbors to search for: {}\n\n", k);
        } else {
            detail::mpi_print<print_rank>(comm_rank, "\nNo number of nearest-neighbors given!\n");
            return EXIT_FAILURE;
        }


        sycl::queue queue(sycl::gpu_selector{}, sycl::async_handler(&sycl_exception_handler));
        detail::mpi_print<print_rank>(comm_rank, "Used device: {}\n", queue.get_device().get_info<sycl::info::device::name>().c_str());


        START_TIMING(creating_hash_tables);

        auto hash_functions = make_hash_functions<memory_layout::aos>(data, communicator);
        auto hash_tables = make_hash_tables(queue, hash_functions, communicator);
        MPI_Comm_free(&communicator);
        MPI_Finalize();
        return EXIT_SUCCESS;

        END_TIMING_MPI_AND_BARRIER(creating_hash_tables, comm_rank, queue);

        auto knns = make_knn<memory_layout::aos>(k, data, communicator);

        // calculate k-nearest-neighbors
        detail::mpi_print<print_rank>(comm_rank, "\n");
        for (int rank = 0; rank < comm_size; ++rank) {
            detail::mpi_print<print_rank>(comm_rank, "Round {} of {}\n", rank + 1, comm_size);
            // calculate k-nearest-neighbors
            hash_tables.calculate_knn(k, data_buffers, knns);
            // asynchronously send data to next rank
            data_buffers.send_receive();

            // wait until all k-nearest-neighbors were calculated
            queue.wait();
            // send calculated k-nearest-neighbors to next rank
            knns.buffers.send_receive();
            // wait until ALL communication has finished
            MPI_Barrier(communicator);
        }
//        auto knns = hash_tables.calculate_knn<memory_layout::aos>(k);

        // wait until all kernels have finished
        queue.wait_and_throw();

        // save the calculated k-nearest-neighbours
        if (parser.has_argv("save_knn")) {
            auto knns_save_file = parser.argv_as<std::string>("save_knn");

            detail::mpi_print<print_rank>(comm_rank, "\nSaving knns to: '{}'\n", knns_save_file.c_str());
            knns.save(knns_save_file, communicator);
        }

//        // TODO 2020-06-23 17:11 marcel: correctly read correct knns
//        using index_type = typename options_type::index_type;
//        index_type* correct_knns = knns.buffers.inactive();
//        for (index_type point = 0; point < data.size; ++point) {
//            for (index_type nn = 0; nn < k; ++nn) {
//                correct_knns[point * k + nn] = point;
//            }
//        }
//
//        detail::mpi_print<print_rank>(comm_rank, "\nrecall: {} %\n", average(communicator, recall(knns)));
////        detail::mpi_print<print_rank>(comm_rank, "error ratio: {}\n", average(communicator, error_ratio(knns, data)));

    } catch (const mpi_exception& e) {
        detail::mpi_print<>(comm_rank, "Exception thrown on rank {}: '{}' (error code: {})\n", e.rank(), e.what(), e.error_code());
        return EXIT_FAILURE;
    } catch (const mpi_file_exception& e) {
        detail::mpi_print<>(comm_rank, "File exception thrown: '{}' (error code: {})\n", e.what(), e.error_code());
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        detail::mpi_print<>(comm_rank, "Exception thrown on rank {}: {}\n", comm_rank, e.what());
        return EXIT_FAILURE;
    } catch (...) {
        detail::mpi_print<>(comm_rank, "Something went terrible wrong on rank {}!\n", comm_rank);
        return EXIT_FAILURE;
    }

    MPI_Comm_free(&communicator);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

