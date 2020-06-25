/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-25
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

        // create data object
//        auto [data, data_buffers] = make_data<memory_layout::aos>(opt, data_file, communicator);
////        auto [data, buffers] = make_data<memory_layout::aos>(opt, 10, 3, communicator);
//        detail::mpi_print(comm_rank, "\nUsed data set: \n{}\n\n", detail::to_string(data).c_str());


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


        // create SYCL queue
        sycl::queue queue(sycl::gpu_selector{}, sycl::async_handler(&sycl_exception_handler));
        detail::mpi_print(comm_rank, "Used device: {}\n", queue.get_device().get_info<sycl::info::device::name>().c_str());


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

