/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-12
 *
 * @brief The main file containing the main logic.
 */

#include <cstdio>
#include <iostream>
#include <optional>
#include <utility>
#include <sstream>

#include <chrono>
#include <thread>

#include <mpi.h>

#include <argv_parser.hpp>
#include <config.hpp>
#include <detail/assert.hpp>
#include <detail/mpi_type.hpp>
#include <data.hpp>
#include <evaluation.hpp>
#include <exceptions/mpi_exception.hpp>
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
 * @param[in] comm the communicator on which the error occurred
 * @param[in] err the occurred error
 * @param[in] ... additional arguments
 */
void mpi_exception_handler(MPI_Comm* comm, int* err, ...) {
    throw mpi_exception(*comm, *err);
}


//int calculate_nearest_neighbors(const MPI_Comm& communicator, const int comm_size, const int comm_rank, const std::size_t size, const std::size_t dims) {
//    using real_type = float;
//
//    // create host buffers
//    mpi_buffers<real_type> buff(communicator, size, dims);
//
//    // fill first buffer (later: with data from file)
//    std::iota(buff.active().begin(), buff.active().end(), comm_rank * size * dims);
//
//    sycl::queue queue(sycl::default_selector{});
//    sycl::buffer<real_type, 1> data_device_buffer(buff.active().begin(), buff.active().end());
//
//    MPI_Barrier(communicator);
//    for (int i = 1; i < comm_size; ++i) {
//
//        sycl::buffer<real_type> current_device_buffer(buff.active().begin(), buff.active().end());
//        {
//            queue.submit([&](sycl::handler& cgh) {
//                cgh.parallel_for<class test_kernel>(sycl::range<>(buff.active().size()), [=](sycl::item<> item) {
//                    const std::size_t idx = item.get_linear_id();
//                    if (idx == 0 && comm_rank == 0) detail::print("Index: {}\n", idx);
//                });
//            });
//        }
//
//        detail::mpi_print<print_rank>(comm_rank, "before sending\n");
//        buff.send_receive();
//        std::this_thread::sleep_for(std::chrono::seconds(2));
//        detail::mpi_print<print_rank>(comm_rank, "after sending\n");
//
////        std::cout << buff;
//
//        MPI_Barrier(communicator);
//        detail::mpi_print<print_rank>(comm_rank, "after MPI barrier\n");
//        queue.wait();
//        detail::mpi_print<print_rank>(comm_rank, "after SYCL barrier\n");
//    }

    ////        sycl::queue queue(sycl::default_selector{}, sycl::async_handler(&exception_handler));
////        std::cout << "Used device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n' << std::endl;
////
////        START_TIMING(creating_hash_tables);
////
////        auto hash_functions = make_hash_functions<memory_layout::aos>(data);
////        auto hash_tables = make_hash_tables(queue, hash_functions);
////
////        END_TIMING_WITH_BARRIER(creating_hash_tables, queue);
////
////        auto knns = hash_tables.calculate_knn<memory_layout::aos>(k);
////
////        // wait until all kernels have finished
////        queue.wait_and_throw();
////
////        // save the calculated k-nearest-neighbours
////        if (parser.has_argv("save_knn")) {
////            auto knns_save_file = parser.argv_as<std::string>("save_knn");
////            knns.save(knns_save_file);
////
////            std::cout << "\nSaved knns to: '" << knns_save_file << '\'' << std::endl;
////        }
////        std::cout << std::endl;
////
////        using index_type = typename decltype(opt)::index_type;
////        std::vector<index_type> vec;
////        vec.reserve(data.size * k);
////        for (index_type i = 0; i < data.size; ++i) {
////            for (index_type j = 0; j < k; ++j) {
////                vec.emplace_back(i);
////            }
////        }
////
////        std::printf("recall: %.2f %%\n", recall(knns, vec));
////        std::printf("error ratio: %.2f %%\n", error_ratio(knns, vec, data));
//
//    return EXIT_SUCCESS;
//}





// TODO 2020-06-10 18:49 marcel: MPI save knn file?
int main(int argc, char** argv) {
    constexpr int print_rank = 0;

    MPI_Comm communicator;
    int comm_rank;

    try {
        // initialize MPI environment
        MPI_Init(&argc, &argv);

        // duplicate MPI_COMM_WORLD
        MPI_Comm_dup(MPI_COMM_WORLD, &communicator);

        // set special MPI error handler
        MPI_Errhandler mpi_error_handler;
        MPI_Comm_create_errhandler(mpi_exception_handler, &mpi_error_handler);
        MPI_Comm_set_errhandler(communicator, mpi_error_handler);

        // print MPI info
        int comm_size;
        MPI_Comm_size(communicator, &comm_size);
        MPI_Comm_rank(communicator, &comm_rank);
        detail::mpi_print<print_rank>(comm_rank, "MPI_Comm_size: {} (on rank: {})\n\n", comm_size, comm_rank);


        // parse command line arguments
        argv_parser parser(argc, argv);

        // display help message
        if (parser.has_argv("help")) {
            detail::mpi_print<print_rank>(comm_rank, parser.description().c_str());
            return EXIT_SUCCESS;
        }

        // read options file
        options<>::factory options_factory;
        if (parser.has_argv("options")) {
            auto options_file = parser.argv_as<std::string>("options");
            options_factory = decltype(options_factory)(options_file);

            detail::mpi_print<print_rank>(comm_rank, "Reading options from file: '{}'\n\n", options_file.c_str());
        }

        // change options values through factory functions using the provided values
        if (parser.has_argv("num_hash_tables")) {
            options_factory.set_num_hash_tables(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_tables)>>("num_hash_tables"));
        }
        if (parser.has_argv("hash_table_size")) {
            options_factory.set_hash_table_size(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().hash_table_size)>>("hash_table_size"));
        }
        if (parser.has_argv("num_hash_functions")) {
            options_factory.set_num_hash_functions(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_functions)>>("num_hash_functions"));
        }
        if (parser.has_argv("w")) {
            options_factory.set_w(
                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().w)>>("w"));
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
//        auto data = make_data<memory_layout::aos>(opt, data_file);
        auto data = make_data<memory_layout::aos>(opt, 10, 3);
        detail::mpi_print<print_rank>(comm_rank, "\nUsed data set: \n{}\n\n", detail::to_string(data).c_str());

        // read the number of nearest-neighbours to search for
        typename decltype(opt)::index_type k = 0;
        if (parser.has_argv("k")) {
            k = parser.argv_as<decltype(k)>("k");
            if (comm_rank == 1) k = 0;
            DEBUG_ASSERT_MPI(0 < k, comm_rank, "Illegal number of nearest neighbors!: 0 < {}", k);

            detail::mpi_print<print_rank>(comm_rank, "Number of nearest-neighbors to search for: {}\n\n", k);
        } else {
            detail::mpi_print<print_rank>(comm_rank, "\nNo number of nearest-neighbors given!\n");
            return EXIT_FAILURE;
        }


//        sycl::queue queue(sycl::default_selector{}, sycl::async_handler(&sycl_exception_handler));
//        std::cout << "Used device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n' << std::endl;
//
//        START_TIMING(creating_hash_tables);
//
//        auto hash_functions = make_hash_functions<memory_layout::aos>(data);
//        auto hash_tables = make_hash_tables(queue, hash_functions);
//
//        END_TIMING_WITH_BARRIER(creating_hash_tables, queue);
//
//        auto knns = hash_tables.calculate_knn<memory_layout::aos>(k);
//
//        // wait until all kernels have finished
//        queue.wait_and_throw();
//
//        // save the calculated k-nearest-neighbours
//        if (parser.has_argv("save_knn")) {
//            auto knns_save_file = parser.argv_as<std::string>("save_knn");
//            knns.save(knns_save_file);
//
//            std::cout << "\nSaved knns to: '" << knns_save_file << '\'' << std::endl;
//        }
//        std::cout << std::endl;
//
//        using index_type = typename decltype(opt)::index_type;
//        std::vector<index_type> vec;
//        vec.reserve(data.size * k);
//        for (index_type i = 0; i < data.size; ++i) {
//            for (index_type j = 0; j < k; ++j) {
//                vec.emplace_back(i);
//            }
//        }
//
//        std::printf("recall: %.2f %%\n", recall(knns, vec));
//        std::printf("error ratio: %.2f %%\n", error_ratio(knns, vec, data));

    } catch (const mpi_exception& e) {
        detail::mpi_print<>(comm_rank, "Exception thrown on rank {}: '{}' (error code: {})\n", e.rank(), e.what(), e.error_code());
    } catch (const std::exception& e) {
        detail::mpi_print<>(comm_rank, "Exception thrown on rank {}: {}", comm_rank, e.what());
        return EXIT_FAILURE;
    } catch (...) {
        detail::mpi_print<>(comm_rank, "Something went terrible wrong on rank {}!", comm_rank);
        return EXIT_FAILURE;
    }

    MPI_Comm_free(&communicator);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

