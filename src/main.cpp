/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-19
 *
 * @brief The main file containing the main logic.
 */

#include <cstdio>
#include <iostream>
#include <utility>

#include <argv_parser.hpp>
#include <config.hpp>
#include <cstdlib>
#include <options.hpp>
#include <data.hpp>
#include <hash_function.hpp>
#include <hash_table.hpp>
#include <knn.hpp>
#include <evaluation.hpp>


/**
 * @brief Asynchronous exception handler for exceptions thrown during SYCL kernel invocations.
 * @param[in] exceptions list of thrown SYCL exceptions
 */
void exception_handler(sycl::exception_list exceptions) {
    for (const std::exception_ptr& ptr : exceptions) {
        try {
            std::rethrow_exception(ptr);
        } catch (const sycl::exception& e) {
            std::cerr << "Asynchronous SYCL exception thrown: " << e.what() << std::endl;
        }
    }
}


int main(int argc, char** argv) {
    try
    {
        argv_parser parser(argc, argv);

        // display help message
        if (parser.has_argv("help")) {
            std::cout << parser.description() << std::endl;
            return EXIT_SUCCESS;
        }

        // read options file
        options<>::factory options_factory;
        if (parser.has_argv("options")) {
            auto options_file = parser.argv_as<std::string>("options");
            options_factory = options<>::factory(options_file);

            std::cout << "Reading options from file: '" << options_file << "'\n" << std::endl;
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
        std::cout << "Used options: \n" << opt << '\n' << std::endl;

        // save the options file
        if (parser.has_argv("save_options")) {
            auto options_save_file = parser.argv_as<std::string>("save_options");
            opt.save(options_save_file);

            std::cout << "Saved options to: '" << options_save_file << "'\n" << std::endl;
        }


        // read data file
        std::string data_file;
        if (parser.has_argv("data")) {
            data_file = parser.argv_as<std::string>("data");

            std::cout << "Reading data from file: '" << data_file << '\'' << std::endl;
        } else {
            std::cerr << "\nNo data file provided!" << std::endl;
            return EXIT_FAILURE;
        }

        // create data object
//        auto data = make_data<memory_layout::aos>(opt, data_file);
        auto data = make_data<memory_layout::aos>(opt, 10, 3);
        std::cout << "\nUsed data set: \n" << data << '\n' << std::endl;

        // read the number of nearest-neighbours to search for
        typename decltype(opt)::index_type k = 0;
        if (parser.has_argv("k")) {
            k = parser.argv_as<decltype(k)>("k");
            DEBUG_ASSERT(0 < k, "Illegal number of nearest neighbors!: 0 < {}", k);

            std::cout << "Number of nearest-neighbours to search for: " << k << '\n' << std::endl;
        } else {
            std::cerr << "\nNo number of nearest-neighbours given!" << std::endl;
            return EXIT_FAILURE;
        }

        sycl::queue queue(sycl::default_selector{}, sycl::async_handler(&exception_handler));
        std::cout << "Used device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n' << std::endl;

        START_TIMING(creating_hash_tables);

        auto hash_functions = make_hash_functions<memory_layout::aos>(data);
        auto hash_tables = make_hash_tables(queue, hash_functions);

        END_TIMING_WITH_BARRIER(creating_hash_tables, queue);

        auto knns = hash_tables.calculate_knn<memory_layout::aos>(k);

        // wait until all kernels have finished
        queue.wait_and_throw();

        // save the calculated k-nearest-neighbours
        if (parser.has_argv("save_knn")) {
            auto knns_save_file = parser.argv_as<std::string>("save_knn");
            knns.save(knns_save_file);

            std::cout << "\nSaved knns to: '" << knns_save_file << '\'' << std::endl;
        }

        using index_type = typename decltype(opt)::index_type;
        std::vector<index_type> vec;
        vec.reserve(data.size * k);
        for (index_type i = 0; i < data.size; ++i) {
            for (index_type j = 0; j < k; ++j) {
                vec.emplace_back(i);
            }
        }

        std::printf("recall: %.2f%%\n", recall(knns, vec) * 100);
        std::printf("error ratio: %.2f%%\n", error_ratio(knns, vec, data) * 100);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Something went terrible wrong!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

