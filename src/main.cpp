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

#include <boost/program_options.hpp>

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
        boost::program_options::options_description desc{"options"};
        desc.add_options()
            ("help,h", "help screen")
            ("options", boost::program_options::value<std::string>(), "path to the options file")
            ("save_options", boost::program_options::value<std::string>(), "save the currently used options to path")
            ("data", boost::program_options::value<std::string>(), "path to the data file (required)")
            ("save_knn", boost::program_options::value<std::string>(), "save the calculate nearest-neighbours to path")
            ("k", boost::program_options::value<typename options<>::index_type>(), "the number of nearest-neighbours to search for (required)")
            ("num_hash_tables", boost::program_options::value<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_tables)>>(), "number of hash tables to create")
            ("hash_table_size", boost::program_options::value<std::remove_cv_t<decltype(std::declval<options<>>().hash_table_size)>>(), "size of each hash table (should be a prime)")
            ("num_hash_functions", boost::program_options::value<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_functions)>>(), "number of hash functions per hash table")
            ("w", boost::program_options::value<std::remove_cv_t<decltype(std::declval<options<>>().w)>>(), "constant used in the hash functions")
        ;

        boost::program_options::variables_map vm;
        boost::program_options::store(parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        // display help message
        if (vm.count("help")) {
            std::cout << "Usage: ./prog --data \"path-to-data_set\" [options]\n";
            std::cout << desc;
            return EXIT_SUCCESS;
        }

        // read options file
        options<>::factory options_factory;
        if (vm.count("options")) {
            std::string options_file = vm["options"].as<std::string>();
            options_factory = options<>::factory(options_file);

            std::cout << "Reading options from file: '" << options_file << "'\n" << std::endl;
        }

        // change options values through factory functions using the provided values
        if (vm.count("num_hash_tables")) {
            options_factory.set_num_hash_tables(
                    vm["num_hash_tables"].as<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_tables)>>());
        }
        if (vm.count("hash_table_size")) {
            options_factory.set_hash_table_size(
                    vm["hash_table_size"].as<std::remove_cv_t<decltype(std::declval<options<>>().hash_table_size)>>());
        }
        if (vm.count("num_hash_functions")) {
            options_factory.set_num_hash_functions(
                    vm["num_hash_functions"].as<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_functions)>>());
        }
        if (vm.count("w")) {
            options_factory.set_w(
                    vm["w"].as<std::remove_cv_t<decltype(std::declval<options<>>().w)>>());
        }

        // create options object from factory
        options opt = options_factory.create();
        std::cout << "Used options: \n" << opt << '\n' << std::endl;

        // save the options file
        if (vm.count("save_options")) {
            std::string options_save_file = vm["save_options"].as<std::string>();
            opt.save(options_save_file);

            std::cout << "Saved options to: '" << options_save_file << "'\n" << std::endl;
        }


        // read data file
        std::string data_file;
        if (vm.count("data")) {
            data_file = vm["data"].as<std::string>();

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
        if (vm.count("k")) {
            k = vm["k"].as<decltype(k)>();
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
        if (vm.count("save_knn")) {
            std::string knns_save_file = vm["save_knn"].as<std::string>();
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

    } catch (const boost::program_options::error& e) {
        std::cerr << "Error while using boost::program_options: " <<  e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Something went terrible wrong!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

