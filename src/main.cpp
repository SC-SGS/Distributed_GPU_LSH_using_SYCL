/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-13
 *
 * @brief The main file containing the main logic.
 */

#include <iostream>

#include <config.hpp>
#include <options.hpp>
#include <data.hpp>
#include <hash_function.hpp>
#include <hash_table.hpp>
#include <detail/print.hpp>


int main() {
    try {
        sycl::queue queue(sycl::default_selector{ });
        options opt;

        auto data = make_data<memory_layout::aos>(opt, 10, 3);
//        auto data = make_data<memory_layout::aos>(opt, "../data_sets/DR5_nowarnings_less05_test.txt");
        auto hash_functions = make_hash_functions<memory_layout::aos>(data);
        auto hash_tables = make_hash_tables(queue, hash_functions);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}

