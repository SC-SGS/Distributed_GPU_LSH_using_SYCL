/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-04
 *
 * @brief The main file containing the main logic.
 */

#include <cstdio>
#include <iostream>
#include <utility>
#include <sstream>

#include <mpi.h>

#include <argv_parser.hpp>
#include <config.hpp>
#include <detail/assert.hpp>
#include <data.hpp>
#include <evaluation.hpp>
#include <hash_function.hpp>
#include <hash_table.hpp>
#include <knn.hpp>
#include <options.hpp>


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


template <typename T>
struct buffers {
    buffers(const std::size_t size, const std::size_t dims, int comm_size, int comm_rank)
        : active_(0), comm_size_(comm_size), comm_rank_(comm_rank), buffer_1_(size * dims), buffer_2_(size * dims) { }

    std::vector<T>& active() { return active_ == 0 ? buffer_1_ : buffer_2_; }
    std::vector<T>& inactive() { return active_ == 1 ? buffer_1_ : buffer_2_; }

    void send_receive() {
        int dest = (comm_rank_ + 1) % comm_size_;
        int source = (comm_size_ + (comm_rank_ - 1) % comm_size_) % comm_size_;
        MPI_Sendrecv(this->active().data(), this->active().size(), MPI_DOUBLE, dest, 0,
                     this->inactive().data(), this->inactive().size(), MPI_DOUBLE, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        active_ = (active_ + 1) % 2;
    }

    friend std::ostream& operator<<(std::ostream& out, const buffers& buf) {
        std::stringstream ss;
        ss << "rank " << buf.comm_rank_ << ": buffer_1 ";
        for (const T val : buf.buffer_1_) {
            if (val < 10) ss << 0;
            ss << val << ' ';
        }
        if (buf.active_ == 0) ss << " -> active";
        ss << "\n        buffer_2 ";
        for (const T val : buf.buffer_2_) {
            if (val < 10) ss << 0;
            ss << val << ' ';
        }
        if (buf.active_ == 1) ss << " -> active";
        out << ss.str() << '\n';
        return out;
    }

    int active_;
    int comm_size_;
    int comm_rank_;
    std::vector<T> buffer_1_;
    std::vector<T> buffer_2_;
};


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    std::printf("%i / %i\n", comm_rank + 1, comm_size);

    using real_type = double;
    const std::size_t size = 10;
    const std::size_t dims = 3;

    // create host buffers
    buffers<real_type> buff(size, dims, comm_size, comm_rank);

    // fill first buffer (later: with data from file)
    std::iota(buff.active().begin(), buff.active().end(), comm_rank * size * dims);

    sycl::queue queue(sycl::default_selector{});
    sycl::buffer<real_type, 1> data_device_buffer(buff.active().begin(), buff.active().end());

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 1; i < comm_size; ++i) {

        sycl::buffer<real_type> current_device_buffer(buff.active().begin(), buff.active().end());
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class test_kernel>(sycl::range<>(buff.active().size()), [=](sycl::item<> item) {
                const std::size_t idx = item.get_linear_id();
                if (idx == 0) detail::print("Index: {}\n", idx);
            });
        });

        if (comm_rank == 0) std::cerr << "Round: " <<  i << std::endl;
        buff.send_receive();
        std::cout << buff;
        MPI_Barrier(MPI_COMM_WORLD);
        queue.wait();
//        if (comm_rank == 0) std::cout << std::endl;
    }

//    try
//    {
//        argv_parser parser(argc, argv);
//
//        // display help message
//        if (parser.has_argv("help")) {
//            std::cout << parser.description() << std::endl;
//            return EXIT_SUCCESS;
//        }
//
//        // read options file
//        options<>::factory options_factory;
//        if (parser.has_argv("options")) {
//            auto options_file = parser.argv_as<std::string>("options");
//            options_factory = decltype(options_factory)(options_file);
//
//            std::cout << "Reading options from file: '" << options_file << "'\n" << std::endl;
//        }
//
//        // change options values through factory functions using the provided values
//        if (parser.has_argv("num_hash_tables")) {
//            options_factory.set_num_hash_tables(
//                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_tables)>>("num_hash_tables"));
//        }
//        if (parser.has_argv("hash_table_size")) {
//            options_factory.set_hash_table_size(
//                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().hash_table_size)>>("hash_table_size"));
//        }
//        if (parser.has_argv("num_hash_functions")) {
//            options_factory.set_num_hash_functions(
//                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().num_hash_functions)>>("num_hash_functions"));
//        }
//        if (parser.has_argv("w")) {
//            options_factory.set_w(
//                    parser.argv_as<std::remove_cv_t<decltype(std::declval<options<>>().w)>>("w"));
//        }
//
//        // create options object from factory
//        options opt = options_factory.create();
//        std::cout << "Used options: \n" << opt << '\n' << std::endl;
//
//        // save the options file
//        if (parser.has_argv("save_options")) {
//            auto options_save_file = parser.argv_as<std::string>("save_options");
//            opt.save(options_save_file);
//
//            std::cout << "Saved options to: '" << options_save_file << "'\n" << std::endl;
//        }
//
//
//        // read data file
//        std::string data_file;
//        if (parser.has_argv("data")) {
//            data_file = parser.argv_as<std::string>("data");
//
//            std::cout << "Reading data from file: '" << data_file << '\'' << std::endl;
//        } else {
//            std::cerr << "\nNo data file provided!" << std::endl;
//            return EXIT_FAILURE;
//        }
//
//        // create data object
////        auto data = make_data<memory_layout::aos>(opt, data_file);
//        auto data = make_data<memory_layout::aos>(opt, 10, 3);
//        std::cout << "\nUsed data set: \n" << data << '\n' << std::endl;
//
//        // read the number of nearest-neighbours to search for
//        typename decltype(opt)::index_type k = 0;
//        if (parser.has_argv("k")) {
//            k = parser.argv_as<decltype(k)>("k");
//            DEBUG_ASSERT(0 < k, "Illegal number of nearest neighbors!: 0 < {}", k);
//
//            std::cout << "Number of nearest-neighbours to search for: " << k << '\n' << std::endl;
//        } else {
//            std::cerr << "\nNo number of nearest-neighbours given!" << std::endl;
//            return EXIT_FAILURE;
//        }
//
//        sycl::queue queue(sycl::default_selector{}, sycl::async_handler(&exception_handler));
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
//
//    } catch (const std::exception& e) {
//        std::cerr << e.what() << std::endl;
//        return EXIT_FAILURE;
//    } catch (...) {
//        std::cerr << "Something went terrible wrong!" << std::endl;
//        return EXIT_FAILURE;
//    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

