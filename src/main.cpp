/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-07
 *
 * @brief The main file containing the main logic.
 */

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
#include <hash_table.hpp>
#include <knn.hpp>
#include <mpi_buffer.hpp>
#include <options.hpp>
#include <vector>


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
void setup_cuda_devices(const MPI_Comm& communicator, const int num_cuda_devices) {
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
    // its not allowed to spawn more MPI processes than CUDA devices
    if (comm_node_size > num_cuda_devices) {
        throw std::invalid_argument("Can't use more MPI processes than available GPUs on a node!");
    }

    // set a CUDA_VISIBLE_DEVICES for each MPI process on the current rank
    int err = setenv("CUDA_VISIBLE_DEVICES", std::to_string(comm_node_rank).c_str(), 1);
    if (err != 0) {
        throw std::logic_error("Error while setting CUDA_VISIBLE_DEVICES environment variable!");
    }
    if (const char* env_val = getenv("CUDA_VISIBLE_DEVICES")) {
        detail::mpi_print(comm_rank, "Used CUDA device on rank {}: CUDA_VISIBLE_DEVICES={}\n", comm_rank, env_val);
    }

    // free communicator
    MPI_Comm_free(&node_communicator);
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

        // set CUDA_VISIBLE_DEVICES
        auto device_list = sycl::platform::get_platforms()[0].get_devices();
        if (device_list[0].is_gpu() && device_list[0].get_info<sycl::info::device::vendor>() == "NVIDIA") {
            setup_cuda_devices(communicator, device_list.size());
        }


//        using index_type = typename options_type::index_type;
//        using real_type = typename options_type::real_type;
//        auto fp = make_file_parser<options_type>(data_file, communicator);
//        mpi_buffers<real_type> buffer(fp->parse_rank_size(), fp->parse_dims(), communicator);
//        fp->parse_content(buffer.active().data());
//
//        constexpr memory_layout layout = memory_layout::soa;
//        if constexpr (layout == memory_layout::soa) {
//            using active_data_type = data<memory_layout::soa, options_type>;
//            using inactive_data_type = data<memory_layout::aos, options_type>;
//
//            auto& active = buffer.active();
//            auto& inactive = buffer.inactive();
//            for (index_type point = 0; point < buffer.rank_size; ++point) {
//                for (index_type dim = 0; dim < buffer.dims; ++dim) {
//                    inactive[inactive_data_type::get_linear_id(comm_rank, point, buffer.rank_size, dim, buffer.dims)] =
//                            active[active_data_type::get_linear_id(comm_rank, point, buffer.rank_size, dim, buffer.dims)];
//                }
//            }
//            buffer.swap_buffers();
//        }
//
//
//        const bool has_smaller_rank_size = static_cast<index_type>(comm_rank) >= (fp->parse_total_size() % comm_size);
//        const index_type rank_size = fp->parse_rank_size() - static_cast<index_type>(has_smaller_rank_size);
//        std::stringstream ss;
//        ss << "rank: " << comm_rank << " (buffer_size: " << buffer.active().size()
//                                    << ", total_size: " << fp->parse_total_size()
//                                    << ", ceiled_rank_size: " << buffer.rank_size
//                                    << ", rank_size: " << rank_size
//                                    << ", dims: " << buffer.dims << ") -> ";
//        for (const auto val : buffer.active()) {
//            ss << val << ' ';
//        }
//        std::cout << ss.str() << std::endl;
//        MPI_Barrier(communicator);
//        ss.str(std::string());
//        ss.clear();


        // create data object
        auto [data, data_buffers] = make_data<memory_layout::aos>(opt, data_file, communicator);
//        auto [data, buffers] = make_data<memory_layout::aos>(opt, 10, 3, communicator);
        detail::mpi_print(comm_rank, "\nUsed data set: \n{}\n\n", detail::to_string(data).c_str());




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


//        std::vector<index_type> knns(fp->parse_rank_size() * k);
//        std::iota(knns.begin(), knns.end(), 0 + std::pow(10, comm_rank));
//        ss << "knns rank: " << comm_rank << ": ";
//        for (const auto val : knns) {
//            ss << val << ' ';
//        }
//        std::cout << ss.str() << std::endl;
//        MPI_Barrier(communicator);
//
//        MPI_File file;
//        MPI_File_open(communicator, "saved_knns.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
//        MPI_File_write_ordered(file, knns.data(), rank_size * k, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
//        MPI_File_close(&file);


        // create a SYCL queue for each device
        std::vector<sycl::queue> queues;
        for (const sycl::platform& platform : sycl::platform::get_platforms()) {
            for (const sycl::device& device : platform.get_devices(sycl::info::device_type::gpu)) {
                queues.emplace_back(device, sycl::async_handler(&sycl_exception_handler));
            }
        }
        // print all available devices
        detail::mpi_print(comm_rank, "Used device(s):\n");
        for (std::size_t i = 0; i < queues.size(); ++i) {
            const std::string device_name = queues[i].get_device().get_info<sycl::info::device::name>();
            detail::mpi_print(comm_rank, "{}/{}: {}\n", i + 1, queues.size(), device_name.c_str());
        }

        for (const sycl::queue& queue : queues) {
            std::cout << std::boolalpha;
            if (comm_rank == 0) {
                using real_type = typename decltype(data)::real_type;
                std::cout << queue.get_context().is_host() << std::endl;
                sycl::buffer<real_type, 1> buf(data_buffers.active().begin(), data_buffers.active().end());
//                auto c = buf.get_property<sycl::property::buffer::context_bound>();
//                std::cout << c.get_context().is_host() << std::endl;
            }
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

