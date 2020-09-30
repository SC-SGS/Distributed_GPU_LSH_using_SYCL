/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-30
 *
 * @brief The main file containing the main logic.
 */

#include <sycl_lsh/core.hpp>

struct sycl_test {};

int custom_main(int argc, char** argv) {
    // create MPI communicator
    sycl_lsh::mpi::communicator comm;
    // optionally: set exception handler for the communicator
    sycl_lsh::mpi::errhandler handler(sycl_lsh::mpi::errhandler::type::comm);
    comm.attach_errhandler(handler);

    // create default logger (logs to std::cout)
    sycl_lsh::mpi::logger logger(comm);

    try {

        // parse command line arguments
        sycl_lsh::argv_parser parser(argc, argv);
        // log help message if requested
        if (parser.has_argv("help")) {
            logger.log(sycl_lsh::argv_parser::description());
            return EXIT_SUCCESS;
        }

        // log current number of MPI ranks
        logger.log("MPI_Comm_size: {}\n\n", comm.size());

        // parse options and print
        const sycl_lsh::options<float, std::uint32_t, std::uint32_t, 10, sycl_lsh::hash_functions_type::random_projections> opt(parser, logger);
        logger.log("Used options: \n{}\n", opt);

        // optionally save generated options to file
        if (parser.has_argv("options_save_file")) {
            opt.save(comm, parser, logger);
        }


        sycl_lsh::mpi::binary_parser<decltype(opt), typename decltype(opt)::real_type> parser_file(parser.argv_as<std::string>("data_file"), comm, logger);

        logger.log_on_all("total size: {} -> {}\n", comm.rank(), parser_file.parse_total_size());
        logger.log_on_all("rank size: {} -> {}\n", comm.rank(), parser_file.parse_rank_size());
        logger.log_on_all("dims: {} -> {}\n", comm.rank(), parser_file.parse_dims());

        std::vector<float> buf(parser_file.parse_rank_size() * parser_file.parse_dims());
        parser_file.parse_content(buf.data());
        logger.log_on_all("content: {} -> {}\n", comm.rank(), fmt::join(buf, ", "));


        logger.log("{}\n", SYCL_LSH_TARGET == SYCL_LSH_TARGET_CPU);
        auto data = sycl_lsh::make_data<sycl_lsh::memory_layout::soa>(parser, opt, comm, logger);
        logger.log("{}\n", data);

        std::vector<float> vec(data.get_attributes().rank_size);
        {
            sycl_lsh::sycl::queue queue(sycl_lsh::sycl::default_selector{});
            logger.log("{}\n", queue.get_device().get_info<sycl_lsh::sycl::info::device::name>());

            sycl_lsh::sycl::buffer<float, 1> buf(vec.data(), vec.size());
            queue.submit([&](sycl_lsh::sycl::handler& cgh) {
                auto acc = data.get_device_buffer().template get_access<sycl_lsh::sycl::access::mode::read>(cgh);
                auto acc_res = buf.get_access<sycl_lsh::sycl::access::mode::discard_write>(cgh);
                const auto data_attr = data.get_attributes();

                cgh.parallel_for<sycl_test>(sycl_lsh::sycl::range<>(data_attr.rank_size), [=](sycl_lsh::sycl::item<> item){
                    const std::uint32_t idx = item.get_linear_id();

                    float val = 0;
                    for (std::uint32_t dim = 0; dim < data_attr.dims; ++dim) {
                        val += acc[sycl_lsh::get_linear_id__data(idx, dim, data_attr)];
                    }
                    acc_res[idx] = val;
                });
            });
        }
        logger.log_on_all("{}\n", fmt::join(vec, ", "));


        // TODO 2020-09-24 14:47 marcel: move at the end of actual k-nearest-neighbor function
#if defined(SYCL_LSH_BENCHMARK)
        if (comm.master_rank()) {
            sycl_lsh::mpi::timer::benchmark_out() << opt.hash_pool_size << ',' << opt.num_hash_functions << ',' << opt.num_hash_tables << ','
                                                  << opt.hash_table_size << ',' << opt.w << ',' << opt.num_cut_off_points << '\n';
        }
#endif

    } catch (const std::exception& e) {
        logger.log("Exception thrown on rank {}: {}\n", comm.rank(), e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
    return sycl_lsh::mpi::main(argc, argv, &custom_main);
}
