#include <hash_table.hpp>

#include <random>
#include <cmath>
#include <vector>

hash_tables::hash_tables(sycl::queue& queue, sycl::buffer<real_t, 1>& data, const options& opt)
    : hash_functions_(sycl::range<1>{ opt.number_of_hash_tables * opt.number_of_hash_functions * (opt.dims + 1) }),
      hash_tables_(sycl::range<1>{ opt.number_of_hash_tables * opt.size })
{
    this->init_hash_functions(opt);

    std::vector<real_t> hash_values_count(opt.number_of_hash_tables * opt.prim, 0.0);
    sycl::buffer<real_t, 1> buf_hash_values_count(hash_values_count.data(), sycl::range<1>{ hash_values_count.size() });
    queue.submit([&](sycl::handler& cgh) {
        auto acc_count = buf_hash_values_count.get_access<sycl::access::mode::atomic>(cgh);
        auto acc_hash_func = hash_functions_.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
        auto acc_data = data.get_access<sycl::access::mode::read>(cgh);

        sycl::accessor<real_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>{ opt.local_size * opt.dims }, cgh);
        auto execution_range = sycl::nd_range<1>{ sycl::range<1>{ opt.global_size }, sycl::range<1>{ opt.local_size } };


        cgh.parallel_for<class hash_point>(execution_range, [=](sycl::nd_item<1> item_id) {
            const std::size_t idx = item_id.get_global_linear_id();
            const std::size_t local_idx = item_id.get_local_linear_id();

            if (idx >= opt.size) { return; }

            for (std::size_t i = 0; i < opt.dims; ++i) {
//                    local_mem[local_idx + i * local_size] = acc_data[i + idx * opt.dims]; // SoA
                local_mem[i + local_idx * opt.dims] = acc_data[i + idx * opt.dims]; // AoS
            }

            item_id.barrier(sycl::access::fence_space::local_space);

            for (integer_t hash_table = 0; hash_table < opt.number_of_hash_tables; ++hash_table) {
                integer_t combined_hash = opt.number_of_hash_functions;
                for (std::size_t j = 0; j < opt.number_of_hash_functions; ++j) {
                    real_t hash = acc_hash_func[opt.dims + j * (opt.dims + 1)];
                    for (std::size_t d = 0; d < opt.dims; ++d) {
                        //                        hash += local_mem[local_idx + d * local_size] * acc_hash_func[d + j * (opt.dims + 1)]; // SoA
                        hash += local_mem[d + local_idx * opt.dims] * acc_hash_func[d + j * (opt.dims + 1)]; // Aos
                    }
                    combined_hash ^= static_cast<integer_t>(hash / opt.w) + 0x9e3779b9 + (combined_hash << 6u) + (combined_hash >> 2u);
                }
                combined_hash %= opt.prim;
                acc_count.fetch_add(1, combined_hash + hash_table * opt.prim);
            }
        });
    });
}


void hash_tables::init_hash_functions(const options& opt) {
//    std::random_device rnd_device;
//    std::mt19937 rnd_normal_gen(rnd_device());
//    std::mt19937 rnd_uniform_gen(rnd_device());
    std::mt19937 rnd_normal_gen;
    std::mt19937 rnd_uniform_gen;
    std::normal_distribution<real_t> rnd_normal_dist;
    std::uniform_real_distribution<real_t> rnd_uniform_dist(0, opt.w);

    auto acc = hash_functions_.template get_access<sycl::access::mode::discard_write>();
    for (integer_t hash_table = 0; hash_table < opt.number_of_hash_tables; ++hash_table) {
        for (integer_t hash_function = 0; hash_function < opt.number_of_hash_functions; ++hash_function) {
            for (integer_t d = 0; d < opt.dims; ++d) {
                acc[this->get_linear_idx<layout>(hash_table, hash_function, d, opt)] = std::abs(rnd_uniform_dist(rnd_normal_gen));
            }
            acc[this->get_linear_idx<layout>(hash_table, hash_function, opt.dims, opt)] = rnd_uniform_dist(rnd_uniform_gen);
        }
    }
}

