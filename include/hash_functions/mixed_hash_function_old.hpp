#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MIXED_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MIXED_HASH_FUNCTION_HPP

#include <config.hpp>
#include <data.hpp>
#include <detail/mpi_type.hpp>
#include <detail/timing.hpp>
#include <options.hpp>

#include <mpi.h>

#include <vector>
#include <iostream>
#include <type_traits>
#include <algorithm>

template <memory_layout layout, typename Options, typename Data>
class mixed_hash_function : detail::hash_functions_base {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
    static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must by a 'data' type!");
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;
    using hash_value_type = typename Options::hash_value_type;
    using options_type = Options;
    using data_type = Data;

    sycl::buffer<real_type, 1> buffer;

    template <typename AccData, typename AccHashFunctions>
    [[nodiscard]]
    static constexpr hash_value_type hash([[maybe_unused]] const int comm_rank,
                                          const index_type hash_table, const index_type point,
                                          AccData& acc_data, AccHashFunctions& acc_hash_functions,
                                          const Options& opt, const Data& data)
    {
        const index_type hash_table_random_projections_offset =
                hash_table * (opt.num_hash_functions * (data.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
        const index_type hash_table_entropy_based_combine_offset =
                hash_table_random_projections_offset + opt.num_hash_functions * (data.dims + 1);
        
        real_type value = 0.0;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            real_type hash = acc_hash_functions[hash_table_random_projections_offset + hash_function * (data.dims + 1) + data.dims];
            for (index_type dim = 0; dim < data.dims; ++dim) {
                hash += acc_data[data_type::get_linear_id(comm_rank, point, data.rank_size, dim, data.dims)] *
                        acc_hash_functions[hash_table_random_projections_offset + hash_function * (data.dims + 1) + dim];
            }
            value += static_cast<hash_value_type>(hash / opt.w) * acc_hash_functions[hash_table_entropy_based_combine_offset + hash_function];
        }
        hash_value_type combined_hash = 0;
        for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
            combined_hash += value > acc_hash_functions[hash_table_entropy_based_combine_offset + opt.num_hash_functions + cop];
        }
        return combined_hash % opt.hash_table_size;
    }

    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    [[nodiscard]] Data& get_data() const noexcept { return data_; }

private:
    template <memory_layout layout_, typename Data_>
    friend auto make_mixed_hash_functions(Data_&, const MPI_Comm&);

    mixed_hash_function(const Options& opt, Data& data, std::vector<real_type>& tmp_buffer, const int comm_rank)
        : buffer(tmp_buffer.size()), comm_rank_(comm_rank), opt_(opt), data_(data)
    {
        auto acc = buffer.template get_access<sycl::access::mode::write>();
        for (index_type i = 0; i < tmp_buffer.size(); ++i) {
            acc[i] = tmp_buffer[i];
        }
    }


    const int comm_rank_;
    const Options& opt_;
    Data& data_;
};


template <memory_layout layout, typename Data>
[[nodiscard]]
inline auto make_mixed_hash_functions(Data& data, const MPI_Comm& communicator) {
    using data_type = Data;
    using options_type = typename Data::options_type;
    using real_type = typename options_type::real_type;
    using index_type = typename options_type::index_type;
    using hash_value_type = typename options_type::hash_value_type;
    using hash_functions_type = mixed_hash_function<layout, options_type, Data>;

    START_TIMING(creating_hash_functions);

    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);

    options_type opt = data.get_options();

    // random projection hash functions + entropy based hash functions
    const index_type num_random_projection_values = opt.num_hash_tables * opt.num_hash_functions * (data.dims + 1);
    const index_type num_entropy_based_combine_values = opt.num_hash_tables * (opt.num_hash_functions + opt.num_cut_off_points - 1);
    std::vector<real_type> buffer(num_random_projection_values + num_entropy_based_combine_values);

    {
        // currently only one MPI rank supported
        int comm_size;
        MPI_Comm_size(communicator, &comm_size);
        if (comm_size != 1) {
            throw std::logic_error("Currenlty only one MPI rank supported!");
        }
    }

    std::mt19937 rnd_normal_gen;   // random projections + entropy based
    std::mt19937 rnd_uniform_gen;  // random projections only
    std::normal_distribution<real_type> rnd_normal_dist;
    std::uniform_real_distribution<real_type> rnd_uniform_dist(0, opt.w);

    sycl::queue queue(sycl::default_selector{});
//        sycl::queue queue(single_gpu_selector(communicator)); // // TODO 2020-08-31 16:31 marcel: change

    for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
        // calculate indexing offsets
        const index_type hash_table_random_projections_offset =
                hash_table * (opt.num_hash_functions * (data.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
        const index_type hash_table_entropy_based_combine_offset =
                hash_table_random_projections_offset + opt.num_hash_functions * (data.dims + 1);
        
        // generate random projections hash functions for current hash table
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            for (index_type dim = 0; dim < data.dims; ++dim) {
                buffer[hash_table_random_projections_offset + hash_function * (data.dims + 1) + dim] = std::abs(rnd_normal_dist(rnd_normal_gen));
            }
            buffer[hash_table_random_projections_offset + hash_function * (data.dims + 1) + data.dims] = rnd_uniform_dist(rnd_uniform_gen);
        }

        // generate entropy based hash functions for current hash table
        for (index_type combine_hash = 0; combine_hash < opt.num_hash_functions; ++combine_hash) {
            buffer[hash_table_entropy_based_combine_offset + combine_hash] = rnd_normal_dist(rnd_normal_gen);
        }

        // calculate hash values for cut off points
        std::vector<real_type> hash_values(data.rank_size, 0.0);
        {

            sycl::buffer<real_type, 1> hash_functions_buffer(buffer.data(), buffer.size());
            sycl::buffer<real_type, 1> hash_values_buffer(hash_values.data(), hash_values.size());

            queue.submit([&](sycl::handler& cgh) {
                auto acc_data = data.buffer.template get_access<sycl::access::mode::read>(cgh);
                auto acc_hash_functions = hash_functions_buffer.template get_access<sycl::access::mode::read>(cgh);
                auto acc_hash_values = hash_values_buffer.template get_access<sycl::access::mode::write>(cgh);
                auto data_ = data;
                auto opt_ = opt;

                cgh.parallel_for<class mixed_hash_values>(sycl::range<>(data.rank_size), [=](sycl::item<> item) {
                    const index_type idx = item.get_linear_id();

                    real_type value = 0.0;
                    for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
                        real_type hash = acc_hash_functions[hash_table_random_projections_offset + hash_function * (data_.dims + 1) + data_.dims];
                        for (index_type dim = 0; dim < data_.dims; ++dim) {
                            hash += acc_data[data_type::get_linear_id(comm_rank, idx, data_.rank_size, dim, data_.dims)]
                                    * acc_hash_functions[hash_table_random_projections_offset + hash_function * (data_.dims + 1) + dim];
                        }
                        value += static_cast<hash_value_type>(hash / opt_.w)
                                * acc_hash_functions[hash_table_entropy_based_combine_offset + hash_function];
                    }
                    acc_hash_values[idx] = value;
                });
            });
        }

        // sort hash values
        std::sort(hash_values.begin(), hash_values.end());

        // calculate cut off points
        const index_type jump = data.rank_size / opt.num_cut_off_points;
        for (index_type i = 0; i < opt.num_cut_off_points - 1; ++i) {
            buffer[hash_table_entropy_based_combine_offset + opt.num_hash_functions + i] = hash_values[(i + 1) * jump];
        }
    }

    END_TIMING_MPI(creating_hash_functions, comm_rank);
    return hash_functions_type(opt, data, buffer, comm_rank);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MIXED_HASH_FUNCTION_HPP