#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP

#include <config.hpp>
#include <options.hpp>
#include <CL/sycl.hpp>


class hash_tables {
public:

    hash_tables(sycl::queue& queue, sycl::buffer<real_t, 1>& data, const options& opt);

    template <memory_type layout>
    integer_t get_linear_idx(const integer_t hash_table, const integer_t hash_function, const integer_t dim, const options& opt) noexcept {
        if constexpr (layout == memory_type::aos) {
            // grouped by hash_tables -> grouped by hash_functions
            return hash_table * (opt.number_of_hash_functions * (opt.dims + 1))
                   + hash_function * (opt.dims + 1)
                   + dim;
        } else {
            // grouped by hash_tables -> grouped by dimensions
            return hash_table * (opt.number_of_hash_functions * (opt.dims + 1))
                   + dim * opt.number_of_hash_functions
                   + hash_function;
        }
    }

//private:
    sycl::buffer<real_t, 1> hash_functions_;
    sycl::buffer<real_t, 1> hash_tables_;

private:
    void init_hash_functions(const options& opt);
};





#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
