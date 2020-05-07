/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-07
 *
 * @brief Implements the @ref hash_functions class representing the used LSH hash functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP

#include <random>

#include <config.hpp>
#include <options.hpp>


/**
 * @brief Class representing the LSH hash functions.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options, typename Data>
class hash_functions {
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;
    using hash_value_type = typename Options::hash_value_type;
    using options_type = Options;
    using data_type = Data;


    sycl::buffer<real_type, 1> buffer;


    template <typename AccData, typename AccHashFunction>
    [[nodiscard]] hash_value_type hash(const index_type hash_table, const index_type point, AccData& acc_data, AccHashFunction& acc_hash_function) const {
        hash_value_type combined_hash = opt_.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
            real_type hash = acc_hash_function[this->get_linear_id(hash_table, hash_function, data_.dims)];
            for (index_type dim = 0; dim < data_.dims; ++dim) {
                hash += acc_data[data_.get_linear_id(point, dim)] * acc_hash_function[this->get_linear_id(hash_table, hash_function, dim)];
            }
            combined_hash ^= static_cast<hash_value_type>(hash / opt_.w)
                    + 0x9e3779b9
                    + (combined_hash << static_cast<hash_value_type>(6))
                    + (combined_hash >> static_cast<hash_value_type>(2));
        }
        if constexpr (!std::is_signed_v<hash_value_type>) {
            if (combined_hash < 0) {
                combined_hash *= -1;
            }
        }
        return combined_hash %= opt_.hash_table_size;
    }


    template <memory_layout new_layout>
    [[nodiscard]] hash_functions<new_layout, Options, Data> get_as()
    __attribute__((diagnose_if(new_layout == layout,
            "get_as called with same memory_layout as *this -> results in a copy of *this -> directly use *this",
            "warning")))
    {
        hash_functions<new_layout, Options, Data> new_hash_functions(data_, opt_, false);
        auto acc_this = buffer.template get_access<sycl::access::mode::read>();
        auto acc_new = new_hash_functions.buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
                for (index_type dim = 0; dim <= data_.dims; ++dim) {
                    // transform memory layout
                    acc_new[new_hash_functions.get_linear_id(hash_table, hash_function, dim)]
                        = acc_this[this->get_linear_id(hash_table, hash_function, dim)];
                }
            }
        }
        return new_hash_functions;
    }


    [[nodiscard]] constexpr index_type get_linear_id(const index_type hash_table, const index_type hash_function, const index_type dim) const noexcept {
        DEBUG_ASSERT(0 <= hash_table && hash_table < opt_.num_hash_tables,
                "Out-of-bounce access!: 0 <= {} < {}", hash_table, opt_.num_hash_tables);
        DEBUG_ASSERT(0 <= hash_function && hash_function < opt_.num_hash_functions,
                "Out-of-bounce access!: 0 <= {} < {}", hash_function, opt_.num_hash_functions);
        DEBUG_ASSERT(0 <= dim && dim < data_.dims + 1,
                "Out-of-bounce access!: 0 <= {} < {}", dim, data_.dims + 1);

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return hash_table * opt_.num_hash_functions * (data_.dims + 1) + hash_function * (data_.dims + 1) + dim;
        } else {
            // Struct of Arrays
            return hash_table * opt_.num_hash_functions * (data_.dims + 1) + dim * opt_.num_hash_functions + hash_function;
        }
    }

    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept {
        return layout;
    }

private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_, typename Data_>
    friend hash_functions<layout_, Options_, Data_> make_hash_functions(const Options_&, const Data_&);


    hash_functions(const Data& data, const Options& opt, const bool init = true)
        : opt_(opt), data_(data), buffer(opt.num_hash_tables * opt.num_hash_functions * (data.dims + 1))
    {
        if (init) {
//        std::random_device rnd_device;
//        std::mt19937 rnd_normal_gen(rnd_device());
//        std::mt19937 rnd_uniform_gen(rnd_device());
            std::mt19937 rnd_normal_gen;
            std::mt19937 rnd_uniform_gen;
            std::normal_distribution<real_type> rnd_normal_dist;
            std::uniform_real_distribution<real_type> rnd_uniform_dist(0, opt.w);

            auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
            for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
                for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
                    for (index_type dim = 0; dim < data_.dims; ++dim) {
                        acc[this->get_linear_id(hash_table, hash_function, dim)] = std::abs(rnd_uniform_dist(rnd_normal_gen));
                    }
                    acc[this->get_linear_id(hash_table, hash_function, data_.dims)] = rnd_uniform_dist(rnd_uniform_gen);
                }
            }
        }
    }


    const options_type& opt_;
    const data_type& data_;

};


template <memory_layout layout, typename Options, typename Data>
inline hash_functions<layout, Options, Data> make_hash_functions(const Options& opt, const Data& data) {
    return hash_functions<layout, Options, Data>(data, opt);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
