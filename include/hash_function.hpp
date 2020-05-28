/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-28
 *
 * @brief Implements the @ref hash_functions class representing the used LSH hash functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP

#include <cmath>
#include <random>
#include <type_traits>

#include <config.hpp>
#include <data.hpp>
#include <options.hpp>


/**
 * @brief Class representing the LSH hash functions.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class hash_functions {
public:
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;
    /// The used type of a hash value.
    using hash_value_type = typename Options::hash_value_type;


    /// The SYCL buffer holding all hash functions: `buffer.get_count() == options::num_hash_tables * options::num_hash_functions * (data::dims + 1)`.
    sycl::buffer<real_type, 1> buffer;


    /**
     * @brief Calculates the hash value of the data point @p point in hash table @p hash_table.
     * @tparam AccData the type of data set accessor
     * @tparam AccHashFunction the type of the hash functions accessor
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] acc_data the data set accessor
     * @param[in] acc_hash_function the hash functions accessor
     * @return the hash value (`[[nodiscard]]`)
     *
     * @pre @p hash_table **must** be greater or equal than `0` and less than `options::num_hash_tables`
     * @pre @p point **must** be greater or equal than `0` and less than `data::size`
     */
    template <typename AccData, typename AccHashFunction>
    [[nodiscard]] hash_value_type hash(const index_type hash_table, const index_type point,
            AccData& acc_data, AccHashFunction& acc_hash_function)
    {
        DEBUG_ASSERT(0 <= hash_table && hash_table < opt_.num_hash_tables,
                "Out-of-bounce access!: 0 <= {} < {}", hash_table, opt_.num_hash_tables);
        DEBUG_ASSERT(0 <= point && point < data_.size,
                "Out-of-bounce access!: 0 <= {} < {}", point, data_.size);

        hash_value_type combined_hash = opt_.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt_.num_hash_functions; ++hash_function) {
            real_type hash = acc_hash_function[this->get_linear_id(hash_table, hash_function, data_.dims)];
            for (index_type dim = 0; dim < data_.dims; ++dim) {
                hash += acc_data[data_.get_linear_id(point, dim)] * acc_hash_function[this->get_linear_id(hash_table, hash_function, dim)];
            }
            combined_hash ^= static_cast<hash_value_type>(hash / opt_.w)
                    + static_cast<hash_value_type>(0x9e3779b9)
                    + (combined_hash << static_cast<hash_value_type>(6))
                    + (combined_hash >> static_cast<hash_value_type>(2));
        }
        if constexpr (std::is_signed_v<hash_value_type>) {
            combined_hash = sycl::fabs(combined_hash);
        }
        return combined_hash %= opt_.hash_table_size;
    }

    /**
     * @brief Returns the current hash functions with `new_layout`.
     * @details If `new_layout == layout` a compiler warning is issued.
     * @tparam new_layout the layout of the hash functions
     * @return the hash functions with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] hash_functions<new_layout, Options, Data> get_as()
//            __attribute__((diagnose_if(new_layout == layout, "new_layout == layout (simple copy)", "warning")))
    {
        hash_functions<new_layout, Options, Data> new_hash_functions(opt_, data_, false);
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

    /**
     * @brief Converts a three-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] hash_table the provided hash table
     * @param[in] hash_function the provided hash function
     * @param[in] dim the provided dimension
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p hash_table **must** be greater or equal than `0` and less than `options::num_hash_tables`.
     * @pre @p hash_function **must** be greater or equal than `0` and less than `options::num_hash_functions`.
     * @pre @p dim **must** be greater or equal than `0` and less than `data::dims + 1`.
     */
    [[nodiscard]] constexpr index_type get_linear_id(const index_type hash_table, const index_type hash_function,
            const index_type dim) const noexcept
    {
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

    /**
     * @brief Returns the @ref options object which has been used to create this @ref hash_functions object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    /**
     * @brief Returns the @ref data object which has been used to create this @ref hash_functions object.
     * @return the @ref data object (`[[nodiscard]]`)
     */
    [[nodiscard]] Data& get_data() const noexcept { return data_; }
    /**
     * @brief Returns the specified @ref memory_layout (*Array of Structs* or *Struct of Arrays*).
     * @return the specified @ref memory_layout (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept { return layout; }

private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Data_>
    friend hash_functions<layout_, typename Data_::options_type, Data_> make_hash_functions(Data_&);
    /// Befriend hash_functions class (including the one with another @ref memory_layout).
    template <memory_layout, typename, typename>
    friend class hash_functions;


    /**
     * @brief Construct new hash functions given the options in @p opt and sizes in @p data.
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] init `true` if the @ref buffer should be initialized, `false` otherwise
     */
    hash_functions(const Options& opt, Data& data, const bool init = true)
        : buffer(opt.num_hash_tables * opt.num_hash_functions * (data.dims + 1)), opt_(opt), data_(data)
    {
        START_TIMING(creating_hash_functions);
        if (init) {
            // TODO 2020-05-07 19:03 marcel: uncomment for truly random numbers
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
        END_TIMING(creating_hash_functions);
    }

    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;

};


/**
 * @brief Factory function for creating a new @ref hash_functions object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[in] data the used data object
 * @return the newly constructed @ref hash_functions object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline hash_functions<layout, typename Data::options_type, Data> make_hash_functions(Data& data) {
    return hash_functions<layout, typename Data::options_type, Data>(data.get_options(), data);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
