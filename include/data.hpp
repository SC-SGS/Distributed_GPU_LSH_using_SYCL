/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-05
 *
 * @brief Implements the @ref data class representing the used data set.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <config.hpp>
#include <detail/convert.hpp>
#include <options.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>


/**
 * @brief Class representing a data set.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options>
class data {
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using data_type = typename Options::real_type;
    /// The size type as specified as in the provided @ref options class.
    using size_type = typename Options::size_type;

    /// The number of data points.
    const size_type size;
    /// The dimension of each data point.
    const size_type dims;
    /// The SYCL buffer holding all data. `buffer.get_count() == size * dims`
    sycl::buffer<data_type, 1> buffer;

    /// Delete copy-constructor because SYCL only does a shallow-copy.
    data(const data&) = delete;
    /// Delete copy-assignment operator because SYCL only does a shallow-copy.
    data& operator=(const data&) = delete;
    /// Enable default move-constructor.
    data(data&&) = default;
    /// Enable default move-assignment operator.
    data& operator=(data&&) = default;


    /**
     * @brief Returns the current data set with `new_layout`.
     * @details If `new_layout == layout` a compiler warning is issued.
     * @tparam new_layout the layout of the data set
     * @return the data set with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] data<new_layout, Options> get_as()
    __attribute__((diagnose_if(new_layout == layout,
            "get_as called with same memory_layout as *this -> results in a copy of *this -> better use *this in the first place",
            "warning")))
    {
        data<new_layout, Options> new_data(size, dims);
        auto acc_this = buffer.template get_access<sycl::access::mode::read>();
        auto acc_new = new_data.buffer.template get_access<sycl::access::mode::discard_write>();
        for (size_type s = 0; s < size; ++s) {
            for (size_type d = 0; d < dims; ++d) {
                // transform memory layout
                acc_new[new_data.get_linear_id(s, d)] = acc_this[this->get_linear_id(s, d)];
            }
        }
        return new_data;
    }

    /**
     * @brief Converts a two-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] point the provided data point
     * @param[in] dim the provided dimension
     * @return the flattened index (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type get_linear_id(const size_type point, const size_type dim) const noexcept {
        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return dim + point * dims;
        } else {
            // Struct of Arrays
            return point + dim * size;
        }
    }
    /**
     * @brief Returns the specified @ref memory_layout (*Array of Structs* or *Struct of Arrays*).
     * @return the specified @ref memory_layout (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept {
        return layout;
    }


private:
    template <memory_layout layout_, typename Options_, typename... Args_>
    friend data<layout_, Options_> make_data(const Options_& opt, Args_&&... args);
    template <memory_layout, typename>
    friend class data;

    data(const std::size_t size, const std::size_t dims) : size(size), dims(dims), buffer(sycl::range<1>{ size * dims }) { }

    data(const std::size_t size, const std::size_t dims, const Options&)
            : size(size), dims(dims), buffer(sycl::range<1>{ size * dims })
    {
        std::random_device rnd_device;
        std::mt19937 rnd_gen(rnd_device());
        std::normal_distribution<data_type> rnd_dist;

        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (size_type i = 0; i < buffer.get_count(); ++i) {
            acc[i] = rnd_dist(rnd_gen);
        }
    }
    data(const std::string& file, const Options& opt)
            : size(this->parse_size(file)), dims(this->parse_dims(file)), buffer(sycl::range<1>{ size * dims })
    {
        std::ifstream in(file);
        std::string line, elem;

        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (size_type point = 0; point < size; ++point) {
            std::getline(in, line);
            std::stringstream ss(line);
            for (size_type dim = 0; dim < dims; ++dim) {
                std::getline(ss, elem, ',');
                acc[point + dim * size] = detail::convert_to<data_type>(elem);
            }
        }
    }


    [[nodiscard]] size_type parse_size(const std::string& file) const {
        std::ifstream in(file);
        return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
    }
    [[nodiscard]] size_type parse_dims(const std::string& file) const {
        if (size == 0) return 0;

        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        return std::count(line.cbegin(), line.cend(), ',') + 1;
    }
};


/**
 * @brief Factory function for creating a new @ref data object.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @tparam Args the types of the additional constructor parameters
 * @param opt the option class
 * @param args additional constructor parameters
 * @return the newly constructed @ref data object
 */
template <memory_layout layout, typename Options, typename... Args>
data<layout, Options> make_data(const Options& opt, Args&&... args) {
    return data<layout, Options>(std::forward<Args>(args)..., opt);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
