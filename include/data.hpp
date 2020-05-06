/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-06
 *
 * @brief Implements the @ref data class representing the used data set.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>

#include <config.hpp>
#include <detail/assert.hpp>
#include <detail/convert.hpp>
#include <options.hpp>


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
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /// The number of data points.
    const index_type size;
    /// The dimension of each data point.
    const index_type dims;
    /// The SYCL buffer holding all data. `buffer.get_count() == size * dims`
    sycl::buffer<data_type, 1> buffer;


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
        for (index_type s = 0; s < size; ++s) {
            for (index_type d = 0; d < dims; ++d) {
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
     *
     * @pre @p point **must** be greater or equal than `0` and less than `size`.
     * @pre @p dim **must** be greater or equal than `0` and less than `dims`.
     */
    [[nodiscard]] index_type get_linear_id(const index_type point, const index_type dim) const noexcept {
        DEBUG_ASSERT(0 <= point && point < size, "Out-of-bounce access!: 0 <= {} < {}", point, size);
        DEBUG_ASSERT(0 <= dim && dim < dims, "Out-of-bounce access!: 0 <= {} < {}", dim, dims);

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
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_, typename... Args_>
    friend data<layout_, Options_> make_data(const Options_& opt, Args_&&... args);
    /// Befriend data class (including the one with another @ref memory_layout).
    template <memory_layout, typename>
    friend class data;

    /**
     * @brief Default construct a data object of size: `size * dims`.
     * @details **Doesn't** initialize the buffer.
     * @param[in] size the number of data points
     * @param[in] dims the number of dimensions of each data point
     *
     * @pre @p size **must** be greater than `0`.
     * @pre @p dims **must** be greater than `0`.
     */
    data(const index_type size, const index_type dims) : size(size), dims(dims), buffer(sycl::range<1>{ size * dims }) {
        DEBUG_ASSERT(0 < size, "Illegal size!: {}", size);
        DEBUG_ASSERT(0 < dims, "Illegal number of dimensions!: {}", dims);
    }
    /**
     * @brief Construct a new data object if size: `size * dims`.
     * @details **Does** initialize the buffer with random values.
     * @param[in] size the number of data points
     * @param[in] dims the number of dimensions of each data point
     *
     * @pre @p size **must** be greater than `0`.
     * @pre @p dims **must** be greater than `0`.
     */
    data(const index_type size, const index_type dims, const Options&) : data(size, dims) {
        // define random facilities
        std::random_device rnd_device;
        std::mt19937 rnd_gen(rnd_device());
        std::normal_distribution<data_type> rnd_dist;

        // memory_layout doesn't matter for random values
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type i = 0; i < buffer.get_count(); ++i) {
            acc[i] = rnd_dist(rnd_gen);
        }
    }
    /**
     * @brief Construct a new data object from the given @p file.
     * @param[in] file the file containing all data points
     *
     * @throw std::invalid_argument if @p file doesn't exist.
     *
     * @pre the number of data points in @p file **must** be greater than `0`.
     * @pre the dimension of the data points in @p file **must** be greater than `0`.
     */
    data(const std::string& file, const Options&) : data(this->parse_size(file), this->parse_dims(file)) {
        // check if file exists
        if (!std::filesystem::exists(file)) {
            throw std::invalid_argument("File '" + file + "' doesn't exist!");
        }
        std::ifstream in(file);
        std::string line, elem;

        // read file line by line, parse value and save it at the correct position (depending on the current memory_layout) in buffer
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type point = 0; point < size; ++point) {
            std::getline(in, line);
            std::stringstream ss(line);
            for (index_type dim = 0; dim < dims; ++dim) {
                std::getline(ss, elem, ',');
                acc[this->get_linear_id(point, dim)] = detail::convert_to<data_type>(elem);
            }
        }
    }

    /**
     * @brief Computes the number of data points in the given @p file.
     * @param[in] file the file containing all data points
     * @return the number of data points in @p file (`[[nodiscard]]`)
     */
    [[nodiscard]] index_type parse_size(const std::string& file) const {
        std::ifstream in(file);
        return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
    }
    /**
     * @brief Computes the number of dimensions of each data point in the given @p file.
     * @param[in] file the file containing all data points
     * @return the number of dimensions (`[[nodiscard]]`)
     */
    [[nodiscard]] index_type parse_dims(const std::string& file) const {
        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        if (line.empty()) {
            return 0;
        } else {
            return std::count(line.cbegin(), line.cend(), ',') + 1;
        }
    }
};


/**
 * @brief Factory function for creating a new @ref data object.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @tparam Args the types of the additional constructor parameters
 * @param[in] opt the option class
 * @param[in] args additional constructor parameters
 * @return the newly constructed @ref data object
 */
template <memory_layout layout, typename Options, typename... Args>
inline data<layout, Options> make_data(const Options& opt, Args&&... args) {
    return data<layout, Options>(std::forward<Args>(args)..., opt);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
