/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-29
 *
 * @brief Implements the @ref data class representing the used data set.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include <config.hpp>
#include <detail/assert.hpp>
#include <detail/convert.hpp>
#include <file_parser/parser_factory.hpp>
#include <options.hpp>


namespace detail {
    /**
     * @brief Empty base class for the @ref data class. Only for static_asserts.
     */
    class data_base {};
}


/**
 * @brief Class representing a data set.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options>
class data : detail::data_base {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;
    /// The type of the provided @ref options class.
    using options_type = Options;


    /// The number of data points.
    const index_type size;
    /// The dimension of each data point.
    const index_type dims;
    /// The SYCL buffer holding all data: `buffer.get_count() == size * dims`.
    sycl::buffer<real_type, 1> buffer;


    /**
     * @brief Returns the current data set with `new_layout`.
     * @details If `new_layout == layout` a compiler warning is issued (currently disabled).
     * @tparam new_layout the layout of the data set
     * @return the data set with the `new_layout` (`[[nodiscard]]`)
     */
    template <memory_layout new_layout>
    [[nodiscard]] data<new_layout, Options> get_as()
//            __attribute__((diagnose_if(new_layout == layout, "new_layout == layout (simple copy)", "warning")))
    {
        data<new_layout, Options> new_data(opt_, size, dims);
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
    [[nodiscard]] constexpr index_type get_linear_id(const index_type point, const index_type dim) const noexcept {
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
     * @brief Returns the @ref options object which has been used to create this @ref data object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept {
        return opt_;
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
    template <memory_layout layout_, typename Options_>
    friend data<layout_, Options_> make_data(const Options_&, const std::string&);
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_>
    friend data<layout_, Options_> make_data(const Options_&, typename Options_::index_type, typename Options_::index_type);
    /// Befriend data class (including the one with another @ref memory_layout).
    template <memory_layout, typename>
    friend class data;


    /**
     * @brief Construct a new data object if size: `size * dims`.
     * @details Initialize the buffer with iota values if @p init is set to `true` (default: `true`).
     * @param[in] opt the provided @ref options object
     * @param[in] size the number of data points
     * @param[in] dims the number of dimensions of each data point
     *
     * @pre @p size **must** be greater than `0`.
     * @pre @p dims **must** be greater than `0`.
     */
    data(const Options& opt, const index_type size, const index_type dims)
            : size(size), dims(dims), buffer(size * dims), opt_(opt)
    {
        DEBUG_ASSERT(0 <= size, "Illegal size!: {}", size);
        DEBUG_ASSERT(0 <= dims, "Illegal number of dimensions!: {}", dims);

        // fill "iota" like
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        real_type val = 0.0;
        for (index_type point = 0; point < size; ++point) {
            for (index_type dim = 0; dim < dims; ++dim) {
                acc[this->get_linear_id(point, dim)] = val++;
            }
        }
    }
    /**
     * @brief Construct a new data object from the given @p file.
     * @param[in] opt the provided @ref options object
     * @param[in] file the file containing all data points
     *
     * @throw std::invalid_argument if @p file doesn't exist.
     *
     * @pre the number of data points in @p file **must** be greater than `0`.
     * @pre the dimension of the data points in @p file **must** be greater than `0`.
     */
    data(const Options& opt, std::unique_ptr<file_parser<layout, Options>> parser)
            : size(parser->parse_size()), dims(parser->parse_dims()), buffer(size * dims), opt_(opt)
    {
        DEBUG_ASSERT(0 <= size, "Illegal size!: {}", size);
        DEBUG_ASSERT(0 <= dims, "Illegal number of dimensions!: {}", dims);

        START_TIMING(reading_data_file);
        parser->parse_content(buffer, size, dims);
        END_TIMING(reading_data_file);
    }

    /**
     * @brief Print the size and dims of the data set @p data.
     * @param[inout] out the output stream to print @p opt
     * @param[in] data the data set
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& out, const data& data) {
        out << "size " << data.size << '\n';
        out << "dims " << data.dims;
        return out;
    }

    /// Const reference to @ref options object.
    const Options& opt_;
};


/**
 * @brief Factory function for creating a new @ref data object from a @p size and @p dims.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @param[in] opt the used @ref options object
 * @param[in] size the number of data points
 * @param[in] dims the number of dimensions per data point
 * @return the newly constructed @ref data object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Options>
[[nodiscard]] inline data<layout, Options> make_data(const Options& opt, const typename Options::index_type size, const typename Options::index_type dims) {
    return data<layout, Options>(opt, size, dims);
}

/**
 * @brief Factory function for creating a new @ref data object from a @p file.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @param[in] opt the used @ref options object
 * @param[in] file the @p file from which the data should get loaded
 * @return the newly constructed @ref data object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Options>
[[nodiscard]] inline data<layout, Options> make_data(const Options& opt, const std::string& file) {
    return data<layout, Options>(opt, file_parser_factory<layout, Options>(file));
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP