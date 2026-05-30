/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines a matrix class used to hiding the data linearization using AoS and SoA.
 */

#ifndef SYCL_LSH_DETAIL_MATRIX_HPP
#define SYCL_LSH_DETAIL_MATRIX_HPP
#pragma once

#include "sycl_lsh/detail/assert.hpp"          // SYCL_LSH_ASSERT
#include "sycl_lsh/detail/shape.hpp"           // sycl_lsh::shape
#include "sycl_lsh/detail/utility.hpp"         // sycl_lsh::detail::{always_false_v, unreachable}
#include "sycl_lsh/exceptions/exceptions.hpp"  // sycl_lsh::matrix_exception
#include "sycl_lsh/memory_layout.hpp"          // sycl_lsh::memory_layout

#include "fmt/base.h"     // fmt::formatter
#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/format.h"   // fmt::format, fmt::runtime
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <algorithm>    // std::equal, std::all_of, std::fill_n
#include <cstddef>      // std::size_t
#include <cstring>      // std::memcpy, std::memset
#include <iosfwd>       // std::istream forward declaration
#include <ostream>      // std::ostream
#include <type_traits>  // std::enable_if, std::is_convertible_v, std::is_arithmetic_v
#include <utility>      // std::swap
#include <vector>       // std::vector

namespace sycl_lsh {

/**
 * @brief A matrix class encapsulating a 1D array automatically handling indexing with AoS and SoA schemes.
 * @attention If a three-dimensional shape is provided, the last dimension (z) is ignored!
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 */
template <typename T, memory_layout layout_>
class matrix {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type!");

  public:
    /// The value type of the entries in this matrix.
    using value_type = T;
    /// The size type used in this matrix.
    using size_type = std::size_t;
    /// The reference type of the entries in this matrix (based on the value type).
    using reference = value_type &;
    /// The const reference type of the entries in this matrix (based on the value type).
    using const_reference = const value_type &;
    /// The pointer type of the entries in this matrix (based on the value type).
    using pointer = value_type *;
    /// The const pointer type of the entries in this matrix (based on the value type).
    using const_pointer = const value_type *;

    /**
     * @brief Default construct an empty matrix, i.e., zero rows and columns.
     */
    matrix() = default;

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and default-initializes all values.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    explicit matrix(detail::shape shape);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and default-initializes all values.
     * @details The padding entries are always initialized to `0`!
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    matrix(detail::shape shape, detail::shape padding);

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize all entries with the value @p init.
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] init the value of all entries in the matrix
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    matrix(detail::shape shape, const U &init);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and initialize all valid entries with the value @p init.
     * @details The padding entries are always initialized to `0`!
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] init the value of all entries in the matrix
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    matrix(detail::shape shape, const U &init, detail::shape padding);

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the data values
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero
     * @throws sycl_lsh::matrix_exception if @p shape.x times @p shape.y is not equal to the number of values in @p data
     */
    matrix(detail::shape shape, const std::vector<value_type> &data);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @note The vector @p data must **not** contain padding entries!
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the data values
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero
     * @throws sycl_lsh::matrix_exception if @p shape.x times @p shape.y is not equal to the number of values in @p data
     */
    matrix(detail::shape shape, const std::vector<value_type> &data, detail::shape padding);

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the pointer to the data values
     * @throws sycl_lsh::matrix_exception if exactly one of @p shape.x or @p shape.y is zero
     */
    matrix(detail::shape shape, const_pointer data);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @note The data pointed to by @p data must **not** contain padding entries!
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the pointer to the data values
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws sycl_lsh::matrix_exception if exactly one of the @p shape values is zero
     */
    matrix(detail::shape shape, const_pointer data, detail::shape padding);

    /**
     * @brief Construct a new matrix from @p other. Respects potential different layout types.
     * @tparam other_layout_ the layout_type of the other matrix
     * @param[in] other the other matrix
     */
    template <memory_layout other_layout_>
    explicit matrix(const matrix<T, other_layout_> &other);
    /**
     * @brief Construct a new matrix from @p other with the new padding sizes @p row_padding and @p col_padding. Respects potential different layout types.
     * @tparam other_layout_ the layout_type of the other matrix
     * @param[in] other the other matrix
     * @param[in] padding the padding of the new matrix, i.e., the number of padding entries for each row and column
     */
    template <memory_layout other_layout_>
    matrix(const matrix<T, other_layout_> &other, detail::shape padding);

    /**
     * @brief Create a matrix from the provided 2D vector @p data.
     * @param[in] data the data used to initialize this matrix
     * @throws sycl_lsh::matrix_exception if the data vectors contain different number of values
     * @throws sycl_lsh::matrix_exception if one vector in the data vector is empty
     */
    explicit matrix(const std::vector<std::vector<value_type>> &data);
    /**
     * @brief Create a matrix from the provided 2D vector @p data including padding.
     * @note The two-dimensional data vector @p data must **not** contain padding entries!
     * @param[in] data the data used to initialize this matrix
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     */
    matrix(const std::vector<std::vector<value_type>> &data, detail::shape padding);

    /**
     * @brief Return the number of entries in the matrix **without** padding.
     * @details It holds: `size() == shape().x * shape().y`.
     * @return the number of entries **without** padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size() const noexcept { return shape_.x * shape_.y; }

    /**
     * @brief Returns the shape of the matrix, i.e., the number of rows and columns **without** padding.
     * @return the shape of the matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] detail::shape shape() const noexcept { return shape_; }

    /**
     * @brief Return the number of rows in the matrix **without** padding.
     * @return the number of rows (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows() const noexcept { return shape_.x; }

    /**
     * @brief Return the number of columns in the matrix **without** padding.
     * @return the number of columns (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols() const noexcept { return shape_.y; }

    /**
     * @brief Check whether the matrix is currently empty, i.e., has zero rows and columns.
     * @details This may only happen for a default initialized matrix or a matrix explicitly created with a shape of `{ 0, 0 }`.
     * @note A matrix with only padding entries is regarded as empty!
     * @return `true` if the matrix is empty, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept { return shape_.x == 0 && shape_.y == 0; }

    /**
     * @brief Return the padding sizes for the rows and columns.
     * @return the padding sizes (`[[nodiscard]]`)
     */
    [[nodiscard]] detail::shape padding() const noexcept { return padding_; }

    /**
     * @brief Return the number of entries in the matrix **including** padding.
     * @details It holds: `size_padded() == shape_padded().x * shape_padded().y`.
     * @return the number of entries **including** padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size_padded() const noexcept { return (shape_.x + padding_.x) * (shape_.y + padding_.y); }

    /**
     * @brief Returns the shape of the matrix **including** padding, i.e., the number of rows + row padding and columns + column padding.
     * @return the shape of the matrix **including** padding (`[[nodiscard]]`)
     */
    [[nodiscard]] detail::shape shape_padded() const noexcept { return detail::shape{ shape_.x + padding_.x, shape_.y + padding_.y }; }

    /**
     * @brief Return the number of rows in the matrix **including** padding.
     * @return the number of rows + row padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows_padded() const noexcept { return shape_.x + padding_.x; }

    /**
     * @brief Return the number of columns in the matrix **including** padding.
     * @return the number of columns + column padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols_padded() const noexcept { return shape_.y + padding_.y; }

    /**
     * @brief Checks whether this matrix contains any padding entries.
     * @return `true` if this matrix is padded, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_padded() const noexcept { return padding_.x != 0 || padding_.y != 0; }

    /**
     * @brief Restore the padding entries, i.e., explicitly set all padding entries to `0` again.
     */
    void restore_padding() noexcept;

    /**
     * @brief Return the layout type used in this matrix.
     * @details The layout type is either Array-of-Structs (AoS) or Struct-of-Arrays (SoA).
     * @return the layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static memory_layout layout() noexcept { return layout_; }

    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type operator()(size_type row, size_type col) const;
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference operator()(size_type row, size_type col);
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @throws sycl_lsh::matrix_exception if the provided @p row is equal or larger than the number of rows in the matrix
     * @throws sycl_lsh::matrix_exception if the provided @p col is equal or larger than the number of columns in the matrix
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type at(size_type row, size_type col) const;
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @throws sycl_lsh::matrix_exception if the provided @p row is equal or larger than the number of rows in the matrix
     * @throws sycl_lsh::matrix_exception if the provided @p col is equal or larger than the number of columns in the matrix
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference at(size_type row, size_type col);

    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type operator[](size_type idx) const;
    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference operator[](size_type idx);
    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @throws sycl_lsh::matrix_exception if the provided @p idx is equal or larger than the number of matrix entries
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type at(size_type idx) const;
    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @throws sycl_lsh::matrix_exception if the provided @p idx is equal or larger than the number of matrix entries
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference at(size_type idx);

    /**
     * @brief Return a pointer to the underlying one-dimensional data structure.
     * @return the one-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] pointer data() noexcept { return data_.data(); }

    /**
     * @brief Return a pointer to the underlying one-dimensional data structure.
     * @return the one-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] const_pointer data() const noexcept { return data_.data(); }

    /**
     * @brief Return the data as a 2D vector.
     * @return the two-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::vector<value_type>> to_2D_vector() const;
    /**
     * @brief Return the data as 2D vector including padding entries.
     * @return the two-dimensional data including padding (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::vector<value_type>> to_2D_vector_padded() const;

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other matrix to swap the entries from
     */
    void swap(matrix &other) noexcept;

  private:
    /**
     * @brief Copy the data from @p source to @p dest row- or column-wise depending on the current layout type.
     * @param[out] dest the destination buffer
     * @param[in] dest_shape the shape of the destination buffer
     * @param[in] source the source buffer
     * @param[in] source_shape the shape of the source buffer
     */
    void opt_mismatched_padding_copy(pointer dest, const detail::shape dest_shape, const_pointer source, const detail::shape source_shape) {
        if constexpr (layout_ == memory_layout::aos) {
// copy row-wise
#pragma omp parallel for
            for (size_type row = 0; row < this->num_rows(); ++row) {
                std::memcpy(dest + row * dest_shape.y, source + row * source_shape.y, this->num_cols() * sizeof(value_type));
            }
        } else if constexpr (layout_ == memory_layout::soa) {
// copy column-wise
#pragma omp parallel for
            for (size_type col = 0; col < this->num_cols(); ++col) {
                std::memcpy(dest + col * dest_shape.x, source + col * source_shape.x, this->num_rows() * sizeof(value_type));
            }
        } else {
            static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
        }
    }

    /// The shape of the matrix.
    detail::shape shape_;
    /// The shape of the padding for each row and column.
    detail::shape padding_;
    /// The (linearized, either in AoS or SoA layout) data.
    std::vector<value_type> data_{};
};

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const detail::shape shape) :
    matrix{ shape, value_type{} } { }

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const detail::shape shape, const detail::shape padding) :
    shape_{ shape },
    padding_{ padding },
    data_(this->size_padded(), value_type{}) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }
}

template <typename T, memory_layout layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const detail::shape shape, const U &init) :
    shape_{ shape },
    padding_{ 0, 0 },
    data_(this->size(), static_cast<value_type>(init)) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }
}

template <typename T, memory_layout layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const detail::shape shape, const U &init, const detail::shape padding) :
    shape_{ shape },
    padding_{ padding },
    data_(this->size_padded(), static_cast<value_type>(0.0)) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }

    if constexpr (layout_ == memory_layout::aos) {
// fill rows with values, respecting padding entries
#pragma omp parallel for
        for (size_type row = 0; row < this->num_rows(); ++row) {
            std::fill_n(this->data() + row * this->num_cols_padded(), this->num_cols(), static_cast<value_type>(init));
        }
    } else if constexpr (layout_ == memory_layout::soa) {
// fill columns with values, respecting padding entries
#pragma omp parallel for
        for (size_type col = 0; col < this->num_cols(); ++col) {
            std::fill_n(this->data() + col * this->num_rows_padded(), this->num_rows(), static_cast<value_type>(init));
        }
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
}

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const detail::shape shape, const std::vector<value_type> &data) :
    matrix{ shape } {
    if (this->size() != data.size()) {
        throw matrix_exception{ fmt::format("The number of entries in the matrix ({}) must be equal to the size of the data ({})!", this->size(), data.size()) };
    }

    // memcpy data to matrix
    std::memcpy(this->data(), data.data(), this->size() * sizeof(value_type));
}

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const detail::shape shape, const std::vector<value_type> &data, const detail::shape padding) :
    matrix{ shape, padding } {
    if (this->size() != data.size()) {
        throw matrix_exception{ fmt::format("The number of entries in the matrix ({}) must be equal to the size of the data ({})!", this->size(), data.size()) };
    }

    // memcpy data row- or column-wise depending on the layout type to the matrix
    this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), data.data(), this->shape());
}

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const detail::shape shape, const_pointer data) :
    matrix{ shape } {
    if (data == nullptr && !this->empty()) {
        throw matrix_exception{ "The provided data pointer may not be a nullptr if the matrix size is greater than 0!" };
    }
    if (!this->empty()) {
        // memcpy data to matrix
        std::memcpy(this->data(), data, this->size() * sizeof(value_type));
    }
}

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const detail::shape shape, const_pointer data, const detail::shape padding) :
    matrix{ shape, padding } {
    if (data == nullptr && !this->empty()) {
        throw matrix_exception{ "The provided data pointer may not be a nullptr if the matrix size is greater than 0!" };
    }
    if (!this->empty()) {
        // memcpy data row- or column-wise depending on the layout type to the matrix
        this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), data, this->shape());
    }
}

template <typename T, memory_layout layout_>
template <memory_layout other_layout_>
matrix<T, layout_>::matrix(const matrix<T, other_layout_> &other) :
    matrix{ other.shape(), other.padding() } {
    if constexpr (layout_ == other_layout_) {
        // same layout -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->size_padded() * sizeof(value_type));
    } else {
        const size_type num_rows = this->num_rows();
        const size_type num_cols = this->num_cols();
// convert AoS -> SoA or SoA -> AoS
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, memory_layout layout_>
template <memory_layout other_layout_>
matrix<T, layout_>::matrix(const matrix<value_type, other_layout_> &other, const detail::shape padding) :
    matrix{ other.shape(), padding } {
    if (layout_ == other_layout_ && this->padding() == other.padding()) {
        // same layout and same padding -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->size_padded() * sizeof(value_type));
    } else if (layout_ == other_layout_) {
        // same layout but different padding -> memcpy each row separately
        this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), other.data(), other.shape_padded());
    } else {
        const size_type num_rows = this->num_rows();
        const size_type num_cols = this->num_cols();
// convert AoS -> SoA or SoA -> AoS or manual copy because of mismatching padding sizes
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data) :
    matrix{ data, detail::shape{ 0, 0 } } { }

template <typename T, memory_layout layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data, const detail::shape padding) :
    padding_{ padding } {
    if (data.empty()) {
        // the provided 2D vector was empty -> set to empty matrix
        shape_ = detail::shape{ 0, 0 };
        data_ = std::vector<value_type>(this->size_padded(), value_type{});
    } else {
        if (!std::all_of(data.cbegin(), data.cend(), [&data](const std::vector<value_type> &row) { return row.size() == data.front().size(); })) {
            throw matrix_exception{ "Each row in the matrix must contain the same amount of columns!" };
        }
        if (data.front().empty()) {
            throw matrix_exception{ "The data to create the matrix must at least have one column!" };
        }

        // the provided 2D vector contains at least one element -> initialize matrix
        shape_ = detail::shape{ data.size(), data.front().size() };
        data_ = std::vector<value_type>(this->size_padded(), value_type{});

        if constexpr (layout_ == memory_layout::aos) {
// in case of AoS layout speed up conversion by using a simple memcpy over each row
#pragma omp parallel for
            for (size_type row = 0; row < this->num_rows(); ++row) {
                std::memcpy(this->data() + row * this->num_cols_padded(), data[row].data(), this->num_cols() * sizeof(value_type));
            }
        } else {
            const size_type num_rows = this->num_rows();
            const size_type num_cols = this->num_cols();
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
            for (size_type row = 0; row < num_rows; ++row) {
                for (size_type col = 0; col < num_cols; ++col) {
                    (*this)(row, col) = data[row][col];
                }
            }
        }
    }
}

template <typename T, memory_layout layout_>
void matrix<T, layout_>::restore_padding() noexcept {
    if constexpr (layout_ == memory_layout::aos) {
// restore padding row-wise
#pragma omp parallel for
        for (size_type row = 0; row < this->num_rows(); ++row) {
            std::memset(this->data() + (row + 1) * this->num_cols_padded() - padding_.y, 0, padding_.y * sizeof(value_type));
        }
        std::memset(this->data() + this->num_rows() * this->num_cols_padded(), 0, padding_.x * this->num_cols_padded() * sizeof(value_type));
    } else if constexpr (layout_ == memory_layout::soa) {
// restore padding column-wise
#pragma omp parallel for
        for (size_type col = 0; col < this->num_cols(); ++col) {
            std::memset(this->data() + (col + 1) * this->num_rows_padded() - padding_.x, 0, padding_.x * sizeof(value_type));
        }
        std::memset(this->data() + this->num_cols() * this->num_rows_padded(), 0, padding_.y * this->num_rows_padded() * sizeof(value_type));
    } else {
        static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
    }
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::value_type matrix<T, layout_>::operator()(const size_type row, const size_type col) const {
    SYCL_LSH_ASSERT(row < this->num_rows_padded(), fmt::format("The current row ({}) must be smaller than the number of padded rows ({})!", row, this->num_rows_padded()));
    SYCL_LSH_ASSERT(col < this->num_cols_padded(), fmt::format("The current column ({}) must be smaller than the number of padded columns ({})!", col, this->num_cols_padded()));
    if constexpr (layout_ == memory_layout::aos) {
        return data_[row * this->num_cols_padded() + col];
    } else if constexpr (layout_ == memory_layout::soa) {
        return data_[col * this->num_rows_padded() + row];
    } else {
        static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
    }
    detail::unreachable();
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::reference matrix<T, layout_>::operator()(const size_type row, const size_type col) {
    SYCL_LSH_ASSERT(row < this->num_rows_padded(), fmt::format("The current row ({}) must be smaller than the number of padded rows ({})!", row, this->num_rows_padded()));
    SYCL_LSH_ASSERT(col < this->num_cols_padded(), fmt::format("The current column ({}) must be smaller than the number of padded columns ({})!", col, this->num_cols_padded()));
    if constexpr (layout_ == memory_layout::aos) {
        return data_[row * this->num_cols_padded() + col];
    } else if constexpr (layout_ == memory_layout::soa) {
        return data_[col * this->num_rows_padded() + row];
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
    detail::unreachable();
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::value_type matrix<T, layout_>::at(const size_type row, const size_type col) const {
    if (row >= this->num_rows_padded()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows including padding ({} + {})!", row, this->num_rows(), padding_.x) };
    }
    if (col >= this->num_cols_padded()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns including padding ({} + {})!", col, this->num_cols(), padding_.y) };
    }

    return (*this)(row, col);
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::reference matrix<T, layout_>::at(const size_type row, const size_type col) {
    if (row >= this->num_rows_padded()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows including padding ({} + {})!", row, this->num_rows(), padding_.x) };
    }
    if (col >= this->num_cols_padded()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns including padding ({} + {})!", col, this->num_cols(), padding_.y) };
    }

    return (*this)(row, col);
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::value_type matrix<T, layout_>::operator[](const size_type idx) const {
    SYCL_LSH_ASSERT(idx < this->size_padded(), fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()));
    return data_[idx];
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::reference matrix<T, layout_>::operator[](const size_type idx) {
    SYCL_LSH_ASSERT(idx < this->size_padded(), fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()));
    return data_[idx];
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::value_type matrix<T, layout_>::at(const size_type idx) const {
    if (idx >= this->size_padded()) {
        throw matrix_exception{ fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()) };
    }
    return data_[idx];
}

template <typename T, memory_layout layout_>
typename matrix<T, layout_>::reference matrix<T, layout_>::at(const size_type idx) {
    if (idx >= this->size_padded()) {
        throw matrix_exception{ fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()) };
    }
    return data_[idx];
}

template <typename T, memory_layout layout_>
auto matrix<T, layout_>::to_2D_vector() const -> std::vector<std::vector<value_type>> {
    std::vector<std::vector<value_type>> ret(this->num_rows(), std::vector<value_type>(this->num_cols()));
    if constexpr (layout_ == memory_layout::aos) {
// in case of AoS layout speed up conversion by using a simple memcpy over each row
#pragma omp parallel for
        for (size_type row = 0; row < this->num_rows(); ++row) {
            std::memcpy(ret[row].data(), this->data() + row * this->num_cols_padded(), this->num_cols() * sizeof(value_type));
        }
    } else {
        const size_type num_rows = this->num_rows();
        const size_type num_cols = this->num_cols();
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                ret[row][col] = (*this)(row, col);
            }
        }
    }
    return ret;
}

template <typename T, memory_layout layout_>
auto matrix<T, layout_>::to_2D_vector_padded() const -> std::vector<std::vector<value_type>> {
    std::vector<std::vector<value_type>> ret(this->num_rows_padded(), std::vector<value_type>(this->num_cols_padded(), value_type{}));
    if constexpr (layout_ == memory_layout::aos) {
// in case of AoS layout speed up conversion by using a simple memcpy over each row
#pragma omp parallel for
        for (size_type row = 0; row < this->num_rows_padded(); ++row) {
            std::memcpy(ret[row].data(), this->data() + row * this->num_cols_padded(), this->num_cols_padded() * sizeof(value_type));
        }
    } else {
        const size_type num_rows = this->num_rows();
        const size_type num_cols = this->num_cols();
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                ret[row][col] = (*this)(row, col);
            }
        }
    }
    return ret;
}

template <typename T, memory_layout layout_>
void matrix<T, layout_>::swap(matrix<value_type, layout_> &other) noexcept {
    using std::swap;
    swap(this->shape_, other.shape_);
    swap(this->padding_, other.padding_);
    swap(this->data_, other.data_);
}

/**
 * @brief Swap the contents of @p lhs with the contents of @p rhs.
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 * @param[in, out] lhs the first matrix
 * @param[in,out] rhs the second matrix
 */
template <typename T, memory_layout layout_>
void swap(matrix<T, layout_> &lhs, matrix<T, layout_> &rhs) noexcept {
    lhs.swap(rhs);
}

/**
 * @brief Compares @p lhs and @p rhs for equality.
 * @details Comparing matrices with the same elements but different shapes, will return `false`.
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return `true` if both matrices are equal, otherwise `false` (`[[nodiscard]]`)
 */
template <typename T, memory_layout layout_>
[[nodiscard]] bool operator==(const matrix<T, layout_> &lhs, const matrix<T, layout_> &rhs) noexcept {
    return lhs.shape() == rhs.shape() && lhs.padding() == rhs.padding() && std::equal(lhs.data(), lhs.data() + lhs.size_padded(), rhs.data());
}

/**
 * @brief Compares @p lhs and @p rhs for inequality.
 * @details Comparing matrices with the same elements but different shapes, will return `true`.
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return `true` if both matrices are equal, otherwise `false` (`[[nodiscard]]`)
 */
template <typename T, memory_layout layout_>
[[nodiscard]] bool operator!=(const matrix<T, layout_> &lhs, const matrix<T, layout_> &rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Output the matrix entries in @p matr to the output-stream @p out.
 * @details **Doesn't** output the padding entries.
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 * @param[in,out] out the output-stream to print the matrix entries to
 * @param[in] matr the matrix to print
 * @return the output-stream
 */
template <typename T, memory_layout layout_>
std::ostream &operator<<(std::ostream &out, const matrix<T, layout_> &matr) {
    using size_type = typename matrix<T, layout_>::size_type;
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            out << fmt::format(fmt::runtime("{} "), matr(row, col));
        }
        if (row < matr.num_rows() - 1) {
            out << '\n';
        }
    }
    return out;
}

/**
 * @brief Typedef for a matrix in Array-of-Struct (AoS) layout.
 */
template <typename T>
using aos_matrix = matrix<T, memory_layout::aos>;
/**
 * @brief Typedef for a matrix in Struct-of-Array (SoA) layout.
 */
template <typename T>
using soa_matrix = matrix<T, memory_layout::soa>;

}  // namespace sycl_lsh

/// @cond Doxygen_suppress

template <typename T, sycl_lsh::memory_layout layout>
struct fmt::formatter<sycl_lsh::matrix<T, layout>> : fmt::ostream_formatter { };

/// @endcond

#endif  // SYCL_LSH_DETAIL_MATRIX_HPP
