/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the @ref sycl_lsh::knn class representing the result of the k-nearest-neighbor search.
 */

#ifndef SYCL_LSH_KNN_HPP
#define SYCL_LSH_KNN_HPP
#pragma once

#include "sycl_lsh/data.hpp"                  // sycl_lsh::data
#include "sycl_lsh/data_attributes.hpp"       // sycl_lsh::data_attributes
#include "sycl_lsh/memory_layout.hpp"         // sycl_lsh::memory_layout
#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/math.hpp"       // sycl_lsh::mpi::detail::sum
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/file_parser/file.hpp"  // sycl_lsh::mpi::file::mode
#include "sycl_lsh/mpi/logger.hpp"            // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"             // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"               // sycl_lsh::options

#include "fmt/format.h"  // fmt::format
#include "mpi.h"     // MPI_Sendrecv_replace, MPI_STATUS_IGNORE

#include <algorithm>  // std::copy, std::transform, std::count, std::sort
#include <cmath>      // std::sqrt
#include <limits>     // std::numeric_limits::max
#include <string>     // std::string
#include <tuple>      // std::tuple, std::make_tuple
#include <vector>     // std::vector

namespace sycl_lsh {

// forward declare knn class
template <memory_layout layout>
class knn;

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::knn class to convert a multidimensional
 *        index to a one-dimensional one.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct get_linear_id<knn<layout>> {
    /**
     * @brief Convert the multidimensional index to a one-dimensional index.
     * @param[in] point the requested data point
     * @param[in] nn the requested nearest-neighbor
     * @param[in] attr the attributes of the used data set
     * @param[in] k the number of nearest-neighbors to search for
     * @return the one-dimensional index (`[[nodiscard]]`)
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)` (currently disabled).
     * @pre @p k must be greater than `0`.
     * @pre @p nn must be in the range `[0, number of nearest-neighbors to search for)` (currently disabled).
     */
    [[nodiscard]] index_type operator()(const index_type point, const index_type nn, [[maybe_unused]] const data_attributes &attr, [[maybe_unused]] const index_type k) const noexcept {  // TODO
        // SYCL_LSH_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!");
        // SYCL_LSH_ASSERT(0 < k, "Illegal number of k-nearest-neighbors!");
        // SYCL_LSH_ASSERT(0 <= nn && nn < k, "Out-of-bounce access for nearest-neighbor!");

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return point * k + nn;
        } else {
            // Struct of Arrays
            return nn * attr.rank_size + point;
        }
    }
};

}  // namespace detail

/**
 * @brief Factory function for the @ref sycl_lsh::knn class.
 * @brief Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @param[in] k the number of nearest-neighbors to search for
 * @param[in] data the used @ref sycl_lsh::data representing the used data set
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] logger the used @ref sycl_lsh::mpi::logger
 * @return the @ref sycl_lsh::knn object representing the result of the nearest-neighbor search (`[[nodiscard]]`)
 */
template <memory_layout layout>
[[nodiscard]] auto make_knn(const index_type k, const options &, const data<layout> &data, const mpi::communicator &comm, const mpi::logger &logger) {
    return knn<layout>(k, data, comm, logger);
}
/**
 * @brief Factory function for the @ref sycl_lsh::knn class.
 * @brief Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @param[in] opt the used @ref sycl_lsh::options
 * @param[in] data the used @ref sycl_lsh::data representing the used data set
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] logger the used @ref sycl_lsh::mpi::logger
 * @return the @ref sycl_lsh::knn object representing the result of the nearest-neighbor search (`[[nodiscard]]`)
 */
template <memory_layout layout>
[[nodiscard]] auto make_knn(const options &opt, const data<layout> &data, const mpi::communicator &comm, const mpi::logger &logger) {
    return make_knn<layout>(opt.k, opt, data, comm, logger);
}

/**
 * @brief Class representing the result of the k-nearest-neighbor search.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
class knn {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                type aliases                                                //

    /// The type of the host buffer representing the k-nearest-neighbor IDs used to hide the MPI communications.
    using knn_host_buffer_type = std::vector<index_type>;
    /// The type of the host buffer representing the k-nearest-neighbor distances used to hide the MPI communications.
    using dist_host_buffer_type = std::vector<real_type>;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                             update host buffer                                             //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Send the elements of the host buffers to the neighboring buffers replacing its content using a ring like send pattern.
     */
    void send_receive_host_buffer();

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                knn results                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the IDs (indices) of the k-nearest-neighbors found for @p point.
     * @param[in] point the data point to return the nearest-neighbors for
     * @return the indices of the found k-nearest-neighbors of @p point (`[[nodiscard]]`)
     *
     * @attention Copies the IDs to the result vector!
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)`.
     */
    [[nodiscard]] knn_host_buffer_type get_knn_ids(index_type point) const;
    /**
     * @brief Returns the distances of the k-nearest-neighbors found for @p point.
     * @param[in] point the data point to return the nearest-neighbors for
     * @return the distances of the found k-nearest-neighbors of @p point (`[[nodiscard]]`)
     *
     * @attention Copies the distances to the result vector!
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)`.
     */
    [[nodiscard]] dist_host_buffer_type get_knn_dists(index_type point) const;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  save knn                                                  //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Saves the calculated k-nearest-neighbor IDs. \n
     *        **Always** saves the k-nearest-neighbor IDs in *Array of Structs* layout.
     * @param[in] opt the used @ref sycl_lsh::options
     *
     * @throws sycl_lsh::exception if the command line argument `knn_save_file` isn't present in @p parser.
     */
    void save_knns(const options &opt);
    /**
     * @brief Saves the calculated k-nearest-neighbor distances. \n
     *        **Always** saves the k-nearest-neighbor distances in *Array of Structs* layout.
     * @param[in] opt the used @ref sycl_lsh::options
     *
     * @throws sycl_lsh::exception if the command line argument `knn_dist_save_file` isn't present in @p parser.
     */
    void save_distances(const options &opt);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                evaluate knn                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
     * @param[in] opt the used @ref sycl_lsh::options
     * @return the resulting recall (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the required command line argument `evaluate_knn_file` isn't present in @p parser.
     * @throws sycl_lsh::exception if the parsed total number of points doesn't match with the current `total_size`.
     * @throws sycl_lsh::exception if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
     * @throws sycl_lsh::exception if the parsed number of dimensions doesn't match with the current `dims`.
     */
    [[nodiscard]] real_type recall(const options &opt);
    /**
     * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
     * @param[in] opt the used @ref sycl_lsh::options
     * @return a `std::tuple` containing the resulting error ratio, the number of points for which no k k-nearest-neighbors could be
     *         found and the total number of nearest-neighbors that couldn't be found (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the required command line argument `evaluate_knn_dist_file` isn't present in @p parser.
     * @throws sycl_lsh::exception if the parsed total number of points doesn't match with the current `total_size`.
     * @throws sycl_lsh::exception if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
     * @throws sycl_lsh::exception if the parsed number of dimensions doesn't match with the current `dims`.
     */
    [[nodiscard]] std::tuple<real_type, index_type, index_type> error_ratio(const options &opt);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the specified @ref sycl_lsh::memory_layout type.
     * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] static constexpr memory_layout get_memory_layout() noexcept { return layout; }

    /**
     * @brief Returns the host buffer containing the k-nearest-neighbor IDs used to hide the MPI communication.
     * @return the knn host buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] knn_host_buffer_type &get_knn_host_buffer() noexcept { return knn_host_buffer_; }
    /**
     * @brief Returns the host buffer containing the k-nearest-neighbor distances used to hide the MPI communication.
     * @details The distances are calculated without the use of `std::sqrt`!
     * @return the knn distances host buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] dist_host_buffer_type &get_distance_host_buffer() noexcept { return dist_host_buffer_; }

  private:
    // befriend the factory function
    friend auto make_knn<layout>(index_type, const options &, const data<layout> &, const mpi::communicator &, const mpi::logger &);
    friend auto make_knn<layout>(const options &, const data<layout> &, const mpi::communicator &, const mpi::logger &);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::knn object given @p k, the number of nearest-neighbors to search for.
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     *
     * @pre @p k **must** be greater than `0`.
     */
    knn(index_type k, const data<layout> &data, const mpi::communicator &comm, const mpi::logger &logger);

    /// The data attributes.
    const data_attributes attr_;
    /// The associated MPI communicator.
    const mpi::communicator &comm_;
    /// The associated MPI logger.
    const mpi::logger &logger_;

    /// The number of nearest-neighbors to calculate.
    const index_type k_;

    /// The SYCL host buffer for the nearest-neighbors.
    knn_host_buffer_type knn_host_buffer_;
    /// The SYCL host buffer for the nearest-neighbor distances.
    dist_host_buffer_type dist_host_buffer_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
knn<layout>::knn(const index_type k, const data<layout> &data, const mpi::communicator &comm, const mpi::logger &logger) :
    attr_{ data.get_attributes() }, comm_{ comm }, logger_{ logger }, k_{ k }, knn_host_buffer_(attr_.rank_size * k), dist_host_buffer_(attr_.rank_size * k, std::numeric_limits<real_type>::max()) {
    const mpi::timer mpi_timer{ comm_ };

    SYCL_LSH_ASSERT(0 < k, "Illegal number of k-nearest-neighbors!");

    // calculate start ID
    const index_type base_id = comm_.rank() * attr_.rank_size;

    const detail::get_linear_id<knn> get_linear_id_functor{};

    // fill default values
    for (index_type point = 0; point < attr_.rank_size; ++point) {
        for (index_type nn = 0; nn < k_; ++nn) {
            knn_host_buffer_[get_linear_id_functor(point, nn, attr_, k_)] = base_id + point;
        }
    }

    // correctly set default values for dummy points on last MPI rank
    if (comm_.rank() == comm_.size() - 1) {
        const index_type correct_rank_size = attr_.total_size - ((comm_.size() - 1) * attr_.rank_size);
        for (index_type point = correct_rank_size; point < attr_.rank_size; ++point) {
            for (index_type nn = 0; nn < k_; ++nn) {
                knn_host_buffer_[get_linear_id_functor(point, nn, attr_, k_)] = base_id + correct_rank_size - 1;
            }
        }
    }

    logger_.log("Created knn object in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                knn results                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
[[nodiscard]] auto knn<layout>::get_knn_ids(const index_type point) const -> knn_host_buffer_type {
    SYCL_LSH_ASSERT(0 <= point && point < attr_.rank_size, "Out-of-bounce access for data point!");

    const detail::get_linear_id<knn> get_linear_id_functor{};

    knn_host_buffer_type res(k_);
    for (index_type nn = 0; nn < k_; ++nn) {
        res[nn] = knn_host_buffer_[get_linear_id_functor(point, nn, attr_, k_)];
    }
    return res;
}
template <memory_layout layout>
[[nodiscard]] auto knn<layout>::get_knn_dists(const index_type point) const -> dist_host_buffer_type {
    SYCL_LSH_ASSERT(0 <= point && point < attr_.rank_size, "Out-of-bounce access for data point!\n");

    const detail::get_linear_id<knn> get_linear_id_functor{};

    dist_host_buffer_type res(k_);
    for (index_type nn = 0; nn < k_; ++nn) {
        res[nn] = dist_host_buffer_[get_linear_id_functor(point, nn, attr_, k_)];
    }
    return res;
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                  save knn                                                  //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
void knn<layout>::save_knns(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // check if the required command line argument is present
    if (!opt.knn_save_file.has_value()) {
        throw exception{ "Required command line argument 'knn_save_file' not provided!" };
    }

    knn_host_buffer_type tmp_buffer(knn_host_buffer_.size());

    if constexpr (layout == memory_layout::soa) {
        // expect the values to be saved in array of structs (aos) layout -> transform if wrong layout
        const detail::get_linear_id<knn<memory_layout::aos>> get_linear_id_aos{};
        const detail::get_linear_id<knn<memory_layout::soa>> get_linear_id_soa{};

        for (index_type point = 0; point < attr_.rank_size; ++point) {
            for (index_type nn = 0; nn < k_; ++nn) {
                tmp_buffer[get_linear_id_aos(point, nn, attr_, k_)] = knn_host_buffer_[get_linear_id_soa(point, nn, attr_, k_)];
            }
        }
    } else {
        // if the layout is correct, simply copy the values to the temporary buffer
        std::copy(knn_host_buffer_.begin(), knn_host_buffer_.end(), tmp_buffer.begin());
    }

    // write content to the respective file
    auto file_parser = mpi::make_file_parser<index_type>(opt.knn_save_file.value(), opt.file_parser, mpi::file::mode::write, comm_, logger_);
    file_parser->write_content(attr_.total_size, k_, tmp_buffer);

    logger_.log("Saved k-nearest-neighbor IDs in {}.\n", mpi_timer.elapsed());
}
template <memory_layout layout>
void knn<layout>::save_distances(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // check if the required command line argument is present
    if (!opt.knn_dist_save_file.has_value()) {
        throw exception{ "Required command line argument 'knn_dist_save_file' not provided!" };
    }

    dist_host_buffer_type tmp_buffer(dist_host_buffer_.size());

    if constexpr (layout == memory_layout::soa) {
        // expect the values to be saved in array of structs (aos) layout -> transform if wrong layout
        const detail::get_linear_id<knn<memory_layout::aos>> get_linear_id_aos{};
        const detail::get_linear_id<knn<memory_layout::soa>> get_linear_id_soa{};

        for (index_type point = 0; point < attr_.rank_size; ++point) {
            for (index_type nn = 0; nn < k_; ++nn) {
                tmp_buffer[get_linear_id_aos(point, nn, attr_, k_)] = dist_host_buffer_[get_linear_id_soa(point, nn, attr_, k_)];
            }
        }
    } else {
        // if the layout is correct, simply copy the values to the temporary buffer (because of the call to `std::sqrt` later on)
        std::copy(dist_host_buffer_.begin(), dist_host_buffer_.end(), tmp_buffer.begin());
    }

    // transform the values using `std::sqrt`
    std::transform(tmp_buffer.begin(), tmp_buffer.end(), tmp_buffer.begin(), [](const real_type val) { return std::sqrt(val); });

    // write content to the respective file
    auto file_parser = mpi::make_file_parser<real_type>(opt.knn_dist_save_file.value(), opt.file_parser, mpi::file::mode::write, comm_, logger_);
    file_parser->write_content(attr_.total_size, k_, tmp_buffer);

    logger_.log("Saved k-nearest-neighbor distances in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                evaluate knn                                                //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
[[nodiscard]]
real_type knn<layout>::recall(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // load correct k-nearest-neighbor IDs
    // check if the required command line argument is present
    if (!opt.evaluate_knn_file.has_value()) {
        throw exception{ "Required command line argument 'evaluate_knn_file' not provided!" };
    }

    // read correct k-nearest-neighbor IDs from the respective file
    const std::string &file_name = opt.evaluate_knn_file.value();
    auto file_parser = mpi::make_file_parser<index_type>(file_name, opt.file_parser, mpi::file::mode::read, comm_, logger_);
    const index_type parsed_total_size = file_parser->parse_total_size();
    const index_type parsed_rank_size = file_parser->parse_rank_size();
    const index_type parsed_dims = file_parser->parse_dims();
    knn_host_buffer_type correct_knn = file_parser->parse_content();

    // perform sanity checks
    if (parsed_total_size != attr_.total_size) {
        throw exception{ fmt::format("The total number of points in '{}' is {}, but should be {}!", file_name, parsed_total_size, attr_.total_size) };
    }
    if (parsed_rank_size != attr_.rank_size) {
        throw exception{ fmt::format("The number of points per MPI rank in '{}' is {}, but should be {}!", file_name, parsed_rank_size, attr_.rank_size) };
    }
    if (parsed_dims != k_) {
        throw exception{ fmt::format("The number of nearest-neighbors in '{}' is {}, but should be {}!", file_name, parsed_dims, k_) };
    }

    const index_type correct_rank_size = comm_.rank() == comm_.size() - 1 ? (attr_.total_size - (comm_.size() - 1) * attr_.rank_size) : attr_.rank_size;

    const detail::get_linear_id<knn> get_linear_id_this{};
    const detail::get_linear_id<knn<memory_layout::aos>> get_linear_id_aos{};

    index_type count = 0;
    for (index_type point = 0; point < correct_rank_size; ++point) {
        for (index_type nn = 0; nn < k_; ++nn) {
            // get calculated k-nearest-neighbor ID
            const index_type calculated_id = knn_host_buffer_[get_linear_id_this(point, nn, attr_, k_)];
            // check if calculated ID is contained in the correct IDs
            for (index_type i = 0; i < k_; ++i) {
                if (calculated_id == correct_knn[get_linear_id_aos(point, i, attr_, k_)]) {
                    // correct ID found
                    ++count;
                    break;
                }
            }
        }
    }

    const real_type res = (static_cast<real_type>(mpi::detail::sum(count, comm_)) / (attr_.total_size * k_)) * real_type{ 100.0 };

    logger_.log("\nCalculated recall in {}.\n", mpi_timer.elapsed());
#if defined(SYCL_LSH_BENCHMARK)
    if (comm_.is_main_rank()) {
        mpi::timer::benchmark_out() << res << ',';
    }
#endif

    return res;
}

template <memory_layout layout>
[[nodiscard]] std::tuple<real_type, index_type, index_type> knn<layout>::error_ratio(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // load correct k-nearest-neighbor distances
    // check if the required command line argument is present
    if (!opt.evaluate_knn_dist_file.has_value()) {
        throw exception{ "Required command line argument 'evaluate_knn_dist_file' not provided!" };
    }

    // read correct k-nearest-neighbor distances from the respective file
    const std::string &file_name = opt.evaluate_knn_dist_file.value();
    auto file_parser = mpi::make_file_parser<real_type>(file_name, opt.file_parser, mpi::file::mode::read, comm_, logger_);
    const index_type parsed_total_size = file_parser->parse_total_size();
    const index_type parsed_rank_size = file_parser->parse_rank_size();
    const index_type parsed_dims = file_parser->parse_dims();
    dist_host_buffer_type correct_knn_dist = file_parser->parse_content();

    // perform sanity checks
    if (parsed_total_size != attr_.total_size) {
        throw exception{ fmt::format("The total number of points in '{}' is {}, but should be {}!", file_name, parsed_total_size, attr_.total_size) };
    }
    if (parsed_rank_size != attr_.rank_size) {
        throw exception{ fmt::format("The number of points per MPI rank in '{}' is {}, but should be {}!", file_name, parsed_rank_size, attr_.rank_size) };
    }
    if (parsed_dims != k_) {
        throw exception{ fmt::format("The number of nearest-neighbor distances in '{}' is {}, but should be {}!", file_name, parsed_dims, k_) };
    }

    const index_type correct_rank_size = comm_.rank() == comm_.size() - 1 ? (attr_.total_size - (comm_.size() - 1) * attr_.rank_size) : attr_.rank_size;

    const detail::get_linear_id<knn> get_linear_id_this{};
    const detail::get_linear_id<knn<memory_layout::aos>> get_linear_id_aos{};

    // calculate error ratio
    index_type num_points_not_found = 0;
    index_type num_knn_not_found = 0;
    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    dist_host_buffer_type calculated_knn_dist_sorted(k_);
    dist_host_buffer_type correct_knn_dist_sorted(k_);

    for (index_type point = 0; point < correct_rank_size; ++point) {
        // fill k-nearest-neighbor distances for current point
        for (index_type nn = 0; nn < k_; ++nn) {
            calculated_knn_dist_sorted[nn] = dist_host_buffer_[get_linear_id_this(point, nn, attr_, k_)];
            correct_knn_dist_sorted[nn] = correct_knn_dist[get_linear_id_aos(point, nn, attr_, k_)];
        }
        // check whether k k-nearest-neighbor could be found
        auto count_not_found = std::count(calculated_knn_dist_sorted.cbegin(), calculated_knn_dist_sorted.cend(), std::numeric_limits<real_type>::max());
        if (count_not_found != 0) {
            ++num_points_not_found;
            num_knn_not_found += count_not_found;
            continue;
        }
        // calculate `std::sqrt` distance
        std::transform(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end(), calculated_knn_dist_sorted.begin(), [](const real_type val) { return std::sqrt(val); });
        // sort distances
        std::sort(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end());
        std::sort(correct_knn_dist_sorted.begin(), correct_knn_dist_sorted.end());

        // calculate error ratio
        index_type error_count = 0;
        real_type error_ratio = 0.0;
        for (index_type nn = 0; nn < k_; ++nn) {
            if (correct_knn_dist_sorted[nn] == real_type{ 0.0 }) {
                // two different points at the same position
                if (calculated_knn_dist_sorted[nn] == real_type{ 0.0 }) {
                    // calculated nearest neighbor is correct
                    error_ratio += 1.0;
                    ++error_count;
                }
            } else {
                // calculate distance ratio
                error_ratio += calculated_knn_dist_sorted[nn] / correct_knn_dist_sorted[nn];
                ++error_count;
            }
        }
        // calculate error ratio for current k-nearest neighbors
        if (error_count != 0) {
            mean_error_ratio += error_ratio / static_cast<real_type>(error_count);
            ++mean_error_count;
        }
    }

    // collect results from each MPI rank
    const real_type avg_mean_error_ratio = mpi::detail::sum(mean_error_ratio, comm_) / mpi::detail::sum(mean_error_count, comm_);
    const index_type total_num_points_not_found = mpi::detail::sum(num_points_not_found, comm_);
    const index_type total_num_knn_not_found = mpi::detail::sum(num_knn_not_found, comm_);

    logger_.log("\nCalculated error ration in {}.\n", mpi_timer.elapsed());
#if defined(SYCL_LSH_BENCHMARK)
    if (comm_.is_main_rank()) {
        mpi::timer::benchmark_out() << avg_mean_error_ratio << ','
                                    << total_num_points_not_found << ','
                                    << total_num_knn_not_found << ',';
    }
#endif

    return std::make_tuple(avg_mean_error_ratio, total_num_points_not_found, total_num_knn_not_found);
}

// ---------------------------------------------------------------------------------------------------------- //
//                                             update host buffer                                             //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
void knn<layout>::send_receive_host_buffer() {
    const int destination = (comm_.rank() + 1) % comm_.size();
    const int source = (comm_.size() + (comm_.rank() - 1) % comm_.size()) % comm_.size();

    // send/receive k-nearest-neighbor IDs
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Sendrecv_replace(knn_host_buffer_.data(), knn_host_buffer_.size(), mpi::detail::mpi_datatype<typename knn_host_buffer_type::value_type>(), destination, 0, source, 0, comm_.get(), MPI_STATUS_IGNORE));

    // send/receive k-nearest-neighbor distances
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Sendrecv_replace(dist_host_buffer_.data(), dist_host_buffer_.size(), mpi::detail::mpi_datatype<typename dist_host_buffer_type::value_type>(), destination, 0, source, 0, comm_.get(), MPI_STATUS_IGNORE));
}

}  // namespace sycl_lsh

#endif  // SYCL_LSH_KNN_HPP
