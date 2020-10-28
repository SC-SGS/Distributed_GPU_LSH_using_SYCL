/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Implements the @ref sycl_lsh::knn class representing the result of the k-nearest-neighbor search.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/data.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/memory_layout.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/file.hpp>
#include <sycl_lsh/mpi/math.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/timer.hpp>
#include <sycl_lsh/options.hpp>

#include <fmt/format.h>
#include <mpi.h>

#include <algorithm>
#include <utility>
#include <type_traits>
#include <vector>

namespace sycl_lsh {

    // forward declare knn class
    template <memory_layout layout, typename Options, typename Data>
    class knn;

    /**
     * @brief Factory function for the @ref sycl_lsh::knn class.
     * @brief Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
     * @tparam layout the used @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     * @return the @ref sycl_lsh::knn object representing the result of the nearest-neighbor search (`[[nodiscard]]`)
     */
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    inline auto make_knn(const typename Options::index_type k, const Options&, const Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        return knn<layout, Options, Data>(k, data, comm, logger);
    }
    /**
     * @brief Factory function for the @ref sycl_lsh::knn class.
     * @brief Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
     * @tparam layout the used @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @param[in] parser the used @ref sycl_lsh::argv_parser
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     * @return the @ref sycl_lsh::knn object representing the result of the nearest-neighbor search (`[[nodiscard]]`)
     */
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    inline auto make_knn(const argv_parser& parser, const Options& opt, const Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        return make_knn<layout>(parser.argv_as<typename Options::index_type>("k"), opt, data, comm, logger);
    }

    /**
     * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::knn class to convert a multi-dimensional
     *        index to an one-dimensional one.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the @ref sycl_lsh::options type
     * @tparam Data the @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    struct get_linear_id<knn<layout, Options, Data>> {

        /// The used @ref sycl_lsh::options type.
        using options_type = Options;
        /// The used integral type (used for indices).
        using index_type = typename options_type::index_type;

        /// The used @ref sycl_lsh::data type.
        using data_type = Data;
        /// The used @ref sycl_lsh::data_attributes type.
        using data_attributes_type = typename data_type::data_attributes_type;

        /**
         * @brief Convert the multi-dimensional index to an one-dimensional index.
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
        [[nodiscard]]
        index_type operator()(const index_type point, const index_type nn,
                              [[maybe_unused]] const data_attributes_type& attr, [[maybe_unused]] const index_type k) const noexcept 
        {
//            SYCL_LSH_DEBUG_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!\n");
//            SYCL_LSH_DEBUG_ASSERT(0 < k, "Illegal number of k-nearest-neighbors!\n");
//            SYCL_LSH_DEBUG_ASSERT(0 <= nn && nn < k, "Out-of-bounce access for nearest-neighbor!\n");

            if constexpr (layout == memory_layout::aos) {
                // Array of Structs
                return point * k + nn;
            } else {
                // Struct of Arrays
                return nn * attr.rank_size + point;
            }
        }

    };


    /**
     * @brief Class representing the result of the k-nearest-neighbor search.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    class knn : private detail::knn_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must be a sycl_lsh::data type!");
    public:
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                type aliases                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /// The type of the @ref sycl_lsh::options object.
        using options_type = Options;
        /// The used floating point type for the k-nearest-neighbor distances.
        using real_type = typename Options::real_type;
        /// The used integral type for the k-nearest-neighbor IDs.
        using index_type = typename Options::index_type;

        /// The type of the @ref sycl_lsh::data object.
        using data_type = Data;
        /// The type of the @ref sycl_lsh::data_attributes object.
        using data_attributes_type = typename data_type::data_attributes_type;

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
        [[nodiscard]]
        knn_host_buffer_type get_knn_ids(const index_type point) const;
        /**
         * @brief Returns the distances of the k-nearest-neighbors found for @p point.
         * @param[in] point the data point to return the nearest-neighbors for
         * @return the distances of the found k-nearest-neighbors of @p point (`[[nodiscard]]`)
         *
         * @attention Copies the distances to the result vector!
         *
         * @pre @p point must be in the range `[0, number of data points on the current MPI rank)`.
         */
        [[nodiscard]]
        dist_host_buffer_type get_knn_dists(const index_type point) const;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                  save knn                                                  //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Saves the calculated k-nearest-neighbor IDs to the file parsed from the command line arguments
         *        @ref sycl_lsh::argv_parser @p parser via the command line argument `knn_save_file`. \n
         *        **Always** saves the k-nearest-neighbor IDs in *Array of Structs* layout.
         * @param[in] parser the used @ref sycl_lsh::argv_parser
         *
         * @throws std::invalid_argument if the command line argument `knn_save_file` isn't present in @p parser.
         */
        void save_knns(const argv_parser& parser);
        /**
         * @brief Saves the calculated k-nearest-neighbor distances to the file parser from the command line arguments
         *        @ref sycl_lsh::argv_parser @p parser via the command line argument `knn_dist_save_file`. \n
         *        **Always** saves the k-nearest-neighbor distances in *Array of Structs* layout.
         * @param[in] parser the used @ref sycl_lsh::argv_parser
         *
         * @throws std::invalid_argument if the command line argument `knn_dist_save_file` isn't present in @p parser.
         */
        void save_distances(const argv_parser& parser);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                evaluate knn                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
         * @param[in] parser the used @ref sycl_lsh::argv_parser
         * @return the resulting recall (`[[nodiscard]]`)
         *
         * @throws std::invalid_argument if the required command line argument `evaluate_knn_file` isn't present in @p parser.
         * @throws std::runtime_error if the parsed total number of points doesn't match with the current `total_size`.
         * @throws std::runtime_error if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
         * @throws std::runtime_error if the parsed number of dimensions doesn't match with the current `dims`.
         */
        [[nodiscard]]
        real_type recall(const argv_parser& parser);
        /**
         * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
         * @param[in] parser the used @ref sycl_lsh::argv_parser
         * @return a `std::tuple` containing the resulting error ratio, the number of points for which no k k-nearest-neighbors could be
         *         found and the total number of nearest-neighbors that couldn't be found (`[[nodiscard]]`)
         *
         * @throws std::invalid_argument if the required command line argument `evaluate_knn_dist_file` isn't present in @p parser.
         * @throws std::runtime_error if the parsed total number of points doesn't match with the current `total_size`.
         * @throws std::runtime_error if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
         * @throws std::runtime_error if the parsed number of dimensions doesn't match with the current `dims`.
         */
        [[nodiscard]]
        std::tuple<real_type, index_type, index_type> error_ratio(const argv_parser& parser);

        
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
 * @brief Returns the specified @ref sycl_lsh::memory_layout type.
 * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
 */
        [[nodiscard]]
        constexpr memory_layout get_memory_layout() const noexcept { return layout; }

        /**
         * @brief Returns the host buffer containing the k-nearest-neighbor IDs used to hide the MPI communication.
         * @return the knn host buffer (`[[nodiscard]]`)
         */
        [[nodiscard]]
        knn_host_buffer_type& get_knn_host_buffer() noexcept { return knn_host_buffer_; }
        /**
         * @brief Returns the host buffer containing the k-nearest-neighbor distances used to hide the MPI communication.
         * @details The distances are calculated without the use of `std::sqrt`!
         * @return the knn distances host buffer (`[[nodiscard]]`)
         */
        [[nodiscard]]
        dist_host_buffer_type& get_distance_host_buffer() noexcept { return dist_host_buffer_; }

    private:
        // befriend the factory function
        friend auto make_knn<layout, Options, Data>(const typename Options::index_type, const Options&, const Data&, const mpi::communicator&, const mpi::logger&);
        friend auto make_knn<layout, Options, Data>(const argv_parser&, const Options&, const Data&, const mpi::communicator&, const mpi::logger&);

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
        /**
         * @brief Construct a new @ref sycl_lsh::knn object given @p k, the number of nearest-neighbors to search for.
         * @param[in] k the number of nearest-neighbors to search for
         * @param[in] data the used @ref sycl_lsh::data representing the used data set
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         *
         * @pre @p k **must** be greater than `0`.
         */
        knn(const index_type k, const data_type& data, const mpi::communicator& comm, const mpi::logger& logger);


        const data_attributes_type attr_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        const index_type k_;

        knn_host_buffer_type knn_host_buffer_;
        dist_host_buffer_type dist_host_buffer_;
    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    knn<layout, Options, Data>::knn(const index_type k, const data_type& data, const mpi::communicator& comm, const mpi::logger& logger)
        : attr_(data.get_attributes()), comm_(comm), logger_(logger),
          k_(k),
          knn_host_buffer_(attr_.rank_size * k),
          dist_host_buffer_(attr_.rank_size * k, std::numeric_limits<real_type>::max())
    {
        mpi::timer t(comm_);

        SYCL_LSH_DEBUG_ASSERT(0 < k, "Illegal number of k-nearest-neighbors!\n");

        // calculate start ID
        const index_type base_id = comm_.rank() * attr_.rank_size;

        const get_linear_id<knn<layout, options_type, data_type>> get_linear_id_functor{};

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

        logger_.log("Created knn object in {}.\n", t.elapsed());
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                knn results                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    typename knn<layout, Options, Data>::knn_host_buffer_type knn<layout, Options, Data>::get_knn_ids(const index_type point) const {
        SYCL_LSH_DEBUG_ASSERT(0 <= point && point < attr_.rank_size, "Out-of-bounce access for data point!\n");

        const get_linear_id<knn<layout, options_type, data_type>> get_linear_id_functor{};

        knn_host_buffer_type res(k_);
        for (index_type nn = 0; nn < k_; ++nn) {
            res[nn] = knn_host_buffer_[get_linear_id_functor(point, nn, attr_, k_)];
        }
        return res;
    }
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    typename knn<layout, Options, Data>::dist_host_buffer_type knn<layout, Options, Data>::get_knn_dists(const index_type point) const {
        SYCL_LSH_DEBUG_ASSERT(0 <= point && point < attr_.rank_size, "Out-of-bounce access for data point!\n")

        const get_linear_id<knn<layout, options_type, data_type>> get_linear_id_functor{};

        dist_host_buffer_type res(k_);
        for (index_type nn = 0; nn < k_; ++nn) {
            res[nn] = dist_host_buffer_[get_linear_id_functor(point, nn, attr_, k_)];
        }
        return res;
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  save knn                                                  //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    void knn<layout, Options, Data>::save_knns(const argv_parser& parser) {
        mpi::timer t(comm_);

        // check if the required command line argument is present
        if (!parser.has_argv("knn_save_file")) {
            throw std::invalid_argument("Required command line argument 'knn_save_file' not provided!");
        }

        knn_host_buffer_type tmp_buffer(knn_host_buffer_.size());

        if constexpr (layout == memory_layout::soa) {
            // expect the values to be saved in array of structs (aos) layout -> transform if wrong layout
            const get_linear_id<knn<memory_layout::aos, options_type, data_type>> get_linear_id_aos{};
            const get_linear_id<knn<memory_layout::soa, options_type, data_type>> get_linear_id_soa{};

            for (index_type point = 0; point < attr_.rank_size; ++point) {
                for (index_type nn = 0; nn < k_; ++nn) {
                    tmp_buffer[get_linear_id_aos(point, nn, attr_, k_)]
                            = knn_host_buffer_[get_linear_id_soa(point, nn, attr_, k_)];
                }
            }
        } else {
            // if the layout is correct, simply copy the values to the temporary buffer
            std::copy(knn_host_buffer_.begin(), knn_host_buffer_.end(), tmp_buffer.begin());
        }


        // write content to the respective file
        const std::string& file_name = parser.argv_as<std::string>("knn_save_file");
        auto file_parser = mpi::make_file_parser<index_type, options_type>(file_name, parser, mpi::file::mode::write, comm_, logger_);
        file_parser->write_content(attr_.total_size, k_, tmp_buffer);

        logger_.log("Saved k-nearest-neighbor IDs in {}.\n", t.elapsed());
    }
    template <memory_layout layout, typename Options, typename Data>
    void knn<layout, Options, Data>::save_distances(const argv_parser& parser) {
        mpi::timer t(comm_);

        // check if the required command line argument is present
        if (!parser.has_argv("knn_dist_save_file")) {
            throw std::invalid_argument("Required command line argument 'knn_dist_save_file' not provided!");
        }
        
        dist_host_buffer_type tmp_buffer(dist_host_buffer_.size());

        if constexpr (layout == memory_layout::soa) {
            // expect the values to be saved in array of structs (aos) layout -> transform if wrong layout
            const get_linear_id<knn<memory_layout::aos, options_type, data_type>> get_linear_id_aos{};
            const get_linear_id<knn<memory_layout::soa, options_type, data_type>> get_linear_id_soa{};

            for (index_type point = 0; point < attr_.rank_size; ++point) {
                for (index_type nn = 0; nn < k_; ++nn) {
                    tmp_buffer[get_linear_id_aos(point, nn, attr_, k_)]
                            = dist_host_buffer_[get_linear_id_soa(point, nn, attr_, k_)];
                }
            }
        } else {
            // if the layout is correct, simply copy the values to the temporary buffer (because of the call to `std::sqrt` later on)
            std::copy(dist_host_buffer_.begin(), dist_host_buffer_.end(), tmp_buffer.begin());
        }

        // transform the values using `std::sqrt`
        std::transform(tmp_buffer.begin(), tmp_buffer.end(), tmp_buffer.begin(), [](const real_type val) { return std::sqrt(val); });


        // write content to the respective file
        const std::string& file_name = parser.argv_as<std::string>("knn_dist_save_file");
        auto file_parser = mpi::make_file_parser<real_type, options_type>(file_name, parser, mpi::file::mode::write, comm_, logger_);
        file_parser->write_content(attr_.total_size, k_, tmp_buffer);

        logger_.log("Saved k-nearest-neighbor distances in {}.\n", t.elapsed());
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                evaluate knn                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    typename Options::real_type knn<layout, Options, Data>::recall(const argv_parser& parser) {
        mpi::timer t(comm_);

        // load correct k-nearest-neighbor IDs
        // check if the required command line argument is present
        if (!parser.has_argv("evaluate_knn_file")) {
            throw std::invalid_argument("Required command line argument 'evaluate_knn_file' not provided!");
        }

        // read correct k-nearest-neighbor IDs from the respective file
        const std::string& file_name = parser.argv_as<std::string>("evaluate_knn_file");
        auto file_parser = mpi::make_file_parser<index_type, options_type>(file_name, parser, mpi::file::mode::read, comm_, logger_);
        const index_type parsed_total_size = file_parser->parse_total_size();
        const index_type parsed_rank_size = file_parser->parse_rank_size();
        const index_type parsed_dims = file_parser->parse_dims();
        knn_host_buffer_type correct_knn = file_parser->parse_content();

        // perform sanity checks
        if (parsed_total_size != attr_.total_size) {
            throw std::runtime_error(fmt::format("The total number of points in '{}' is {}, but should be {}!", file_name, parsed_total_size, attr_.total_size));
        } else if (parsed_rank_size != attr_.rank_size) {
            throw std::runtime_error(fmt::format("The number of points per MPI rank in '{}' is {}, but should be {}!", file_name, parsed_rank_size, attr_.rank_size));
        } else if (parsed_dims != k_) {
            throw std::runtime_error(fmt::format("The number of nearest-neighbors in '{}' is {}, but should be {}!", file_name, parsed_dims, k_));
        }

        const index_type correct_rank_size = comm_.rank() == comm_.size() - 1 ? (attr_.total_size - (comm_.size() - 1) * attr_.rank_size) : attr_.rank_size;

        const sycl_lsh::get_linear_id<knn<layout, options_type, data_type>> get_linear_id_this{};
        const sycl_lsh::get_linear_id<knn<memory_layout::aos, options_type, data_type>> get_linear_id_aos{};

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

        const real_type res = (static_cast<real_type>(mpi::sum(count, comm_)) / (attr_.total_size * k_)) * 100.0;
        logger_.log("\nCalculated recall in {}.\n", t.elapsed());
        return res;
    }

    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    std::tuple<typename Options::real_type, typename Options::index_type, typename Options::index_type> knn<layout, Options, Data>::error_ratio(const argv_parser& parser) {
        mpi::timer t(comm_);

        // load correct k-nearest-neighbor distances
        // check if the required command line argument is present
        if (!parser.has_argv("evaluate_knn_dist_file")) {
            throw std::invalid_argument("Required command line argument 'evaluate_knn_dist_file' not provided!");
        }

        // read correct k-nearest-neighbor distances from the respective file
        const std::string& file_name = parser.argv_as<std::string>("evaluate_knn_dist_file");
        auto file_parser = mpi::make_file_parser<real_type, options_type>(file_name, parser, mpi::file::mode::read, comm_, logger_);
        const index_type parsed_total_size = file_parser->parse_total_size();
        const index_type parsed_rank_size = file_parser->parse_rank_size();
        const index_type parsed_dims = file_parser->parse_dims();
        dist_host_buffer_type correct_knn_dist = file_parser->parse_content();

        // perform sanity checks
        if (parsed_total_size != attr_.total_size) {
            throw std::runtime_error(fmt::format("The total number of points in '{}' is {}, but should be {}!", file_name, parsed_total_size, attr_.total_size));
        } else if (parsed_rank_size != attr_.rank_size) {
            throw std::runtime_error(fmt::format("The number of points per MPI rank in '{}' is {}, but should be {}!", file_name, parsed_rank_size, attr_.rank_size));
        } else if (parsed_dims != k_) {
            throw std::runtime_error(fmt::format("The number of nearest-neighbor distances in '{}' is {}, but should be {}!", file_name, parsed_dims, k_));
        }

        const index_type correct_rank_size = comm_.rank() == comm_.size() - 1 ? (attr_.total_size - (comm_.size() - 1) * attr_.rank_size) : attr_.rank_size;

        const sycl_lsh::get_linear_id<knn<layout, options_type, data_type>> get_linear_id_this{};
        const sycl_lsh::get_linear_id<knn<memory_layout::aos, options_type, data_type>> get_linear_id_aos{};

        // calculate error ratio
        index_type num_points = 0;
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
                ++num_points;
                num_knn_not_found += count_not_found;
                continue;
            }
            // calculate `std::sqrt` distance
            std::transform(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end(), calculated_knn_dist_sorted.begin(),
                           [](const real_type val) { return std::sqrt(val); });
            // sort distances
            std::sort(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end());
            std::sort(correct_knn_dist_sorted.begin(), correct_knn_dist_sorted.end());

            // calculate error ratio
            // TODO 2020-10-06 13:16 marcel: check again
            index_type error_count = 0;
            real_type error_ratio = 0.0;
            for (index_type nn = 0; nn < k_; ++nn) {
                if (calculated_knn_dist_sorted[nn] != 0.0 && correct_knn_dist_sorted[nn] != 0.0) {
                    ++error_count;
                    error_ratio += calculated_knn_dist_sorted[nn] / correct_knn_dist_sorted[nn];
                } else {
                    ++error_count;
                    ++error_ratio;
                }
            }
            if (error_count != 0) {
                ++mean_error_count;
                mean_error_ratio += error_ratio / error_count;
            }
        }

        // collect results from each MPI rank
        const real_type avg_mean_error_ratio = mpi::average(mean_error_ratio / mean_error_count, comm_);
        const index_type total_num_points = mpi::sum(num_points, comm_);
        const index_type total_num_knn_not_found = mpi::sum(num_knn_not_found, comm_);

        logger_.log("\nCalculated error ration in {}.\n", t.elapsed());
        return std::make_tuple(avg_mean_error_ratio, total_num_points, total_num_knn_not_found);
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                             update host buffer                                             //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    void knn<layout, Options, Data>::send_receive_host_buffer() {
        const int destination = (comm_.rank() + 1) % comm_.size();
        const int source = (comm_.size() + (comm_.rank() - 1) % comm_.size()) % comm_.size();

        // send/receive k-nearest-neighbor IDs
        MPI_Sendrecv_replace(knn_host_buffer_.data(), knn_host_buffer_.size(), mpi::type_cast<typename knn_host_buffer_type::value_type>(),
                             destination, 0, source, 0, comm_.get(), MPI_STATUS_IGNORE);

        // send/receive k-nearest-neighbor distances
        MPI_Sendrecv_replace(dist_host_buffer_.data(), dist_host_buffer_.size(), mpi::type_cast<typename dist_host_buffer_type::value_type>(),
                             destination, 0, source, 0, comm_.get(), MPI_STATUS_IGNORE);
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_KNN_HPP
