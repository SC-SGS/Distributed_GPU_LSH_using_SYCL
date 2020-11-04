/**
 * @file
 * @author Marcel Breyer
 * @date 2020-11-04
 *
 * @brief Implements the @ref sycl_lsh::hash_tables class representing the used LSH hash tables.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/data.hpp>
#include <sycl_lsh/data_attributes.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/detail/utility.hpp>
#include <sycl_lsh/device_selector.hpp>
#include <sycl_lsh/hash_functions/hash_functions.hpp>
#include <sycl_lsh/knn.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/memory_layout.hpp>
#include <sycl_lsh/options.hpp>
#include <sycl_lsh/mpi/timer.hpp>

#include <fmt/format.h>

#include <cmath>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace sycl_lsh {

    // SYCL kernel name needed to silence ComputeCpp warnings
    class kernel_count_hash_values;
    class kernel_calculate_offsets;
    class kernel_fill_hash_tables;
    class kernel_calculate_knn;

    // forward declare hash_tables class
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    class hash_tables;

    /**
     * @brief Factory function for the @ref sycl_lsh::hash_tables class.
     * @brief Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
     * @tparam layout the used @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     * @return the @ref sycl_lsh::hash_tables object representing the hash tables used in the LSH algorithm (`[[nodiscard]]`)
     */
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]] 
    auto make_hash_tables(const Options& opt, Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        using type_of_hash_functions = detail::get_hash_functions_type_t<layout, Options, Data, Options::used_hash_functions_type>;
        return hash_tables<layout, Options, Data, type_of_hash_functions>(opt, data, comm, logger);
    }


    /**
     * @brief Class which represents the hash tables used om the LSH algorithm. Performs the actual calculation of the k-nearest-neighbors.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @tparam HashFunctionType the used type of hash functions in the LSH algorithm
     */
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    class hash_tables final : private detail::hash_tables_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must be a sycl_lsh::data type!");
        static_assert(std::is_base_of_v<detail::hash_functions_base, HashFunctionType>, "The fourth template parameter must be a hash function type!");
    public:
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                type aliases                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /// The type of the @ref sycl_lsh::options object.
        using options_type = Options;
        /// The used floating point type.
        using real_type = typename options_type::real_type;
        /// The used integral type (used for indices).
        using index_type = typename options_type::index_type;
        /// The used unsigned type (used for the hash values).
        using hash_value_type = typename options_type::hash_value_type;

        /// The type of the @ref sycl_lsh::data object.
        using data_type = Data;
        /// The type of the @ref sycl_lsh::data_attributes object.
        using data_attributes_type = typename data_type::data_attributes_type;
        /// The type of the device buffer used in the @ref sycl_lsh::data object.
        using data_device_buffer_type = typename data_type::device_buffer_type;
        /// The type of the host buffer used in the @ref sycl_lsh::data object.
        using data_host_buffer_type = typename data_type::host_buffer_type;

        /// The type of the @ref sycl_lsh::knn object as the result of the k-nearest-neighbor search.
        using knn_type = knn<layout, options_type, data_type>;
        using knn_device_buffer_type = sycl::buffer<index_type, 1>;
        using knn_dist_device_buffer_type = sycl::buffer<real_type, 1>;

        /// The type of the used LSH hash functions.
        using hash_function_type = HashFunctionType;

        /// The type of the device buffer used by SYCL.
        using device_buffer_type = sycl::buffer<index_type, 1>;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                       calculate k-nearest-neighbors                                        //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Calculate the k-nearest-neighbors using **Locality Sensitive Hashing**, **SYCL** and **MPI**.
         * @param[in] parser the used @ref sycl_lsh::argv_parser to get the number of nearest-neighbors to search for from
         * @return the found k-nearest-neighbors (`[[nodiscard]]`)
         *
         * @throws std::invalid_argument if the number of nearest-neighbors @p k is less or equal than `0` or greater and equal than `rank_size`.
         */
        [[nodiscard]] 
        knn_type get_k_nearest_neighbors(const argv_parser& parser);
        /**
         * @brief Calculate the k-nearest-neighbors using **Locality Sensitive Hashing**, **SYCL** and **MPI**.
         * @param[in] k the number of nearest-neighbors to search for
         * @return the found k-nearest-neighbors (`[[nodiscard]]`)
         *
         * @throws std::invalid_argument if the number of nearest-neighbors @p k is less or equal than `0` or greater and equal than `rank_size`.
         */
        [[nodiscard]]
        knn_type get_k_nearest_neighbors(const index_type k);
        /**
         * @brief Performs the k-nearest-neighbor search given the data set @p data_buffer and already calculate nearest-neighbors @p knns.
         * @param[in] k the number of nearest neighbors to search for
         * @param[in] data_buffer the data to perfrom the nearest-neighbors search on
         * @param[in,out] knns the (already partially) calculated nearest-neighbors
         */
        void calculate_knn_round(const index_type k, data_device_buffer_type& data_buffer, knn_type& knns);


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
         * @brief Returns the @ref sycl_lsh::options object used to control the behavior of the used algorithm.
         * @return the @ref sycl_lsh::options (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const options_type get_options() const noexcept { return options_; }
        /**
         * @brief Returns the @ref sycl_lsh::data object representing the used data set.
         * @return the @ref sycl_lsh::data (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const data_type& get_data() const noexcept { return data_; }

    private:
        // befriend factory function
        friend auto make_hash_tables<layout, Options, Data>(const options_type&, data_type&, const mpi::communicator&, const mpi::logger&);

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Constructs a new @ref sycl_lsh::hash_tables object initializing the LSH hash tables.
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] data the used @ref sycl_lsh::data representing the used data set
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         */
        hash_tables(const options_type& opt, data_type& data, const mpi::communicator& comm, const mpi::logger& logger);

        /**
         * @brief Calculate the number of data points assigned to each hash bucket in each hash table.
         * @param[in,out] hash_values_count the number of data points per hash bucket
         */
        void count_hash_values(device_buffer_type& hash_values_count);
        /**
         * @brief Calculates the offset of each hash bucket in each hash table.
         * @param[in] hash_values_count the number of data points per hash bucket
         */
        void calculate_offsets(device_buffer_type& hash_values_count);
        /**
         * @brief Fill each hash table based on the previously calculated offsets.
         */
        void fill_hash_tables();


        const options_type& options_;
        data_type& data_;
        const data_attributes_type attr_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        hash_function_type hash_functions_;

        sycl::queue queue_;
        device_buffer_type hash_tables_buffer_;
        device_buffer_type offsets_buffer_;
    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                       calculate k-nearest-neighbors                                        //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    [[nodiscard]]
    typename sycl_lsh::hash_tables<layout, Options, Data, HashFunctionType>::knn_type
    sycl_lsh::hash_tables<layout, Options, Data, HashFunctionType>::get_k_nearest_neighbors(const sycl_lsh::argv_parser& parser) {
        return get_k_nearest_neighbors(parser.argv_as<index_type>("k"));
    }

    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    [[nodiscard]]
    typename sycl_lsh::hash_tables<layout, Options, Data, HashFunctionType>::knn_type
    sycl_lsh::hash_tables<layout, Options, Data, HashFunctionType>::get_k_nearest_neighbors(const index_type k) {
        mpi::timer t(comm_);

        if (k < 1 || k > attr_.rank_size) {
            throw std::invalid_argument(fmt::format("k ({}) must be in the range [1, number of data point per MPI rank ({}))!", k, attr_.rank_size));
        }

        knn_type knns = make_knn<layout>(k, options_, data_, comm_, logger_);

        for (int round = 0; round < comm_.size(); ++round) {
            mpi::timer rt(comm_);

            logger_.log("Round {} of {} ... ", round + 1, comm_.size());
            // create thread to asynchronously perform MPI communication
            std::thread mpi_thread(&data_type::send_receive_host_buffer, &data_);

            // calculate k-nearest-neighbors on current MPI rank
            if (round == 0) {
                // use data already on device in round 0
                calculate_knn_round(k, data_.get_device_buffer(), knns);
            } else {
                // copy received data to device in other rounds
                data_host_buffer_type data_host_buffer = data_.get_host_buffer();
                data_device_buffer_type data_device_buffer(data_host_buffer.size());

                // copy data to device buffer
                auto acc = data_device_buffer.template get_access<sycl::access::mode::discard_write>();
                for (index_type i = 0; i < data_host_buffer.size(); ++i) {
                    acc[i] = data_host_buffer[i];
                }

                calculate_knn_round(k, data_device_buffer, knns);
            }

            // send calculated k-nearest-neighbors and distances to next rank
            knns.send_receive_host_buffer();
            // wait until all MPI communication has been finished
            mpi_thread.join();
            comm_.wait();

            logger_.log("finished in {}.\n", rt.elapsed());
        }

        logger_.log("Calculated {}-nearest-neighbors in {}.\n\n", k, t.elapsed());

        return knns;
    }

    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void sycl_lsh::hash_tables<layout, Options, Data, HashFunctionType>::calculate_knn_round(const index_type k, data_device_buffer_type& data_buffer, knn_type& knns) {

        // TODO 2020-10-07 15:52 marcel: check if correct and useful
        const index_type local_mem_size = queue_.get_device().template get_info<sycl::info::device::local_mem_size>();
        const index_type max_local_size = local_mem_size / (k * sizeof(index_type) + k * sizeof(real_type));
        const index_type max_work_group_size = queue_.get_device().template get_info<sycl::info::device::max_work_group_size>();
        const index_type local_size = std::min<index_type>(std::pow(2, std::floor(std::log2(max_local_size))), max_work_group_size);
        const index_type global_size = ((attr_.rank_size + local_size - 1) / local_size) * local_size;

        // create SYCL buffers for knn class
        knn_device_buffer_type knn_buffer(knns.get_knn_host_buffer().data(), knns.get_knn_host_buffer().size());
        knn_dist_device_buffer_type knn_dist_buffer(knns.get_distance_host_buffer().data(), knns.get_distance_host_buffer().size());

        queue_.submit([&](sycl::handler& cgh) {
            // get accessors
            auto acc_data_owned = data_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_data_received = data_buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_functions_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets_buffer_.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_tables = hash_tables_buffer_.template get_access<sycl::access::mode::read>(cgh);
            auto acc_knn = knn_buffer.template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_knn_dist = knn_dist_buffer.template get_access<sycl::access::mode::read_write>(cgh);
            // get additional information
            auto options = options_;
            auto attr = attr_;
            const index_type base_id = comm_.rank() * attr_.rank_size;
            // get get_linear_id functor instantiation
            const get_linear_id<data_type> get_linear_id_data{};
            const get_linear_id<knn_type> get_linear_id_knn{};
            // get hasher functor instantiation
            const lsh_hash<hash_function_type> hasher{};

            // create local memory accessors
            sycl::accessor<index_type, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    knn_local_mem(sycl::range<>(local_size * k), cgh);
            sycl::accessor<real_type, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    knn_dist_local_mem(sycl::range<>(local_size * k), cgh);

            const auto execution_range = sycl::nd_range<>(sycl::range<>(global_size), sycl::range<>(local_size));

            cgh.parallel_for<kernel_calculate_knn>(execution_range, [=](sycl::nd_item<> item) {
                const index_type global_idx = item.get_global_linear_id();
                const index_type local_idx  = item.get_local_linear_id();

                // immediately return if global_idx is out-of-range
                if (global_idx >= attr.rank_size) return;

                index_type knn_blocked[options_type::blocking_size];
                real_type knn_dist_blocked[options_type::blocking_size];

                // initialize local memory arrays
                for (index_type nn = 0; nn < k; ++nn) {
                    knn_local_mem[local_idx * k + nn] = acc_knn[get_linear_id_knn(global_idx, nn, attr, k)];
                    knn_dist_local_mem[local_idx * k + nn] = acc_knn_dist[get_linear_id_knn(global_idx, nn, attr, k)];
                }

                // perform nearest-neighbor search for all hash tables
                for (index_type hash_table = 0; hash_table < options.num_hash_tables; ++hash_table) {
                    // calculate hash value (= hash bucket) for current point
                    const hash_value_type hash_bucket = hasher(hash_table, global_idx, acc_data_received, acc_hash_functions, options, attr);

                    // calculate hash bucket offsets
                    const index_type bucket_begin = acc_offsets[hash_table * (options.hash_table_size + 1) + hash_bucket];
                    const index_type bucket_end   = acc_offsets[hash_table * (options.hash_table_size + 1) + hash_bucket + 1];

                    // perform nearest-neighbor search for all data points in the calculate hash bucket
                    for (index_type bucket_elem = bucket_begin; bucket_elem < bucket_end; bucket_elem += options_type::blocking_size) {
                        // initialize thread local blocking array
                        for (index_type block = 0; block < options_type::blocking_size; ++block) {
                            knn_blocked[block] = acc_hash_tables[hash_table * attr.rank_size + bucket_elem + block];
                            knn_dist_blocked[block] = 0.0;
                        }

                        // calculate distances
                        for (index_type block = 0; block < options_type::blocking_size; ++block) {
                            for (index_type dim = 0; dim < attr.dims; ++dim) {
                                const real_type x = acc_data_received[get_linear_id_data(global_idx, dim, attr)];
                                const real_type y = acc_data_owned[get_linear_id_data(knn_blocked[block] - base_id, dim, attr)];
                                knn_dist_blocked[block] += (x - y) * (x - y);
                            }
                        }

                        // check candidate function
                        const auto is_candidate = [&](const index_type candidate_idx, const index_type global_idx, const index_type local_idx) {
                            if (candidate_idx - base_id == global_idx) return false;
                            for (index_type nn = 0; nn < k; ++nn) {
                                if (knn_local_mem[local_idx * k + nn] == candidate_idx) return false;
                            }
                            return true;
                        };

                        // update nearest-neighbors
                        for (index_type block = 0; block < options_type::blocking_size; ++block) {
                            if (knn_dist_blocked[block] < knn_dist_local_mem[local_idx * k] && is_candidate(knn_blocked[block], global_idx, local_idx)) {
                                knn_local_mem[local_idx * k] = knn_blocked[block];
                                knn_dist_local_mem[local_idx * k] = knn_dist_blocked[block];

                                // ensure that the greatest distance is at pos 0
                                for (index_type nn = 0; nn < k - 1; ++nn) {
                                    if (knn_dist_local_mem[local_idx * k + nn] < knn_dist_local_mem[local_idx * k + nn + 1]) {
                                        detail::swap(knn_local_mem[local_idx * k + nn], knn_local_mem[local_idx * k + nn + 1]);
                                        detail::swap(knn_dist_local_mem[local_idx * k + nn], knn_dist_local_mem[local_idx * k + nn + 1]);
                                    }
                                }
                            }
                        }
                    }
                }

                // write back to global buffer
                for (index_type nn = 0; nn < k; ++nn) {
                    acc_knn[get_linear_id_knn(global_idx, nn, attr, k)] = knn_local_mem[local_idx * k + nn];
                    acc_knn_dist[get_linear_id_knn(global_idx, nn, attr, k)] = knn_dist_local_mem[local_idx * k + nn];
                }
            });
        });

        // wait until all k-nearest-neighbors were calculated on the current MPI rank
        queue_.wait_and_throw();
    }



    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    hash_tables<layout, Options, Data, HashFunctionType>::hash_tables(const Options& opt, Data& data, const mpi::communicator& comm, const mpi::logger& logger)
            : options_(opt), data_(data), attr_(data.get_attributes()), comm_(comm), logger_(logger),
              hash_functions_(opt, data, comm, logger),
              queue_(device_selector{comm}, sycl::async_handler(&sycl_exception_handler)),
              hash_tables_buffer_(opt.num_hash_tables * data.get_attributes().rank_size + options_type::blocking_size),
              offsets_buffer_(opt.num_hash_tables * (opt.hash_table_size + 1))
    {
        // log used devices
        logger_.log_on_all("[{}, {}]\n", comm_.rank(), queue_.get_device().template get_info<sycl::info::device::name>());
        mpi::timer t(comm_);

        {
            // create temporary buffer to count the occurrence of each hash value
            device_buffer_type hash_values_count(options_.num_hash_tables * options_.hash_table_size);

            // count the occurrence of each hash value per hash table
            this->count_hash_values(hash_values_count);
            // calculate the offset values
            this->calculate_offsets(hash_values_count);
        }
        // fill the hash tables based on the previously calculated offsets
        this->fill_hash_tables();

        logger_.log("Created hash tables in {}.\n", t.elapsed());
    }

    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void hash_tables<layout, Options, Data, HashFunctionType>::count_hash_values(device_buffer_type& hash_values_count) {
        mpi::timer t(comm_);

        queue_.submit([&](sycl::handler& cgh) {
            // get accessors
            auto acc_hash_values_count = hash_values_count.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_functions = hash_functions_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_data = data_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            // get additional information
            auto options = options_;
            auto attr = attr_;
            // get hasher functor instantiation
            const lsh_hash<hash_function_type> hasher{};

            cgh.parallel_for<kernel_count_hash_values>(sycl::range<>(attr.rank_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                for (index_type hash_table = 0; hash_table < options.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value = hasher(hash_table, idx, acc_data, acc_hash_functions, options, attr);
                    acc_hash_values_count[hash_table * options.hash_table_size + hash_value].fetch_add(1);
                }
            });
        });
        #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
            queue_.wait_and_throw();
        #endif

        logger_.log("Counted hash values in {}.\n", t.elapsed());
    }
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void hash_tables<layout, Options, Data, HashFunctionType>::calculate_offsets(device_buffer_type& hash_values_count) {
        mpi::timer t(comm_);

        queue_.submit([&](sycl::handler& cgh) {
            // get accessors
            auto acc_hash_values_count = hash_values_count.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offset = offsets_buffer_.template get_access<sycl::access::mode::discard_write>(cgh);
            // get additional information
            auto options = options_;

            cgh.parallel_for<kernel_calculate_offsets>(sycl::range<>(options.num_hash_tables), [=](sycl::item<> item){
                const index_type idx = item.get_linear_id();

                // calculate constant offsets
                const index_type hash_table_offset = idx * (options.hash_table_size + 1);
                const index_type hash_value_count_offset = idx * options.hash_table_size;
                // zero out the first two offsets in each hash table
                acc_offset[hash_table_offset] = 0;
                acc_offset[hash_table_offset + 1] = 0;
                // fill remaining offset values
                for (index_type hash_value = 2; hash_value <= options.hash_table_size; ++hash_value) {
                    // calculated modified prefix sum
                    acc_offset[hash_table_offset + hash_value] =
                            acc_offset[hash_table_offset + hash_value - 1] +
                            acc_hash_values_count[hash_value_count_offset + hash_value - 2];
                }
            });
        });
        #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
            queue_.wait_and_throw();
        #endif

        logger_.log("Calculated offsets in {}.\n", t.elapsed());
    }
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void hash_tables<layout, Options, Data, HashFunctionType>::fill_hash_tables() {
        mpi::timer t(comm_);

        queue_.submit([&](sycl::handler& cgh) {
            // get accessors
            auto acc_data = data_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_functions_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets_buffer_.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_tables = hash_tables_buffer_.template get_access<sycl::access::mode::write>(cgh);
            // get additional information
            auto options = options_;
            auto attr = attr_;
            const index_type base_id = comm_.rank() * attr_.rank_size;
            const index_type comm_rank = comm_.rank();
            const index_type comm_size = comm_.size();
            // get hasher functor instantiation
            const lsh_hash<hash_function_type> hasher{};

            cgh.parallel_for<kernel_fill_hash_tables>(sycl::range<>(attr.rank_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                index_type val = base_id + idx;
                if (comm_rank == comm_size - 1) {
                    // set correct values IDs for dummy points
                    const index_type correct_rank_size = attr.total_size - ((comm_size - 1) * attr.rank_size);
                    if (idx >= correct_rank_size) {
                        val = base_id + correct_rank_size - 1;
                    }
                }

                for (index_type hash_table = 0; hash_table < options.num_hash_tables; ++hash_table) {
                    // get hash value
                    const hash_value_type hash_value = hasher(hash_table, idx, acc_data, acc_hash_functions, options, attr);
                    // update offsets
                    const index_type hash_table_idx = acc_offsets[hash_table * (options.hash_table_size + 1) + hash_value + 1].fetch_add(1);
                    acc_hash_tables[hash_table * attr.rank_size + hash_table_idx] = val;
                }
                
                // fill additional values needed for blocking
                if (idx == attr.rank_size - 1) {
                    for (index_type block = 0; block < options_type::blocking_size; ++block) {
                        acc_hash_tables[options.num_hash_tables * attr.rank_size + block] = val;
                    }
                }
            });
        });
        #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
            queue_.wait_and_throw();
        #endif

        logger_.log("Filled hash tables in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
