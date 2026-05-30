/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/hash_functions/mixed_hash_functions.hpp"

#include "sycl_lsh/data_attributes.hpp"     // sycl_lsh::data_attributes
#include "sycl_lsh/detail/device_ptr.hpp"   // sycl_lsh::detail::device_ptr
#include "sycl_lsh/mpi/communicator.hpp"    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/sort.hpp"     // sycl_lsh::mpi::detail::sort
#include "sycl_lsh/mpi/detail/utility.hpp"  // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/logger.hpp"          // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"           // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"             // sycl_lsh::locality_sensitive_hashing_options

#include "sycl/sycl.hpp"  // sycl::queue, sycl::handler, sycl::range, sycl::item

#include "mpi.h"  // MPI_Bcast, MPI_Allreduce

#include <random>  // std::mt19937, std::random_device, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution
#include <vector>  // std::vector

namespace sycl_lsh {

mixed_hash_functions::mixed_hash_functions(const locality_sensitive_hashing_options &opt, const detail::device_ptr<real_type> &data, const data_attributes attributes, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger) :
    queue_{ queue },
    device_ptr_{ opt.num_hash_tables * opt.num_hash_functions * (attributes.dims + 1) +            // random projections as hash functions
                     opt.num_hash_tables * (opt.num_hash_functions + opt.num_cut_off_points - 1),  // entropy-based as hash combine
                 queue_ } {
    const mpi::timer mpi_timer{ comm };

    std::vector<real_type> host_buffer(device_ptr_.size());

    //
    // CREATE RANDOM PROJECTIONS HASH FUNCTIONS
    //

    // create hash pool only on MPI master rank
    if (comm.is_main_rank()) {
// create random generators
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_normal_pool_gen{};
        std::mt19937 rnd_uniform_pool_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_pool_device{};
        std::mt19937 rnd_normal_pool_gen{ rnd_pool_device() };
        std::mt19937 rnd_uniform_pool_gen{ rnd_pool_device() };
#endif
        std::normal_distribution<real_type> rnd_normal_pool_dist{};
        std::uniform_real_distribution<real_type> rnd_uniform_pool_dist{ 0, opt.w };

        // fill hash pool
        std::vector<real_type> hash_pool(opt.hash_pool_size * (attributes.dims + 1));
        for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
            for (index_type dim = 0; dim < attributes.dims; ++dim) {
                hash_pool[hash_function * (attributes.dims + 1) + dim] = std::abs(rnd_normal_pool_dist(rnd_normal_pool_gen));
            }
            hash_pool[hash_function * (attributes.dims + 1) + attributes.dims] = rnd_uniform_pool_dist(rnd_uniform_pool_gen);
        }

// select actual hash functions
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_uniform_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_device{};
        std::mt19937 rnd_uniform_gen{ rnd_device() };
#endif
        std::uniform_int_distribution<index_type> rnd_uniform_dist{ 0, opt.hash_pool_size - 1 };

        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                for (index_type dim = 0; dim <= attributes.dims; ++dim) {
                    host_buffer[hash_table * (opt.num_hash_functions * (attributes.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + hash_function * (attributes.dims + 1) + dim] = hash_pool[pool_hash_function * (attributes.dims + 1) + dim];
                }
            }
        }
    }

    //
    // CREATE ENTROPY-BASED HASH FUNCTIONS
    //

    if (comm.is_main_rank()) {
// create random generator
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_normal_pool_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_pool_device{};
        std::mt19937 rnd_normal_pool_gen{ rnd_pool_device() };
#endif
        std::normal_distribution<real_type> rnd_normal_dist{};

        // fill hash functions
        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                host_buffer[hash_table * (opt.num_hash_functions * (attributes.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + opt.num_hash_functions * (attributes.dims + 1) + hash_function] = rnd_normal_dist(rnd_normal_pool_gen);
            }
        }
    }

    // broadcast random projections hash functions to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::detail::mpi_datatype<real_type>(), 0, comm));

    // calculate cut-off points
    std::vector<real_type> hash_values(attributes.rank_size * opt.num_hash_tables);
    {
        // copy the hash function pool to the device
        detail::device_ptr<real_type> hash_functions_ptr{ host_buffer.size(), queue_ };
        hash_functions_ptr.copy_to_device(host_buffer);

        detail::device_ptr<real_type> hash_values_ptr{ detail::shape{ attributes.rank_size, opt.num_hash_tables }, queue_ };

        queue_.submit([&](sycl::handler &cgh) {
            // get device data
            const real_type *data_d = data.get();
            const real_type *hash_functions_d = hash_functions_ptr.get();
            real_type *hash_values_d = hash_values_ptr.get();

            // get additional information
            const locality_sensitive_hashing_options options = opt;
            const data_attributes attr = attributes;

            cgh.parallel_for(sycl::range<2>{ opt.num_hash_tables, attributes.rank_size }, [=](sycl::item<2> item) {
                const index_type idx = item.get_id(1);
                const index_type hash_table = item.get_id(0);

                real_type value = 0.0;
                for (index_type hash_function = 0; hash_function < options.num_hash_functions; ++hash_function) {
                    real_type hash = hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + hash_function * (attr.dims + 1) + attr.dims];
                    for (index_type dim = 0; dim < attr.dims; ++dim) {
                        hash += data_d[idx * attr.dims + dim]
                                * hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + hash_function * (attr.dims + 1) + dim];
                    }
                    value += static_cast<hash_value_type>(hash / options.w)
                             * hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + opt.num_hash_functions * (attr.dims + 1) + hash_function];
                }
                hash_values_d[hash_table * attr.rank_size + idx] = value;
            });
        });

        // wait until the kernel has finished
        queue_.wait_and_throw();

        // copy the hash values back to the host
        hash_values_ptr.copy_to_host(hash_values);
    }

#pragma omp parallel for
    for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
        // sort hash_values vector in a distributed fashion
        mpi::detail::sort(hash_values.begin() + hash_table * attributes.rank_size, hash_values.begin() + (hash_table + 1) * attributes.rank_size, comm);

        std::vector<real_type> cut_off_points(opt.num_cut_off_points - 1, 0.0);

        // calculate cut-off points indices
        std::vector<index_type> cut_off_points_idx(cut_off_points.size());
        const index_type jump = (attributes.rank_size * comm.size()) / opt.num_cut_off_points;
        for (index_type cop = 0; cop < cut_off_points_idx.size(); ++cop) {
            cut_off_points_idx[cop] = (cop + 1) * jump;
        }

        // fill cut-off points which are located on the current MPI rank
        for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
            // check if index belongs to current MPI rank
            if (cut_off_points_idx[cop] >= attributes.rank_size * comm.rank() && cut_off_points_idx[cop] < attributes.rank_size * (comm.rank() + 1)) {
                cut_off_points[cop] = hash_values[hash_table * attributes.rank_size + cut_off_points_idx[cop] % attributes.rank_size];
            }
        }

        // combine to final cut-off points on all MPI ranks
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), cut_off_points.size(), mpi::detail::mpi_datatype<real_type>(), MPI_SUM, comm));

        // copy current cut-off points to hash functions
        for (index_type cop = 0; cop < cut_off_points.size(); ++cop) {
            host_buffer[hash_table * (opt.num_hash_functions * (attributes.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + opt.num_hash_functions * (attributes.dims + 1) + opt.num_hash_functions + cop] = cut_off_points[cop];
        }
    }

    // broadcast hash function to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::detail::mpi_datatype<real_type>(), 0, comm));

    // copy the host data to the device
    device_ptr_.copy_to_device(host_buffer);

    logger.log("Created 'mixed_hash_functions' hash functions in {}.\n", mpi_timer.elapsed());
}

}  // namespace sycl_lsh
