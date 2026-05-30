/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/detail/hashing/random_projections.hpp"

#include "sycl_lsh/data_set.hpp"            // sycl_lsh::data_set::attributes
#include "sycl_lsh/detail/device_ptr.hpp"   // sycl_lsh::detail::device_ptr
#include "sycl_lsh/mpi/communicator.hpp"    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/logging.hpp"  // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/detail/utility.hpp"  // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/timer.hpp"           // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"             // sycl_lsh::locality_sensitive_hashing_options

#include "sycl/sycl.hpp"  // sycl::queue

#include "mpi.h"  // MPI_Bcast

#include <cmath>   // std::abs
#include <random>  // std::mt19937, std::random_device, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution
#include <vector>  // std::vector

namespace sycl_lsh::detail::hashing {

random_projections::random_projections(const locality_sensitive_hashing_options &opt, const device_ptr<real_type> &, const data_set::attributes attributes, sycl::queue &queue, const mpi::communicator &comm) :
    queue_{ queue },
    device_ptr_{ shape{ opt.num_hash_tables, opt.num_hash_functions, (attributes.dims + 1) }, queue_ } {
    const mpi::timer mpi_timer{ comm };

    std::vector<real_type> host_buffer(device_ptr_.size());

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
                    host_buffer[hash_table * opt.num_hash_functions * (attributes.dims + 1) + hash_function * (attributes.dims + 1) + dim] = hash_pool[pool_hash_function * (attributes.dims + 1) + dim];
                }
            }
        }
    }

    // broadcast hash functions to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::detail::mpi_datatype<real_type>(), 0, comm));

    // copy the host data to the device
    device_ptr_.copy_to_device(host_buffer);

    mpi::detail::log(comm, "Created 'random_projections' hash functions in {}.\n", mpi_timer.elapsed());
}

}  // namespace sycl_lsh::detail::hashing
