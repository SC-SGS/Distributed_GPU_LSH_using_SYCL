/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/hash_functions/entropy_based.hpp"

#include "sycl_lsh/data_set.hpp"            // sycl_lsh::data_set
#include "sycl_lsh/detail/assert.hpp"       // SYCL_LSH_ASSERT
#include "sycl_lsh/detail/device_ptr.hpp"   // sycl_lsh::detail::device_ptr
#include "sycl_lsh/mpi/communicator.hpp"    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/sort.hpp"     // sycl_lsh::mpi::detail::sort
#include "sycl_lsh/mpi/detail/utility.hpp"  // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/logger.hpp"          // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"           // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"             // sycl_lsh::options

#include "sycl/sycl.hpp"  // sycl::queue, sycl::handler, sycl::range, sycl::item

#include "mpi.h"  // MPI_Bcast, MPI_Allreduce

#include <random>  // std::mt19937, std::random_device, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution
#include <vector>  // std::vector

namespace sycl_lsh {

entropy_based::entropy_based(const device_accessible_options &opt, data_set &data, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger) :
    queue_{ queue },
    device_ptr_{ detail::shape{ opt.num_hash_tables, opt.num_hash_functions, data.get_attributes().dims + opt.num_cut_off_points - 1 }, queue_ } {
    const mpi::timer mpi_timer{ comm };

    const data_attributes attr = data.get_attributes();

    // create hash pool functions on MPI master rank and distribute to all other ranks
    std::vector<real_type> hash_functions_pool(opt.hash_pool_size * attr.dims);

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
        for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash_functions_pool[hash_function * attr.dims + dim] = rnd_normal_dist(rnd_normal_pool_gen);
            }
        }
    }

    // broadcast pool hash functions to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(hash_functions_pool.data(), static_cast<int>(hash_functions_pool.size()), mpi::detail::mpi_datatype<real_type>(), 0, comm.get()));

    std::vector<real_type> cut_off_points_pool(opt.hash_pool_size * (opt.num_cut_off_points - 1));

    // calculate cut-off points
    std::vector<real_type> hash_values(attr.rank_size * opt.hash_pool_size);
    {
        // copy the hash function pool to the device
        detail::device_ptr<real_type> hash_functions_pool_ptr{ detail::shape{ opt.hash_pool_size, attr.dims }, queue_ };
        hash_functions_pool_ptr.copy_to_device(hash_functions_pool);

        detail::device_ptr<real_type> hash_values_ptr{ detail::shape{ attr.rank_size, opt.hash_pool_size }, queue_ };

        queue_.submit([&](sycl::handler &cgh) {
            // get device data
            const real_type *data_d = data.get_device_ptr().get();
            const real_type *hash_functions_pool_d = hash_functions_pool_ptr.get();
            real_type *hash_values_d = hash_values_ptr.get();

            // get additional information
            const device_accessible_options options = opt;
            const data_attributes attribute = attr;

            cgh.parallel_for(sycl::range<1>{ attr.rank_size }, [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                for (index_type hash_function = 0; hash_function < options.hash_pool_size; ++hash_function) {
                    real_type value = 0.0;
                    for (index_type dim = 0; dim < attr.dims; ++dim) {
                        value += data_d[idx * attr.dims + dim]
                                 * hash_functions_pool_d[hash_function * attribute.dims + dim];
                    }
                    hash_values_d[hash_function * attribute.rank_size + idx] = value;
                }
            });
        });

        // wait until the kernel has finished
        queue_.wait_and_throw();

        // copy the hash values back to the host
        hash_values_ptr.copy_to_host(hash_values);
    }

#pragma omp parallel for
    for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
        // sort hash_values vector in a distributed fashion
        mpi::detail::sort(hash_values.begin() + hash_function * attr.rank_size, hash_values.begin() + (hash_function + 1) * attr.rank_size, comm);

        std::vector<real_type> cut_off_points(opt.num_cut_off_points - 1, 0.0);

        // calculate cut-off points indices
        std::vector<index_type> cut_off_points_idx(cut_off_points.size());
        const index_type jump = (attr.rank_size * comm.size()) / opt.num_cut_off_points;
        for (index_type cop = 0; cop < cut_off_points_idx.size(); ++cop) {
            cut_off_points_idx[cop] = (cop + 1) * jump;
        }

        // fill cut-off points which are located on the current MPI rank
        for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
            // check if index belongs to current MPI rank
            if (cut_off_points_idx[cop] >= attr.rank_size * comm.rank() && cut_off_points_idx[cop] < attr.rank_size * (comm.rank() + 1)) {
                cut_off_points[cop] = hash_values[hash_function * attr.rank_size + cut_off_points_idx[cop] % attr.rank_size];
            }
        }

        // combine to final cut-off points on all MPI ranks
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), static_cast<int>(cut_off_points.size()), mpi::detail::mpi_datatype<real_type>(), MPI_SUM, comm.get()));

        // copy current cut-off points to pool
        std::copy(cut_off_points.begin(), cut_off_points.end(), cut_off_points_pool.begin() + static_cast<std::vector<real_type>::difference_type>(hash_function * cut_off_points.size()));
    }

    // select actual hash functions
    std::vector<real_type> host_buffer(opt.num_hash_tables * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1));
    if (comm.is_main_rank()) {
// create random generator
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
                for (index_type dim = 0; dim < attr.dims; ++dim) {
                    host_buffer[hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1) + hash_function * (attr.dims + opt.num_cut_off_points - 1) + dim] = hash_functions_pool[pool_hash_function * attr.dims + dim];
                }
                for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                    host_buffer[hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1) + hash_function * (attr.dims + opt.num_cut_off_points - 1) + attr.dims + cop] = cut_off_points_pool[pool_hash_function * (opt.num_cut_off_points - 1) + cop];
                }
            }
        }
    }

    // broadcast hash function to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), static_cast<int>(host_buffer.size()), mpi::detail::mpi_datatype<real_type>(), 0, comm.get()));

    // copy the host data to the device
    device_ptr_.copy_to_device(host_buffer);

    logger.log("Created 'entropy_based' hash functions in {}.\n", mpi_timer.elapsed());
}

}  // namespace sycl_lsh
