/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-08
 *
 * @brief Implements a device selector that every MPI rank allocates only **one** device based on the `SYCL_LSH_TARGET` specified during
 *        [CMake](https://cmake.org/)'s configuration step (e.g. NVIDIA GPU).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP

#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/mpi/communicator.hpp>

namespace sycl_lsh {

    /**
     * @brief SYCL device selector class to only select **one** device per MPI rank.
     */
    class device_selector final : public sycl::device_selector {
    public:
        /**
         * @brief Construct a new @ref sycl_lsh::device_selector.
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         */
        device_selector(const mpi::communicator& comm) : sycl::device_selector{}, comm_(comm) { }

        /**
         * @brief Selects the device equal to the current MIP rank.
         * @details The device type can be specified during [CMake](https://cmake.org/)'s configuration step.
         * @param[in] device the current `sycl::device`
         * @return the device score
         */
        int operator()([[maybe_unused]] const sycl::device& device) const override {
            // TODO 2020-10-02 17:25 marcel: implement
            return sycl::default_selector{}.operator()(device);
        }

    private:
        const mpi::communicator& comm_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP
