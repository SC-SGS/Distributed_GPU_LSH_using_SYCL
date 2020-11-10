/**
 * @file
 * @author Marcel Breyer
 * @date 2020-11-10
 *
 * @brief Implements a device selector that every MPI rank allocates only **one** device based on the `SYCL_LSH_TARGET` specified during
 *        [CMake](https://cmake.org/)'s configuration step (e.g. NVIDIA GPU).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP

#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/mpi/communicator.hpp>


namespace sycl_lsh {

    namespace detail {

        /**
         * @brief Select exactly one NVIDIA GPU device per MPI rank.
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         *
         * @note Only supports NVIDIA GPUs!
         */
        void setup_cuda_devices(const sycl_lsh::mpi::communicator& comm);

        /**
         * @brief Compares the two devices @p lhs and @p rhs on equality.
         * @param[in] lhs a SYCL device
         * @param[in] rhs a SYCL device
         * @return `true` if the devices compare equal, `false` otherwise (`[[nodiscard]]`)
         */
        [[nodiscard]]
        bool compare_devices(const sycl_lsh::sycl::device& lhs, const sycl_lsh::sycl::device& rhs);

    }

    /**
     * @brief SYCL device selector class to only select **one** device per MPI rank.
     */
    class device_selector final : public sycl::device_selector {
    public:
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::device_selector.
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         */
        device_selector(const mpi::communicator& comm);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                              device selecting                                              //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Selects the device equal to the current MIP rank.
         * @details The device type can be specified during [CMake](https://cmake.org/)'s configuration step.
         * @param[in] device the current `sycl::device`
         * @return the device score
         */
        int operator()([[maybe_unused]] const sycl::device& device) const override;

    private:
        const mpi::communicator& comm_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP
