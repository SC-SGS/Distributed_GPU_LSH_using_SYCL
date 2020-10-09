/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-09
 *
 * @brief Implements a device selector that every MPI rank allocates only **one** device based on the `SYCL_LSH_TARGET` specified during
 *        [CMake](https://cmake.org/)'s configuration step (e.g. NVIDIA GPU).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP

#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/detail/utility.hpp>
#include <sycl_lsh/mpi/communicator.hpp>

#include <fmt/format.h>

#include <stdexcept>

namespace sycl_lsh {

    namespace detail {

        /**
         * @brief Compares the two devices @p lhs and @p rhs on equality.
         * @param[in] lhs a SYCL device
         * @param[in] rhs a SYCL device
         * @return `true` if the devices compare equal, `false` otherwise (`[[nodiscard]]`)
         */
        [[nodiscard]]
        inline bool compare_devices(const sycl_lsh::sycl::device& lhs, const sycl_lsh::sycl::device& rhs) {
            #if SYCL_LSH_IMPLEMENTATION == SYCL_LSH_IMPLEMENTATION_HIPSYCL
                return lhs == rhs;
            #else
                return lhs.get() == rhs.get();
            #endif
        }

    }

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

            #if SYCL_LSH_TARGET == SYCL_LSH_TARGET_NVIDIA
                // get platform associated with the current device
                auto platform = device.get_platform();
                // check if we are currently on a NVIDIA platform as requested
                if (detail::contains(platform.get_info<sycl::info::platform::name>(), "NVIDIA CUDA")) {
                    auto device_list = platform.get_devices();
                    // check whether the current platform has enough devices to satisfy the requested number of slots
                    if (device_list.size() < static_cast<std::size_t>(comm_.size())) {
                        throw std::runtime_error(fmt::format("Found {} devices, but need {} devices to satisfy the requested number of slots!",
                                device_list.size(), comm_.size()));
                    }

                    // select current device, if the current device is the ith device in the list given the current MPI rank is i
                    if (detail::compare_devices(device_list[comm_.rank()], device)) {
                        return 100;
                    }
                }
                // never choose current device otherwise
                return -1;
            #else
                return sycl::default_selector{}.operator()(device);
            #endif
        }

    private:
        const mpi::communicator& comm_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEVICE_SELECTOR_HPP
