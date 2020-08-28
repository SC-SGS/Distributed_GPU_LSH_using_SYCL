/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-28
 *
 * @brief Implements a device selector such that every MPI rank allocates only one device (GPU).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SINGLE_GPU_SELECTOR_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SINGLE_GPU_SELECTOR_HPP

#include <iostream>

#include <mpi.h>

#include <config.hpp>

/**
 * @brief SYCL device selector to select only one device per MPI rank.
 */
class single_gpu_selector : public sycl::device_selector {
public:
    /**
     * @brief Construct a new device selector.
     * @param[in] communicator the currently used MPI_Comm communicator
     */
    single_gpu_selector(const MPI_Comm& communicator) : sycl::device_selector{}, communicator_(communicator) { }

    /**
     * @brief Selects the GPU device equal to the current MPI rank.
     * @param[in] device the current device
     * @return the device score
     */
    int operator()(const sycl::device& device) const override {
        // only select GPUs
        if (device.get_info<sycl::info::device::device_type>() != sycl::info::device_type::gpu) return -1;

        int comm_rank;
        MPI_Comm_rank(communicator_, &comm_rank);

        auto device_list = device.get_platform().get_devices();
        for (std::size_t i = 0; i < device_list.size(); ++i) {
            // select GPU which equals to the current MPI rank
            if (device_list[i] == device && i == static_cast<std::size_t>(comm_rank)) {
                return 100;
            }
        }
        // TODO 2020-08-28 14:57 marcel: throw exception?
        return -1;
    }

private:
    /// The currently used MPI_Comm communicator.
    const MPI_Comm& communicator_;
};

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SINGLE_GPU_SELECTOR_HPP
