/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-31
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
// target CPU
#if SYCL_TARGET == 0

    if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::cpu) {
        return 100;
    } else {
        return -1;
    }

#else // target GPU

        // get MPI rank nad size
        int comm_rank, comm_size;
        MPI_Comm_rank(communicator_, &comm_rank);
        MPI_Comm_size(communicator_, &comm_size);

        // only select GPUs
        if (device.get_info<sycl::info::device::device_type>() != sycl::info::device_type::gpu) return -1;

        // get list of devices
        auto device_list = device.get_platform().get_devices();
        if (device_list.size() < static_cast<std::size_t>(comm_size)) {
            throw std::logic_error("Not enough devices to satisfy the number of MPI processes!");
        }
        for (std::size_t i = 0; i < device_list.size(); ++i) {
            // select GPU which equals to the current MPI rank
            if (device_list[i] == device && i == static_cast<std::size_t>(comm_rank)) {
                return 100;
            }
        }
        return -1;
        // TODO 2020-08-31 13:34 marcel: check multi node GPU
#endif
    }

private:
    /// The currently used MPI_Comm communicator.
    const MPI_Comm& communicator_;
};

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SINGLE_GPU_SELECTOR_HPP
