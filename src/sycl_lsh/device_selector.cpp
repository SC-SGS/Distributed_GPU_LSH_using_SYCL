/**
 * @file
 * @author Marcel Breyer
 * @date 2020-12-01
 */

#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/detail/utility.hpp>
#include <sycl_lsh/device_selector.hpp>
#include <sycl_lsh/exceptions/not_implemented_exception.hpp>
#include <sycl_lsh/mpi/communicator.hpp>

#include <fmt/format.h>

#include <stdexcept>
#include <string>
#include <string_view>


void sycl_lsh::detail::setup_devices(const sycl_lsh::mpi::communicator& comm, const std::string& env_var_name) {
    // create communicator for each node
    MPI_Comm node_communicator;
    sycl_lsh::mpi::communicator node_comm(node_communicator, true);
    MPI_Comm_split_type(comm.get(), MPI_COMM_TYPE_SHARED, comm.size(), MPI_INFO_NULL, &node_comm.get());

    // set a CUDA_VISIBLE_DEVICES for each MPI process on the current rank
    int err = setenv(env_var_name.c_str(), std::to_string(node_comm.rank()).c_str(), 1);
    if (err != 0) {
        throw std::logic_error("Error while setting CUDA_VISIBLE_DEVICES environment variable!");
    }
    if (const char* env_val = getenv(env_var_name.c_str()); env_val != nullptr) {
        // fmt::print("Used CUDA device on world rank {} and node rank {}: CUDA_VISIBLE_DEVICES={}\n", comm.rank(), node_comm.rank(), env_val);
    }
}


[[nodiscard]]
bool sycl_lsh::detail::compare_devices(const sycl_lsh::sycl::device& lhs, const sycl_lsh::sycl::device& rhs) {
    #if SYCL_LSH_IMPLEMENTATION == SYCL_LSH_IMPLEMENTATION_HIPSYCL
        return lhs == rhs;
    #else
        return lhs.get() == rhs.get();
    #endif
}


sycl_lsh::device_selector::device_selector(const sycl_lsh::mpi::communicator& comm) : sycl::device_selector{}, comm_(comm) { }


int sycl_lsh::device_selector::operator()([[maybe_unused]] const sycl_lsh::sycl::device& device) const {
    #if SYCL_LSH_TARGET == SYCL_LSH_TARGET_CPU
        // TODO 2020-10-12 17:02 marcel: implement correctly
        return sycl::cpu_selector{}.operator()(device);
    #elif SYCL_LSH_TARGET == SYCL_LSH_TARGET_NVIDIA || SYCL_LSH_TARGET == SYCL_LSH_TARGET_AMD
        if (device.is_gpu()) {
            return sycl::default_selector{}.operator()(device);
        } else {
            return -1;
        }
    #elif SYCL_LSH_TARGET == SYCL_LSH_TARGET_INTEL
        return sycl::gpu_selector{}.operator()(device);
//        throw sycl_lsh::not_implemented("Can't currently select AMD or INTEL devices!");
    #else
        // never choose current device otherwise
        return -1;
    #endif
}