#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_BUFFER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_BUFFER_HPP

#include <vector>

#include <mpi.h>

#include <detail/mpi_type.hpp>


template <typename T>
class mpi_buffers {
public:
    using value_type = T;

    mpi_buffers(const MPI_Comm& communicator, const std::size_t size, const std::size_t dims)
            : communicator_(communicator), active_buffer_(0), buffer_0_(size * dims), buffer_1_(size * dims)
    {
        int comm_size, comm_rank;
        MPI_Comm_size(communicator_, &comm_size);
        MPI_Comm_rank(communicator_, &comm_rank);

        dest_ = (comm_rank + 1) % comm_size;
        source_ = (comm_size + (comm_rank - 1) % comm_size) % comm_size;
    }

    std::vector<value_type>& active() noexcept { return active_buffer_ == 0 ? buffer_0_ : buffer_1_; }
    std::vector<value_type>& inactive() noexcept { return active_buffer_ == 1 ? buffer_0_ : buffer_1_; }

    void send_receive() {
        MPI_Sendrecv(this->active().data(), this->active().size(), detail::mpi_type_cast<value_type>(), dest_, 0,
                     this->inactive().data(), this->inactive().size(), detail::mpi_type_cast<value_type>(), source_, 0,
                     communicator_, MPI_STATUS_IGNORE);
        active_buffer_ = (active_buffer_ + 1) % 2;
    }

private:
    const MPI_Comm& communicator_;
    int active_buffer_;
    int dest_;
    int source_;
    std::vector<value_type> buffer_0_;
    std::vector<value_type> buffer_1_;
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_BUFFER_HPP
