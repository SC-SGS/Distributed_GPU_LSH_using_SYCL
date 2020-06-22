/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-17
 *
 * @brief Implements buffers for the MPI communication.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_BUFFER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_BUFFER_HPP

#include <numeric>
#include <mpi.h>

#include <detail/mpi_type.hpp>


/**
 * @brief Buffers for the MPI communication. Used to hide the communication costs behind the calculation costs.
 * @tparam T buffer data type
 */
template <typename value_type, typename size_type = std::size_t>
class mpi_buffers {
public:
    /// The number of data points in total.
    const size_type total_size;
    /// The number of data points per MPI rank.
    const size_type size;
    /// The dimension of each data point.
    const size_type dims;


    /**
     * @brief Construct two new buffers, each of size `size * dims`.
     * @details Calculates the destination and source MPI ranks. Sets @ref buffer_0_ to the active buffer.
     * @param[in] communicator the communicator used in the *MPI_Sendrecv* call.
     * @param[in] size the number of data points in the buffers
     * @param[in] dims the number of dimensions per data point in the buffer
     */
    mpi_buffers(const MPI_Comm& communicator, const size_type size, const size_type dims)
            : total_size([&]() { int comm_size; MPI_Comm_size(communicator, &comm_size); return comm_size * size; }()),
              size(size), dims(dims), communicator_(communicator),
              active_buffer_(0), buffer_0_(new value_type[size * dims]), buffer_1_(new value_type[size * dims])
    {
        int comm_size, comm_rank;
        MPI_Comm_size(communicator_, &comm_size);
        MPI_Comm_rank(communicator_, &comm_rank);

        dest_ = (comm_rank + 1) % comm_size;
        source_ = (comm_size + (comm_rank - 1) % comm_size) % comm_size;
    }

    ~mpi_buffers() {
        delete[] buffer_0_;
        delete[] buffer_1_;
    }

    // make sure an object of this class will NEVER be copied
    mpi_buffers(const mpi_buffers&) = delete;
    mpi_buffers(mpi_buffers&& other) noexcept
        : total_size(other.total_size), size(other.size), dims(other.dims),
          communicator_(other.communicator_), active_buffer_(other.active_buffer_), dest_(other.dest_), source_(other.source_),
          buffer_0_(other.buffer_0_), buffer_1_(other.buffer_1_)
    {
        other.buffer_0_ = nullptr;
        other.buffer_1_ = nullptr;
    }
    mpi_buffers& operator=(const mpi_buffers&) = delete;
    mpi_buffers& operator=(mpi_buffers&&) = delete;

    /**
     * @brief Returns the currently active buffer, i.e. the buffer which holds the data currently worked on.
     * @return the active buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type* active() noexcept { return active_buffer_ == 0 ? buffer_0_ : buffer_1_; }
    /**
     * @brief Returns the currently inactive buffer, i.e. the buffer which holds the data that can be savely discarded.
     * @return the inactive buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type* inactive() noexcept { return active_buffer_ == 1 ? buffer_0_ : buffer_1_; }

    /**
     * @brief Send the currently active buffer to the neighboring inactive buffer using a ring like send pattern.
     * @details Swaps the currently active and inactive buffers.
     */
    void send_receive() {
        MPI_Sendrecv(this->active(), size * dims, detail::mpi_type_cast<value_type>(), dest_, 0,
                     this->inactive(), size * dims, detail::mpi_type_cast<value_type>(), source_, 0,
                     communicator_, MPI_STATUS_IGNORE);
        active_buffer_ = (active_buffer_ + 1) % 2;
    }

private:
    /// The MPI communicator used for the *MPI_Sendrecv* call.
    const MPI_Comm& communicator_;
    /// The currently active buffer.
    int active_buffer_;
    /// The destination MPI rank.
    int dest_;
    /// The source MPI rank.
    int source_;
    /// The first buffer.
    value_type* buffer_0_;
    /// The second buffer.
    value_type* buffer_1_;
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_BUFFER_HPP
