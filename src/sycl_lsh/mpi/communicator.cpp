/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include "mpi/mpi.h"  // MPI_Comm related functionality

#include <memory>   // std::addressof
#include <utility>  // std::exchange

namespace sycl_lsh::mpi {
// ---------------------------------------------------------------------------------------------------------- //
//                                        constructors and destructor                                         //
// ---------------------------------------------------------------------------------------------------------- //
communicator::communicator() : is_freeable_{ true } {
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
}

communicator::communicator(const communicator &other) {
    if (other.comm_ == MPI_COMM_NULL) {
        // copy a communicator which refers to MPI_COMM_NULL
        comm_ = MPI_COMM_NULL;
        is_freeable_ = other.is_freeable_;
    } else {
        // copy normal communicator
        MPI_Comm_dup(other.comm_, &comm_);
        is_freeable_ = true;
    }
}

communicator::communicator(communicator &&other) noexcept
    : comm_{ std::exchange(other.comm_, MPI_COMM_NULL) },
      is_freeable_{ std::exchange(other.is_freeable_, false) } {}

communicator::communicator(MPI_Comm comm, const bool is_freeable) noexcept : comm_{ comm }, is_freeable_{ is_freeable } {}

communicator::~communicator() {
    // destroy communicator if marked as freeable
    if (is_freeable_ && comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&comm_);
    }
}

// ---------------------------------------------------------------------------------------------------------- //
//                                            assignment operators                                            //
// ---------------------------------------------------------------------------------------------------------- //
communicator &communicator::operator=(const communicator &rhs) {
    // check against self-assignment
    if (this != std::addressof(rhs)) {
        // delete current communicator if and only if it is marked as freeable
        if (is_freeable_) {
            MPI_Comm_free(&comm_);
        }

        // copy rhs communicator
        if (rhs.comm_ == MPI_COMM_NULL) {
            // copy a communicator which refers to MPI_COMM_NULL
            comm_ = MPI_COMM_NULL;
            is_freeable_ = rhs.is_freeable_;
        } else {
            // copy normal communicator
            MPI_Comm_dup(rhs.comm_, &comm_);
            is_freeable_ = true;
        }
    }
    return *this;
}

communicator &communicator::operator=(communicator &&rhs) noexcept {
    // check against self-assignment
    if (this != std::addressof(rhs)) {
        // delete current communicator if and only if it is marked as freeable
        if (is_freeable_) {
            MPI_Comm_free(&comm_);
        }
        // transfer ownership
        comm_ = std::exchange(rhs.comm_, MPI_COMM_NULL);
        is_freeable_ = std::exchange(rhs.is_freeable_, false);
    }
    return *this;
}

// ---------------------------------------------------------------------------------------------------------- //
//                                         MPI communicator functions                                         //
// ---------------------------------------------------------------------------------------------------------- //
int communicator::rank() const {
    int comm_rank;
    MPI_Comm_rank(comm_, &comm_rank);
    return comm_rank;
}
int communicator::size() const {
    int comm_size;
    MPI_Comm_size(comm_, &comm_size);
    return comm_size;
}
bool communicator::is_main_rank() const {
    return this->rank() == main_rank();
}
void communicator::barrier() const {
    MPI_Barrier(comm_);
}

}  // namespace sycl_lsh::mpi
