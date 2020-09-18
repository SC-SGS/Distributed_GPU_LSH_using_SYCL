/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 */

#include <sycl_lsh/mpi/communicator.hpp>

#include <stdexcept>

// ---------------------------------------------------------------------------------------------------------- //
//                                        constructors and destructor                                         //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::communicator::communicator() : is_freeable_(true) {
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
}

sycl_lsh::communicator::communicator(const sycl_lsh::communicator& other) {
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

sycl_lsh::communicator::communicator(sycl_lsh::communicator&& other) noexcept
    : comm_(std::move(other.comm_)), is_freeable_(std::move(other.is_freeable_))
{
    // set other to the moved-from state
    other.comm_ = MPI_COMM_NULL;
    other.is_freeable_ = false;
}

sycl_lsh::communicator::communicator(MPI_Comm comm, const bool is_freeable) noexcept : comm_(comm), is_freeable_(is_freeable) { }

sycl_lsh::communicator::~communicator() {
    // destroy communicator if marked as freeable
    if (is_freeable_) {
        MPI_Comm_free(&comm_);
    }
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            assignment operators                                            //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::communicator& sycl_lsh::communicator::operator=(const sycl_lsh::communicator& rhs) {
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

sycl_lsh::communicator& sycl_lsh::communicator::operator=(sycl_lsh::communicator&& rhs) {
    // delete current communicator if and only if it is marked as freeable
    if (is_freeable_) {
        MPI_Comm_free(&comm_);
    }
    // transfer ownership
    comm_ = std::move(rhs.comm_);
    is_freeable_ = std::move(rhs.is_freeable_);
    // set rhs to the moved-from state
    rhs.comm_ = MPI_COMM_NULL;
    rhs.is_freeable_ = false;
    return *this;
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            errhandler functions                                            //
// ---------------------------------------------------------------------------------------------------------- //
void sycl_lsh::communicator::attach_errhandler(const sycl_lsh::errhandler& handler) {
    if (handler.handler_type() != sycl_lsh::errhandler::type::comm) {
        throw std::logic_error("Illegal errhandler type!");
    }
    MPI_Comm_set_errhandler(comm_, handler.get());
}