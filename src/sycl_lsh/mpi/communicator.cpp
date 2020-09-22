/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-22
 */

#include <sycl_lsh/mpi/communicator.hpp>

#include <stdexcept>


// ---------------------------------------------------------------------------------------------------------- //
//                                        constructors and destructor                                         //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::mpi::communicator::communicator() : is_freeable_(true) {
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
}

sycl_lsh::mpi::communicator::communicator(const sycl_lsh::mpi::communicator& other) {
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

sycl_lsh::mpi::communicator::communicator(sycl_lsh::mpi::communicator&& other) noexcept
    : comm_(std::move(other.comm_)), is_freeable_(std::move(other.is_freeable_))
{
    // set other to the moved-from state
    other.comm_ = MPI_COMM_NULL;
    other.is_freeable_ = false;
}

sycl_lsh::mpi::communicator::communicator(MPI_Comm comm, const bool is_freeable) noexcept : comm_(comm), is_freeable_(is_freeable) { }

sycl_lsh::mpi::communicator::~communicator() {
    // destroy communicator if marked as freeable
    if (is_freeable_) {
        MPI_Comm_free(&comm_);
    }
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            assignment operators                                            //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::mpi::communicator& sycl_lsh::mpi::communicator::operator=(const sycl_lsh::mpi::communicator& rhs) {
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

sycl_lsh::mpi::communicator& sycl_lsh::mpi::communicator::operator=(sycl_lsh::mpi::communicator&& rhs) noexcept {
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
//                                         MPI communicator functions                                         //
// ---------------------------------------------------------------------------------------------------------- //
int sycl_lsh::mpi::communicator::rank() const {
    int comm_rank;
    MPI_Comm_rank(comm_, &comm_rank);
    return comm_rank;
}
int sycl_lsh::mpi::communicator::size() const {
    int comm_size;
    MPI_Comm_size(comm_, &comm_size);
    return comm_size;
}
bool sycl_lsh::mpi::communicator::master_rank() const {
    return this->rank() == 0;
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            errhandler functions                                            //
// ---------------------------------------------------------------------------------------------------------- //
void sycl_lsh::mpi::communicator::attach_errhandler(const sycl_lsh::mpi::errhandler& handler) {
    if (handler.handler_type() != sycl_lsh::mpi::errhandler::type::comm) {
        throw std::logic_error("Illegal errhandler type!");
    }
    MPI_Comm_set_errhandler(comm_, handler.get());
}