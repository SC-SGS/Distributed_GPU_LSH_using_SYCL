/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 */

#include <sycl_lsh/exceptions/communicator_exception.hpp>
#include <sycl_lsh/exceptions/file_exception.hpp>
#include <sycl_lsh/mpi/errhandler.hpp>

#include <stdexcept>


// ---------------------------------------------------------------------------------------------------------- //
//                                        default errhandler functions                                        //
// ---------------------------------------------------------------------------------------------------------- //
namespace {
    // TODO 2020-09-18 18:02 marcel: correct exceptions
    /*
     * Default @ref sycl_lsh::errhandler function for MPI communicators.
     */
    void comm_exception_errhandler(MPI_Comm* comm, int* err, ...) {
        throw sycl_lsh::communicator_exception(*comm, *err);
    }
    /*
     * Default @ref sycl_lsh::errhandler function for MPI files.
     */
    void file_exception_errhandler(MPI_File* file, int* err, ...) {
        throw sycl_lsh::file_exception(*file, *err);
    }
    /*
     * Default @ref sycl_lsh::errhandler function for MPI windows.
     */
    void win_exception_errhandler(MPI_Win* win, int* err, ...) {
        throw std::logic_error("WIN ERROR!");
    }
}


// ---------------------------------------------------------------------------------------------------------- //
//                                        constructors and destructor                                         //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::errhandler::errhandler(MPI_Comm_errhandler_function func) : type_(type::comm), is_freeable_(true) {
    MPI_Comm_create_errhandler(func, &errhandler_);
}
sycl_lsh::errhandler::errhandler(MPI_File_errhandler_function func) : type_(type::file), is_freeable_(true) {
    MPI_File_create_errhandler(func, &errhandler_);
}
sycl_lsh::errhandler::errhandler(MPI_Win_errhandler_function func) : type_(type::win), is_freeable_(true) {
    MPI_Win_create_errhandler(func, &errhandler_);
}

sycl_lsh::errhandler::errhandler(const type t) : type_(t), is_freeable_(true) {
    switch (type_) {
        case type::comm:
            MPI_Comm_create_errhandler(comm_exception_errhandler, &errhandler_);
            break;
        case type::file:
            MPI_File_create_errhandler(file_exception_errhandler, &errhandler_);
            break;
        case type::win:
            MPI_Win_create_errhandler(win_exception_errhandler, &errhandler_);
            break;
    }
}

sycl_lsh::errhandler::errhandler(sycl_lsh::errhandler&& other) noexcept
    : errhandler_(std::move(other.errhandler_)), type_(std::move(other.type_)), is_freeable_(std::move(other.is_freeable_))
{
    // set other to the moved-from state
    other.errhandler_ = MPI_ERRHANDLER_NULL;
    other.is_freeable_ = false;
}

sycl_lsh::errhandler::errhandler(MPI_Errhandler errhandler, const type t, const bool is_freeable) noexcept
    : errhandler_(errhandler), type_(t), is_freeable_(is_freeable) { }

sycl_lsh::errhandler::~errhandler() {
    // destroy errhandler if marked as freeable
    if (is_freeable_) {
        MPI_Errhandler_free(&errhandler_);
    }
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            assignment operators                                            //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::errhandler& sycl_lsh::errhandler::operator=(sycl_lsh::errhandler&& rhs) {
    // delete current errhandler if and only if it is marked as freeable
    if (is_freeable_) {
        MPI_Errhandler_free(&errhandler_);
    }
    // transfer ownership
    errhandler_ = std::move(rhs.errhandler_);
    is_freeable_ = std::move(rhs.is_freeable_);
    // set rhs to the moved-from state
    rhs.errhandler_ = MPI_ERRHANDLER_NULL;
    rhs.is_freeable_ = false;
    return *this;
}