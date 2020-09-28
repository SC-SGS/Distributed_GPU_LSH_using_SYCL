/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-28
 */

#include <sycl_lsh/detail/filesystem.hpp>
#include <sycl_lsh/mpi/file.hpp>


// ---------------------------------------------------------------------------------------------------------- //
//                                        constructors and destructor                                         //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::mpi::file::file(const std::string_view file_name, const sycl_lsh::mpi::communicator& comm, const mode m) {
    // check if the file exists
    if (m == mode::read && !fs::exists(file_name.data())) {
        throw std::invalid_argument(fmt::format("Illegal file '{}'!", file_name));
    }
    // open the file
    MPI_File_open(comm.get(), file_name.data(), static_cast<std::underlying_type_t<mode>>(m), MPI_INFO_NULL, &file_);
}

sycl_lsh::mpi::file::file(sycl_lsh::mpi::file&& other) noexcept : file_(std::move(other.file_)) {
    other.file_ = MPI_FILE_NULL;
}

sycl_lsh::mpi::file::~file() {
    MPI_File_close(&file_);
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            assignment operators                                            //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::mpi::file& sycl_lsh::mpi::file::operator=(sycl_lsh::mpi::file&& rhs) noexcept {
    if (file_ != MPI_FILE_NULL) {
        MPI_File_close(&file_);
    }
    file_ = std::move(rhs.file_);
    rhs.file_ = MPI_FILE_NULL;
    return *this;
}


// ---------------------------------------------------------------------------------------------------------- //
//                                            errhandler functions                                            //
// ---------------------------------------------------------------------------------------------------------- //
void sycl_lsh::mpi::file::attach_errhandler(const sycl_lsh::mpi::errhandler& handler) {
    if (handler.handler_type() != sycl_lsh::mpi::errhandler::type::file) {
        throw std::logic_error("Illegal errhandler type!");
    }
    MPI_File_set_errhandler(file_, handler.get());
}