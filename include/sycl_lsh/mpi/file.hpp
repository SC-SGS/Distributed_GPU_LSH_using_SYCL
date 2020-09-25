/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-25
 *
 * @brief Minimalistic wrapper class around a MPI file.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_HPP

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/errhandler.hpp>

#include <fmt/format.h>
#include <mpi.h>

#include <filesystem>
#include <string_view>

namespace sycl_lsh::mpi {

    /**
     * @brief Minimalistic wrapper around a MPI file.
     */
    class file {
    public:
        /**
         * @brief Enum class for the different file open types (read or write).
         */
        enum class mode {
            /** open file in read only mode */
            read = MPI_MODE_RDONLY,
            /** open file in write only mode*/
            write = MPI_MODE_WRONLY | MPI_MODE_APPEND | MPI_MODE_CREATE
        };

        
        // ---------------------------------------------------------------------------------------------------------- //
        //                                        constructors and destructor                                         //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::mpi::file, i.e. open the file @p file_name ind the mode @p m.
         * @param[in] file_name the file to open
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] m the open mode (read or write)
         *
         * @throws std::invalid_argument if the file @p file_name doesn't exist and the open mode is `read`
         */
        file(std::string_view file_name, const communicator& comm, mode m);
        // delete copy constructor
        file(const file& other) = delete;
        /**
         * @brief Construct a new @ref sycl_lsh::mpi::file from the resources hold by @p other.
         * @param[in,out] other the @ref sycl_lsh::mpi::file to move-from
         */
        file(file&& other) noexcept;
        /**
         * @brief Destruct the @ref sycl_lsh::mpi::file object, i.e. closes the previously opened file.
         */
        ~file();


        // ---------------------------------------------------------------------------------------------------------- //
        //                                            assignment operators                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        // delete copy assignment operator
        file& operator=(const file& rhs) = delete;
        /**
         * @brief Move-assigns @p rhs to `*this`.
         * @param[in] rhs the @ref sycl_lsh::mpi::file to move-from
         * @return `*this`
         */
        file& operator=(file&& rhs) noexcept;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                            errhandler functions                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Attaches the given @ref sycl_lsh::mpi::file @p handler to `*this` @ref sycl_lsh::mpi::file.
         * @param[in] handler the @ref sycl_lsh::mpi::errhandler to attach
         *
         * @throws std::logic_error if @p handler isn't of @ref sycl_lsh::mpi::errhandler::handler_type() `file`.
         */
        void attach_errhandler(const errhandler& handler);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Get the underlying MPI file.
         * @return the MPI file wrapped in this @ref sycl_lsh::mpi::file object (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const MPI_File& get() const noexcept { return file_; }
        /**
         * @brief Get the underlying MPI file.
         * @return the MPI file wrapped in this @ref sycl_lsh::mpi::file object (`[[nodiscard]]`)
         */
        [[nodiscard]]
        MPI_File& get() noexcept { return file_; }
        
    private:
        MPI_File file_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_HPP
