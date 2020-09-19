/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-19
 *
 * @brief Minimalistic wrapper class around a MPI communicator.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_COMMUNICATOR_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_COMMUNICATOR_HPP

#include <sycl_lsh/mpi/errhandler.hpp>

#include <mpi.h>

namespace sycl_lsh {

    /**
     * @brief Minimalistic wrapper around a MPI communicator.
     */
    class communicator {
    public:
        // ---------------------------------------------------------------------------------------------------------- //
        //                                        constructors and destructor                                         //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::communicator as copy of *MPI_COMM_WORLD*.
         */
        communicator();
        /**
         * @brief Construct a new @ref sycl_lsh::communicator as a copy of @p other.
         * @param[in] other the @ref sycl_lsh::communicator to copy
         */
        communicator(const communicator& other);
        /**
         * @brief Construct a new @ref sycl_lsh::communicator from the resources hold by @p other.
         * @param[in,out] other the @ref sycl_lsh::communicator to move-from
         */
        communicator(communicator&& other) noexcept;
        /**
         * @brief Construct a new @ref sycl_lsh::communicator from the given MPI_Comm.
         * @param[in] comm the MPI_Comm to wrap
         * @param[in] is_freeable `true` if @p comm should be freed at the end of `*this` lifetime, `false` otherwise
         */
        communicator(MPI_Comm comm, const bool is_freeable) noexcept;
        /**
         * @brief Destruct the @ref sycl_lsh::communicator object.
         * @details Only calls *MPI_Comm_free* if @ref sycl_lsh::communicator::freeable() returns `true`.
         */
        ~communicator();


        // ---------------------------------------------------------------------------------------------------------- //
        //                                            assignment operators                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Copy-assigns @p rhs to `*this`.
         * @param[in] rhs the @ref sycl_lsh::communicator to copy
         * @return `*this`
         */
        communicator& operator=(const communicator& rhs);
        /**
         * @brief Move-assigns @p rhs to `*this`.
         * @param[in] rhs the @ref sycl_lsh::communicator to move-from
         * @return `*this`
         */
        communicator& operator=(communicator&& rhs);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                         MPI communicator functions                                         //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Returns the current MPI rank.
         * @return the rank
         */
        int rank() const;
        /**
         * @brief Returns the size of the MPI communicator.
         * @return the communicator size
         */
        int size() const;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                            errhandler functions                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Attaches the given @ref sycl_lsh::errhandler @p handler to `*this` @ref sycl_lsh::communicator.
         * @param[in] handler the @ref sycl_lsh::errhandler to attach
         *
         * @throws std::logic_error if @p handler isn't of @ref sycl_lsh::errhandler::handler_type() `comm`
         */
        void attach_errhandler(const errhandler& handler);

        
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Get the underlying MPI communicator.
         * @return the MPI communicator wrapped in this @ref sycl_lsh::communicator object (`[nodiscard]]`)
         */
        [[nodiscard]]
        const MPI_Comm& get() const noexcept { return comm_; }
        /**
         * @brief Get the underlying MPI communicator.
         * @return the MPI communicator wrapped in this @ref sycl_lsh::communicator object (`[nodiscard]]`)
         */
        [[nodiscard]]
        MPI_Comm& get() noexcept { return comm_; }
        /**
         * @brief Returns whether the underlying MPI communicator gets automatically freed upon destruction.
         * @return `true` if *MPI_Comm_free* gets called upon destruction, `false` otherwise (`[nodiscard]]`)
         */
        [[nodiscard]]
        bool freeable() const noexcept { return is_freeable_; }

    private:
        MPI_Comm comm_;
        bool is_freeable_;
    };
    
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_COMMUNICATOR_HPP
