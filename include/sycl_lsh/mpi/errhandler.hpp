/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 *
 * @brief Minimalistic wrapper class around a MPI errhandler.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ERRHANDLER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ERRHANDLER_HPP

#include <mpi.h>

namespace sycl_lsh {

    /**
    * @brief Minimalistic wrapper around a MPI errhandler.
    */
    class errhandler {
    public:
        /**
         * @brief Enum class for the different @ref sycl_lsh::errhandler types.
         */
        enum class type {
            /** errhandler for MPI communicators */
            comm,
            /** errhandler for MPI files */
            file,
            /** errhandler for MPI windows */
            win
        };

        // ---------------------------------------------------------------------------------------------------------- //
        //                                        constructors and destructor                                         //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new communicator errhandler with the handler function @p func.
         * @param[in] func the handler function to call
         */
        errhandler(MPI_Comm_errhandler_function func);
        /**
         * @brief Construct a new file errhandler with the handler function @p func.
         * @param[in] func the handler function to call
         */
        errhandler(MPI_File_errhandler_function func);
        /**
         * @brief Construct a new window errhandler with the handler function @p func.
         * @param[in] func the handler function to call
         */
        errhandler(MPI_Win_errhandler_function func);
        /**
         * @brief Construct a new errhandler of type @p t with a default exception error handler function.
         * @details TODO: exceptions
         * @param[in] t the type of the errhandler
         */
        errhandler(const type t);
        // delete copy constructor
        errhandler(const errhandler&) = delete;
        /**
         * @brief Construct a new @ref sycl_lsh::errhandler from the resources hold by @p other.
         * @param[in,out] other the @ref sycl_lsh::errhandler to move-from
         */
        errhandler(errhandler&& other) noexcept;
        /**
         * @brief Construct a new @ref sycl_lsh::errhandler from the given MPI_Errhandler.
         * @param[in] errhandler the MPI_Errhandler to wrap
         * @param[in] t the type of MPI_Errhandler
         * @param[in] is_freeable `true` if @p errhandler should be freed at the end of `*this` lifetime, `false` otherwise
         */
        errhandler(MPI_Errhandler errhandler, const type t, const bool is_freeable) noexcept;
        /**
         * @brief Destruct the @ref sycl_lsh::errhandler object.
         * @details Only calls *MPI_Errhandler_free* if @ref sycl_lsh::errhandler::freeable() returns `true`.
         */
        ~errhandler();

        // ---------------------------------------------------------------------------------------------------------- //
        //                                            assignment operators                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        // delete copy assignment operator
        errhandler& operator=(const errhandler&) = delete;
        /**
         * @brief Move-assigns @p rhs to `*this`.
         * @param[in] rhs the @ref sycl_lsh::errhandler to move-from
         * @return `*this`
         */
        errhandler& operator=(errhandler&& rhs);

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Get the underlying MPI errhandler.
         * @return the MPI errhandler wrapped in this @ref sycl_lsh::errhandler object (`[nodiscard]]`)
         */
        [[nodiscard]]
        const MPI_Errhandler& get() const noexcept { return errhandler_; }
        /**
         * @brief Get the underlying MPI errhandler.
         * @return the MPI errhandler wrapped in this @ref sycl_lsh::errhandler object (`[nodiscard]]`)
         */
        [[nodiscard]]
        MPI_Errhandler& get() noexcept { return errhandler_; }
        /**
         * @brief Get the type of the @ref sycl_lsh::errhandler.
         * @return the type of this @ref sycl_lsh::errhandler object (`[nodiscard]]`)
         */
        [[nodiscard]]
        type handler_type() const noexcept { return type_; }
        /**
         * @brief Returns whether the underlying MPI errhandler gets automatically freed upon destruction.
         * @return `true` if *MPI_Errhandler_free* gets called upon destruction, `false` otherwise (`[nodiscard]]`)
         */
        [[nodiscard]]
        bool freeable() const noexcept { return is_freeable_; }

    private:
        MPI_Errhandler errhandler_;
        const type type_;
        bool is_freeable_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ERRHANDLER_HPP
