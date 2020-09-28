/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-28
 *
 * @brief Minimalistic wrapper class around a MPI errhandler.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ERRHANDLER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ERRHANDLER_HPP

#include <mpi.h>

namespace sycl_lsh::mpi {

    /**
    * @brief Minimalistic wrapper around a MPI errhandler.
    */
    class errhandler {
    public:
        /**
         * @brief Enum class for the different @ref sycl_lsh::mpi::errhandler types.
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
         * @brief Construct a new errhandler of type @p t with the error handler function @p func.
         * @tparam Func type of the MPI_Errhandler function
         * @param[in] func the MPI_Errhandler function
         * @param[in] t the type of the errhandler
         */
        template <typename Func>
        errhandler(Func func, type t);
        /**
         * @brief Construct a new errhandler of type @p t with a default exception error handler function.
         * @param[in] t the type of the errhandler
         */
        explicit errhandler(type t);
        // delete copy constructor
        errhandler(const errhandler&) = delete;
        /**
         * @brief Construct a new @ref sycl_lsh::mpi::errhandler from the resources hold by @p other.
         * @param[in,out] other the @ref sycl_lsh::mpi::errhandler to move-from
         */
        errhandler(errhandler&& other) noexcept;
        /**
         * @brief Construct a new @ref sycl_lsh::mpi::errhandler from the given MPI_Errhandler.
         * @param[in] errhandler the MPI_Errhandler to wrap
         * @param[in] t the type of MPI_Errhandler
         * @param[in] is_freeable `true` if @p errhandler should be freed at the end of `*this` lifetime, `false` otherwise
         */
        errhandler(MPI_Errhandler errhandler, type t, bool is_freeable) noexcept;
        /**
         * @brief Destruct the @ref sycl_lsh::mpi::errhandler object.
         * @details Only calls *MPI_Errhandler_free* if @ref sycl_lsh::mpi::errhandler::freeable() returns `true`.
         */
        ~errhandler();

        // ---------------------------------------------------------------------------------------------------------- //
        //                                            assignment operators                                            //
        // ---------------------------------------------------------------------------------------------------------- //
        // delete copy assignment operator
        errhandler& operator=(const errhandler&) = delete;
        /**
         * @brief Move-assigns @p rhs to `*this`.
         * @param[in] rhs the @ref sycl_lsh::mpi::errhandler to move-from
         * @return `*this`
         */
        errhandler& operator=(errhandler&& rhs);

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Get the underlying MPI errhandler.
         * @return the MPI errhandler wrapped in this @ref sycl_lsh::mpi::errhandler object (`[nodiscard]]`)
         */
        [[nodiscard]]
        const MPI_Errhandler& get() const noexcept { return errhandler_; }
        /**
         * @brief Get the underlying MPI errhandler.
         * @return the MPI errhandler wrapped in this @ref sycl_lsh::mpi::errhandler object (`[nodiscard]]`)
         */
        [[nodiscard]]
        MPI_Errhandler& get() noexcept { return errhandler_; }
        /**
         * @brief Get the type of the @ref sycl_lsh::mpi::errhandler.
         * @return the type of this @ref sycl_lsh::mpi::errhandler object (`[nodiscard]]`)
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


    // ---------------------------------------------------------------------------------------------------------- //
    //                                        constructors and destructor                                         //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename Func>
    errhandler::errhandler(Func func, const type t) : type_(t), is_freeable_(true) {
        switch (type_) {
            case type::comm:
                MPI_Comm_create_errhandler(func, &errhandler_);
                break;
            case type::file:
                MPI_File_create_errhandler(func, &errhandler_);
                break;
            case type::win:
                MPI_Win_create_errhandler(func, &errhandler_);
                break;
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ERRHANDLER_HPP
