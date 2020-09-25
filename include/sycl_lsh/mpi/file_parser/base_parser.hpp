/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-25
 *
 * @brief Base class for all different file parsers.
 * @details Pure virtual.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_PARSER_HPP

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/file.hpp>
#include <sycl_lsh/detail/defines.hpp>

#include <string_view>
#include <type_traits>

namespace sycl_lsh::mpi {

    /**
     * @brief Pure virtual base class for all different file parsers.
     * @tparam Options type of the used @ref sycl_lsh::options class
     * @tparam T the type of the data to parse
     */
    template <typename Options, typename T>
    class file_parser {
        static_assert(std::is_base_of_v<sycl_lsh::detail::options_base, Options>, "The first template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_arithmetic_v<T>, "The second template parameter must be an arithmetic type!");
    public:
        /// The index type as specified in the provided @ref sycl_lsh::options template type.
        using index_type = typename Options::index_type;
        /// The type of the data which should get parsed.
        using parsing_type = T;

        /**
         * @brief Construct a new @ref sycl_lsh::mpi::file_parser object and opens the file @p file_name.
         * @param[in] file_name the file to parse
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator 
         */
        file_parser(const std::string_view file_name, const communicator& comm) : comm_(comm), file_(file_name, comm, file::mode::read) { }
        /**
         * @brief Virtual descructor to enable proper inheritance.
         */
        virtual ~file_parser() = default;

        /**
         * @brief Parse the **total** number of data points in the file.
         * @return the total number of data points (`[[nodiscard]]`)
         */
        [[nodiscard]]
        virtual index_type parse_total_size() const = 0;
        /**
         * @brief Parse the number of data points **per MPI rank** of the file.
         * @details If the total number of data points isn't dividable by the MPI communicator size,
         *          the **last** MPI rank will be filled with dummy points. \n
         *          Example: \n
         *          data = xxxxxxxxxx (size = 10), communicator size = 3 \n
         *          rank 1: xxxx, rank 2: xxxx, rank 3: xxdd \n
         *          -> the rank size is 4 for each MPI rank!
         * @return the number of data points per MPI rank (`[[nodiscard]]`)
         */
        [[nodiscard]]
        virtual index_type parse_rank_size() const = 0;
        /**
         * @brief Parse the number of dimensions of each data point in the file.
         * @details Assumes that each data points has the same number of dimensions.
         * @return the number of dimensions (`[[nodiscard]]`)
         */
        [[nodiscard]]
        virtual index_type parse_dims() const = 0;
        /**
         * @brief Parse the content of the file.
         * @param[out] buffer to write the data to
         */
        virtual void parse_content(parsing_type* buffer) const = 0;

    private:
        const communicator& comm_;
        file file_;
    };

}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_PARSER_HPP
