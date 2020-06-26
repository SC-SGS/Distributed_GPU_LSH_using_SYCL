/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-26
 *
 * @brief Base class for all file parsers.
 * @details Pure virtual.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_PARSER_HPP

#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <mpi.h>

#include <config.hpp>
#include <options.hpp>


/**
 * @brief Pure virtual base class for all file parser.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <typename Options>
class file_parser {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by an 'options' type!");
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref file_parser object and opens the file to retrieve a *MPI_File* handle.
     * @param[in] file_name the file to parse
     * @param[in] communicator the *MPI_Comm* communicator used to open the @p file with
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
    file_parser(const std::string& file_name, const MPI_Comm& communicator) {
        // check if the file exists
        if (!std::filesystem::exists(file_name)) {
            throw std::invalid_argument("File '" + file_name + "' doesn't exist!");
        }
        // open the file
        MPI_File_open(communicator, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_);
        // save comm_size and comm_rank for later parsing
        MPI_Comm_size(communicator, &comm_size_);
        MPI_Comm_rank(communicator, &comm_rank_);
    }
    /**
     * @brief Close the *MPI_File* handle upon destruction.
     */
    virtual ~file_parser() {
        MPI_File_close(&file_);
    }

    /**
     * @brief Pareses the **total** number of data points in the file.
     * @return the total number of data points (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_total_size() const = 0;
    /**
     * @brief Parses the number of data points **per MPI rank** of the file.
     * @details Example: \n
     *          `this->parse_size() = 14, comm_size = 4` \n
     *          -> `this->parse_rank_size() = 4` for **ALL** MPI ranks and **NOT** `{ 4, 4, 3, 3 }`.
     * @return the number of data points per MPI rank (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_rank_size() const = 0;
    /**
     * @brief Parses the number of dimensions of each data point in the file.
     * @return the number of dimensions (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_dims() const = 0;
    /**
     * @brief Parse the content of the file.
     * @param[out] buffer to write the data to
     */
    virtual void parse_content(real_type* buffer) const = 0;

protected:
    /// The size of the used MPI communicator.
    int comm_size_ = 0;
    /// The current MPI rank.
    int comm_rank_ = 0;
    /// The opened *MPI_File* handle.
    MPI_File file_ = MPI_FILE_NULL;
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_PARSER_HPP
