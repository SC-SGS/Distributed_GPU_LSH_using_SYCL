/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-18
 *
 * @brief File parser for parsing plain data files.
 */


#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP


#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <config.hpp>
#include <detail/convert.hpp>
#include <file_parser/base_file_parser.hpp>
#include <detail/mpi_type.hpp>
#include <mpi_buffer.hpp>


/**
 * brief File parser for parsing `.arff` files.
 * @details Expected file format:
 * @code
 * @RELATION "foobar.arff.gz"
 *
 * @ATTRIBUTE x0 NUMERIC
 * @ATTRIBUTE class NUMERIC
 *
 * @DATA
 * 0.2,0.4
 * 1.3,1.6
 * 2.4,2.8
 * 3.6,4.2
 * 3.8,4.6
 * @endcode
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 *
 * @note The file **must** be saved in **binary** form.
 */
template <memory_layout layout, typename Options>
class arff_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    /// The type of the base @ref file_parser.
    using base = file_parser<layout, Options>;
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref default_parser object for parsing plain data files.
     * @param[in] file the file to parse
     * @param[in] communicator the *MPI_Comm* communicator used to open the @p file with
     * @param[in] comm_rank the current MPI rank
     *
     * @throw std::invalid_argument if @p file doesn't exist
     * @throw std::logic_error **always** since reading text files isn't supported (yet)
     */
    arff_parser(const std::string& file, const MPI_Comm& communicator, const int comm_rank)
        : file_parser<layout, Options>(file, communicator, comm_rank)
    {
//        detail::mpi_print<print_rank>(communicator, "Parsing an '.arff' file using the default_parser together with MPI IO!\n");
        throw std::logic_error("Parsing an '.arff' file is currently no supported! Maybe run data_sets/convert_arff_to_binary.py first?");
    }

    [[nodiscard]] index_type parse_size() const override { throw std::logic_error("Not implemented yet!"); }
    [[nodiscard]] index_type parse_rank_size() const override { throw std::logic_error("Not implemented yet!"); }
    [[nodiscard]] index_type parse_dims() const override { throw std::logic_error("Not implemented yet!"); }
    [[nodiscard]] mpi_buffers<real_type, index_type> parse_content() const override { throw std::logic_error("Not implemented yet!"); }

};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
