/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 */

#include <sycl_lsh/exceptions/not_implemented_exception.hpp>

#include <stdexcept>


sycl_lsh::not_implemented::not_implemented() : std::logic_error("Function not yet implemented!") { }