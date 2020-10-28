/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 */

#include <sycl_lsh/detail/utility.hpp>

#include <string_view>

[[nodiscard]]
bool sycl_lsh::detail::contains_substr(const std::string_view str, const std::string_view substr) noexcept {
    return str.find(substr) != std::string_view::npos;
}