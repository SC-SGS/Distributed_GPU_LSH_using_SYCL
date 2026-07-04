/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/detail/utility.hpp"

#include <algorithm>    // std::transform
#include <cctype>       // std::tolower
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace sycl_lsh::detail {

std::string to_lower_case(const std::string_view str) {
    std::string lowercase_str{ str };
    std::transform(str.begin(), str.end(), lowercase_str.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(static_cast<int>(c))); });
    return lowercase_str;
}

bool contains_substr(const std::string_view str, const std::string_view substr) noexcept {
    return str.find(substr) != std::string_view::npos;
}

}  // namespace sycl_lsh::detail
