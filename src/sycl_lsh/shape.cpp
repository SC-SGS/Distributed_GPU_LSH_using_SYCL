/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/shape.hpp"

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::swap
#include <cstddef>    // std::size_t
#include <iostream>   // std:ostream, std::istream

namespace sycl_lsh {

shape::shape(const std::size_t x_p, const std::size_t y_p) noexcept :
    x{ x_p },
    y{ y_p },
    z{ std::size_t{ 1 } } { }

shape::shape(const std::size_t x_p, const std::size_t y_p, const std::size_t z_p) noexcept :
    x{ x_p },
    y{ y_p },
    z{ z_p } { }

void shape::swap(shape &other) noexcept {
    using std::swap;
    swap(this->x, other.x);
    swap(this->y, other.y);
    swap(this->z, other.z);
}

std::ostream &operator<<(std::ostream &out, const shape &s) {
    return out << fmt::format("[{}, {}, {}]", s.x, s.y, s.z);
}

std::istream &operator>>(std::istream &in, shape &s) {
    return in >> s.x >> s.y >> s.z;
}

void swap(shape &lhs, shape &rhs) noexcept {
    lhs.swap(rhs);
}

bool operator==(const shape &lhs, const shape &rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool operator!=(const shape &lhs, const shape &rhs) noexcept {
    return !(lhs == rhs);
}

}  // namespace sycl_lsh::detail
