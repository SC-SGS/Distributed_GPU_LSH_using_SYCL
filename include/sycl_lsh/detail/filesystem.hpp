/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-28
 *
 * @brief Header to switch between `<filesystem>` and `<experimental/filesystem>`.
 * @details Only needed because linking against `<filesystem>` currently fails on Intel's
 *          [DevCloud](https://software.intel.com/content/www/us/en/develop/tools/devcloud.html).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILESYSTEM_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILESYSTEM_HPP

#if defined(SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM)

// use the experimental header
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#else

// use the "normal" C++17 header
#include <filesystem>
namespace fs = std::filesystem;

#endif

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILESYSTEM_HPP
