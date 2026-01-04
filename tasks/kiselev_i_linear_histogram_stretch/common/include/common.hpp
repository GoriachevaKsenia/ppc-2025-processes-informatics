#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kiselev_i_linear_histogram_stretch {

struct GrayImage {
  std::vector<uint8_t> pixels;
  std::size_t width = 0;
  std::size_t height = 0;
};

using InType = GrayImage;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kiselev_i_linear_histogram_stretch
