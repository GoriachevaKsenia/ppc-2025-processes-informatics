#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace titaev_m_metod_pryamougolnikov {

struct RectangleInput {
  std::vector<double> left_bounds;
  std::vector<double> right_bounds;
  int partitions{};
};

using InType = RectangleInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace titaev_m_metod_pryamougolnikov
