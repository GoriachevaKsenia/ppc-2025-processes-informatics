#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ashihmin_d_calculate_integrals_by_simpson {

struct SimpsonInput {
  std::vector<double> left_bounds;
  std::vector<double> right_bounds;
  int partitions{};
};

using InType = SimpsonInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ashihmin_d_calculate_integrals_by_simpson
