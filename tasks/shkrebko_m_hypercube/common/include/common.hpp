#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shkrebko_m_hypercube {

struct HypercubeData {
  int value{0};
  int destination{0};
  std::vector<int> path;
  bool finish{false};

  HypercubeData() = default;
};

using InType = std::vector<int>;
using OutType = HypercubeData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkrebko_m_hypercube
