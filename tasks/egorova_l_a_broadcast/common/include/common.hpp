#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace egorova_l_a_broadcast {

struct InType {
  int root = 0;
  std::vector<int> data_int;
  std::vector<float> data_float;
  std::vector<double> data_double;
  int type_indicator = 0;  // 0: int, 1: float, 2: double
};

using OutType = std::vector<uint8_t>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egorova_l_a_broadcast
