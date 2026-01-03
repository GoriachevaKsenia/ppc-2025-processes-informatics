#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kutergin_v_reduce {

struct InputData {
  std::vector<int> data;
  int root = 0;  // root по умолчанию
};

using InType = InputData;
using OutType = int;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<int, int, std::string>;

}  // namespace kutergin_v_reduce
