#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kutergin_a_allreduce {

struct InData {
  std::vector<int> elements;
  int root;

  bool operator==(const InData &other) const {
    return elements == other.elements && root == other.root;
  }
};

using InType = InData;
using OutType = int;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<int, int, std::string>;

}  // namespace kutergin_a_allreduce
