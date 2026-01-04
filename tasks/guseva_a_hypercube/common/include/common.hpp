#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace guseva_a_hypercube {
using InType = std::tuple<int, int, double (*)(int), std::string>;
using OutType = double;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace guseva_a_hypercube
