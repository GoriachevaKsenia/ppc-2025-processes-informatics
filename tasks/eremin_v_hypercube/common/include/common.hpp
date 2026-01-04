#pragma once

#include <functional>
#include <tuple>

#include "task/include/task.hpp"

namespace eremin_v_hypercube {

using InType = std::tuple<double, double, int, std::function<double(double)>>;
using OutType = double;
using TestType = std::tuple<int, double, double, int, std::function<double(double)>, std::function<double(double)>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace eremin_v_hypercube
