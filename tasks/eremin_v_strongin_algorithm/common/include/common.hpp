#pragma once

#include <functional>
#include <tuple>

#include "task/include/task.hpp"

namespace eremin_v_strongin_algorithm {

using InType = std::tuple<double, double, double, int, std::function<double(double)>>;
using OutType = double;
using TestType = std::tuple<int, double, double, double, int, std::function<double(double)>, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace eremin_v_strongin_algorithm
