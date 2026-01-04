#pragma once
#include <functional>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace krykov_e_multistep_sad {

using Function2D = std::function<double(double, double)>;

struct Region {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double value;
};

using InType = std::tuple<Function2D,  // f(x, y)
                          double,      // x_min
                          double,      // x_max
                          double,      // y_min
                          double       // y_max
                          >;

using OutType = std::vector<double>;  // x*, y*, f(x*, y*)
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace krykov_e_multistep_sad
