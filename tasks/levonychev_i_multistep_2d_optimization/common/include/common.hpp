#pragma once

#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace levonychev_i_multistep_2d_optimization {

struct OptimizationParams {
  std::function<double(double, double)> func;

  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 1.0;

  int num_steps = 3;

  int grid_size_step1 = 10;

  int candidates_per_step = 4;

  bool use_local_optimization = true;
};

struct OptimizationResult {
  double x_min = 0.0;
  double y_min = 0.0;
  double value = 0.0;
};

using InType = OptimizationParams;
using OutType = OptimizationResult;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace levonychev_i_multistep_2d_optimization
