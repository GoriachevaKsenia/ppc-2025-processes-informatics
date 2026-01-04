#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

struct TaskInput {
  double x_min = 0.0;
  double x_max = 0.0;
  double y_min = 0.0;
  double y_max = 0.0;
  std::function<double(double, double)> f;
  int64_t num_points = 0;

  TaskInput() = default;

  TaskInput(double x_min_val, double x_max_val, double y_min_val, double y_max_val,
            std::function<double(double, double)> func, int64_t n_points)
      : x_min(x_min_val),
        x_max(x_max_val),
        y_min(y_min_val),
        y_max(y_max_val),
        f(std::move(func)),
        num_points(n_points) {}
};

using InType = TaskInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace liulin_y_integ_mnog_func_monte_carlo
