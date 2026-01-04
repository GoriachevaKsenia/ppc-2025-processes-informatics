#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

struct TaskData {
  int n_steps;
  int func_id;
  double x1, x2;
  double y1, y2;

  bool operator==(const TaskData &other) const {
    return n_steps == other.n_steps && func_id == other.func_id && std::abs(x1 - other.x1) < 1e-9 &&
           std::abs(x2 - other.x2) < 1e-9 && std::abs(y1 - other.y1) < 1e-9 && std::abs(y2 - other.y2) < 1e-9;
  }
};

using InType = TaskData;
using OutType = double;
using TestType = std::tuple<TaskData, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::function<double(double, double)> GetFunction(int id) {
  switch (id) {
    case 0:
      return [](double x, double y) { return x + y; };
    case 1:
      return [](double x, double y) { return (x * x) + (y * y); };
    case 2:
      return [](double x, double y) { return std::sin(x) * std::cos(y); };
    case 3:
      return [](double x, double y) { return std::exp(x + y); };
    case 4:
      return [](double x, double y) { return std::sqrt((x * x) + (y * y)); };
    default:
      return [](double x, double y) {
        (void)x;
        (void)y;
        return 1.0;
      };
  }
}

inline double GetExactIntegral(const TaskData &data) {
  switch (data.func_id) {
    case 0:
      return (0.5 * ((data.x2 * data.x2) - (data.x1 * data.x1)) * (data.y2 - data.y1)) +
             (0.5 * ((data.y2 * data.y2) - (data.y1 * data.y1)) * (data.x2 - data.x1));
    case 1:
      return ((1.0 / 3.0) * ((data.x2 * data.x2 * data.x2) - (data.x1 * data.x1 * data.x1)) * (data.y2 - data.y1)) +
             ((1.0 / 3.0) * ((data.y2 * data.y2 * data.y2) - (data.y1 * data.y1 * data.y1)) * (data.x2 - data.x1));
    case 2:
      return (std::cos(data.x1) - std::cos(data.x2)) * (std::sin(data.y2) - std::sin(data.y1));
    case 3: {
      return (std::exp(data.x2) - std::exp(data.x1)) * (std::exp(data.y2) - std::exp(data.y1));
    }
    case 5: {
      return (data.x2 - data.x1) * (data.y2 - data.y1);
    }
    default:
      return 0.0;
  }
}

}  // namespace gasenin_l_mult_int_mstep_trapez
