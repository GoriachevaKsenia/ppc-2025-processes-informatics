#pragma once

#include <cmath>
#include <string>
#include <tuple>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include "task/include/task.hpp"

namespace kruglova_a_2d_multistep_par_opt {

struct InType {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double eps;
  int max_iters;

  InType() : x_min(0.0), x_max(0.0), y_min(0.0), y_max(0.0), eps(0.0), max_iters(0) {}

  InType(double xmin, double xmax, double ymin, double ymax, double e, int iters)
      : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax), eps(e), max_iters(iters) {}
};

struct OutType {
  double x;
  double y;
  double f_value;

  OutType() : x(0.0), y(0.0), f_value(0.0) {}

  OutType(double x_val, double y_val, double f_val) : x(x_val), y(y_val), f_value(f_val) {}
};

struct Interval1D {
  double a, b;
  double f_a, f_b;
  double characteristic;
  int iteration;
};

struct Trial {
  double point;
  double value;
};

using TestType = std::tuple<std::string, InType>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline double ObjectiveFunction(double x, double y) {
  constexpr double kA = 10.0;
  constexpr double kN = 2.0;
  return (kA * kN) + (x * x) + (y * y) - (kA * (std::cos(2.0 * M_PI * x) + std::cos(2.0 * M_PI * y)));
}

}  // namespace kruglova_a_2d_multistep_par_opt
