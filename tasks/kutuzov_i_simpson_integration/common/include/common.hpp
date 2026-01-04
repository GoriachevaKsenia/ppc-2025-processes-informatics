#pragma once

#include <cmath>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace kutuzov_i_simpson_integration {

// n, x_min-x_max, y_min-y_max, function_id
using InType = std::tuple<int, std::pair<double, double>, std::pair<double, double>, int>;
using OutType = double;
// n, x_min-x_max, y_min-y_max, function_id
using TestType = std::tuple<int, std::pair<double, double>, std::pair<double, double>, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline double FunctionPolynomial(double x, double y) {
  return pow(x, 3) + (0.5 * pow(x, 2)) + (3.0 * pow(y, 6)) + (15.0 * y) + 37.0;
}

inline double FunctionTrigonometric(double x, double y) {
  return pow(sin(x), 5) + (1.3 * cos(0.7 * x) * sin(1.4 * y)) - atan((15.0 * x) + (7.0 * y));
}

inline double FunctionExponents(double x, double y) {
  return (1.7 * exp(3.7 * x)) + (exp(3.0 * x * y) * log(pow(x + y, 2) + 1.0)) -
         log(pow((17.0 * x) - (8.0 * y), 4) + 0.1);
}

inline double FunctionComplex(double x, double y) {
  double sum = 0.0;
  for (int i = 1; i <= 200; i++) {
    double add = (sin((0.3 * pow(x * i, 4) * pow(y, 2)) + (0.5 * cos(y / i) * pow(x, 7)) + (1.8 * pow(y, 5))));
    if (i % 2 == 0) {
      sum += add;
    } else {
      sum -= add;
    }
  }
  return sum;
}

inline double CallFunction(int function_id, double x, double y) {
  switch (function_id) {
    case 1:
      return FunctionPolynomial(x, y);
    case 2:
      return FunctionTrigonometric(x, y);
    case 3:
      return FunctionExponents(x, y);
    case 4:
      return FunctionComplex(x, y);
    default:
      return 1.0;
  }
}

}  // namespace kutuzov_i_simpson_integration
