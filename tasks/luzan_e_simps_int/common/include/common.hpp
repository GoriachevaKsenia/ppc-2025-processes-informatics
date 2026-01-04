#pragma once

#include <cmath>
#include <tuple>

#include "task/include/task.hpp"

namespace luzan_e_simps_int {

// n, a-b, c-d, func_num
using InType = std::tuple<int, std::tuple<double, double>, std::tuple<double, double>, int>;
using OutType = double;
// n, a-b, c-d, func_num
using TestType = std::tuple<int, std::tuple<double, double>, std::tuple<double, double>, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

const double kEpsilon = 0.001;

inline double F1(double x, double y) {
  return (pow(x, 5) / 5.0) + (y * sin(y)) + 2.0;
}

inline double F2(double x, double y) {
  return (pow(x, 5) / 5.0) + (y * cos(y)) + 2.0;
}

inline double F3(double x, double y) {
  return (exp(-(x * x) - (y * y)) * sin(10 * x)) * cos(10 * y);
}

inline double F4(double x, double y) {
  double r = sqrt((x * x) + (y * y));
  return (exp(-r) * sin(20 * r)) + log(1 + (x * x) + (y * y));
}

inline double F5(double x, double y) {
  double s = 0.0;
  for (int k = 1; k <= 150; k++) {
    s += (sin(pow(x, 5)) * cos(k * y)) + (log(1 + (x * x) + (y * y)) * 5);
  }
  return s;
}

inline double GetWeight(int i, int n) {
  if (i == 0 || i == n) {
    return 1.0;
  }
  if (i % 2 == 1) {
    return 4.0;
  }
  return 2.0;
}

inline auto GetFunc(int num) {
  switch (num) {
    case 1:
      return &F1;
    case 2:
      return &F2;
    case 3:
      return &F3;
    case 4:
      return &F4;
    case 5:
      return &F5;
    default:
      return &F1;
  }
}

}  // namespace luzan_e_simps_int
