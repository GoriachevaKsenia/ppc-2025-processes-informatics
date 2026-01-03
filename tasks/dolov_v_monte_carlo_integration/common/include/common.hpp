#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace dolov_v_monte_carlo_integration {

enum class IntegrationDomain : std::uint8_t { kHyperCube, kHyperSphere };

struct InputParams {
  std::function<double(const std::vector<double> &)> func;
  int dimension = 1;
  int samples_count = 0;
  std::vector<double> center;
  double radius = 1.0;
  IntegrationDomain domain_type = IntegrationDomain::kHyperCube;
};

using InType = InputParams;
using OutType = double;

using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline double FuncSumCoords(const std::vector<double> &x) {
  double sum = 0.0;
  for (double val : x) {
    sum += val;
  }
  return sum;
}

inline double FuncConstOne(const std::vector<double> & /*unused*/) {
  return 1.0;
}

}  // namespace dolov_v_monte_carlo_integration
