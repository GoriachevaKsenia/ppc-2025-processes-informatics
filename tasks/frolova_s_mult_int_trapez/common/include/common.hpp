#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace frolova_s_mult_int_trapez {

struct TrapezoidalIntegrationInput {
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> number_of_intervals;
  std::function<double(std::vector<double>)> function;
};

using InType = TrapezoidalIntegrationInput;
using OutType = double;
using TestType = std::tuple<unsigned int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace frolova_s_mult_int_trapez
