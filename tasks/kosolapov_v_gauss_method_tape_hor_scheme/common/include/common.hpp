#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

struct LinSystem {
  std::vector<std::vector<double>> matrix;
  std::vector<double> r_side;
  LinSystem() = default;
  LinSystem(const LinSystem &other) = default;
  LinSystem &operator=(const LinSystem &other) {
    if (this != &other) {
      matrix = std::vector<std::vector<double>>(other.matrix);
      r_side = std::vector<double>(other.r_side);
    }
    return *this;
  }
};

using InType = LinSystem;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
